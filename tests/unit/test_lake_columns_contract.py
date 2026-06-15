"""Cross-cutting CONTRACT test for the routing_decision lake column schema (RouteIQ-199e).

The 14-key flat-scalar lake schema is authored in TWO places that must stay
byte-for-byte in sync, because nothing imports across the seam at runtime:

  * the gateway emitter side -- ``litellm_llmrouter.observability.ROUTING_DECISION_COLUMNS``
    (the tuple the app writes at the TOP LEVEL of each JSON log line); and
  * the infra side -- ``deploy/cdk/lib/data_lake_construct.py`` ``_COLUMNS`` (the
    Glue table's identity-SerDe column list).

OpenXJsonSerDe extracts the Parquet columns BY IDENTITY NAME (no column->json-key
mapping) AND order matters for the Glue table definition, so a one-sided edit
(rename or reorder a column on EITHER side) would silently drop those Parquet
columns to NULL with no runtime error. The CDK subtree and the gateway package
cannot import each other (separate dependency closures: the CDK side needs
``aws_cdk``, which the gateway test venv lacks), so this test bridges them the
credential-free way: it imports ``ROUTING_DECISION_COLUMNS`` directly and reads
the ``_COLUMNS`` literal out of the CDK source file via the ``ast`` module
(static parse -- no ``aws_cdk`` import, no synth, no AWS creds), then asserts the
two are identical in NAMES *and* ORDER.

A negative-control test proves the assertion is load-bearing: it shows the same
comparison would FAIL if either side drifted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from litellm_llmrouter.observability import ROUTING_DECISION_COLUMNS

# This test lives at <repo>/tests/unit/, so the CDK construct is two parents up.
# Resolve relative to THIS file (not the installed package) so the test reads the
# CDK file from the SAME checkout it itself was loaded from.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_LAKE_CONSTRUCT = _REPO_ROOT / "deploy" / "cdk" / "lib" / "data_lake_construct.py"


def _extract_cdk_column_names(source_path: Path) -> list[str]:
    """Statically parse ``_COLUMNS`` out of the CDK construct, returning the names.

    ``_COLUMNS`` is a module-level ``list[tuple[str, str]]`` of ``(name, glue_type)``
    pairs; we want the names in declaration order. Done with ``ast`` so the CDK
    file is read without importing ``aws_cdk`` (which the gateway test venv lacks)
    and without any AWS credentials.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        # ``_COLUMNS`` is annotated (``_COLUMNS: list[...] = [...]``) -> AnnAssign;
        # tolerate a bare Assign too in case the annotation is ever dropped.
        if isinstance(node, ast.AnnAssign):
            target = node.target
            is_columns = isinstance(target, ast.Name) and target.id == "_COLUMNS"
            value = node.value
        elif isinstance(node, ast.Assign):
            names_assigned = [t.id for t in node.targets if isinstance(t, ast.Name)]
            is_columns = "_COLUMNS" in names_assigned
            value = node.value
        else:
            continue
        if not is_columns:
            continue
        if not isinstance(value, ast.List):
            raise AssertionError(
                "_COLUMNS in data_lake_construct.py is not a list literal"
            )
        names: list[str] = []
        for element in value.elts:
            if not isinstance(element, ast.Tuple) or not element.elts:
                raise AssertionError(
                    "_COLUMNS element is not a (name, type) tuple literal"
                )
            name_node = element.elts[0]
            if not isinstance(name_node, ast.Constant) or not isinstance(
                name_node.value, str
            ):
                raise AssertionError("_COLUMNS column name is not a string literal")
            names.append(name_node.value)
        return names
    raise AssertionError("_COLUMNS not found in data_lake_construct.py")


def test_cdk_construct_source_is_present() -> None:
    """The CDK construct file must be locatable from the test's own checkout."""
    assert _DATA_LAKE_CONSTRUCT.is_file(), (
        f"expected the CDK data-lake construct at {_DATA_LAKE_CONSTRUCT}; the "
        "lake column contract cannot be verified without it"
    )


def test_lake_columns_match_in_names_and_order() -> None:
    """observability.ROUTING_DECISION_COLUMNS == data_lake_construct._COLUMNS names.

    Equality in BOTH names and order. A drift on either side (the documented
    silent-null footgun) fails here.
    """
    cdk_names = _extract_cdk_column_names(_DATA_LAKE_CONSTRUCT)
    gateway_names = list(ROUTING_DECISION_COLUMNS)
    assert cdk_names == gateway_names, (
        "routing_decision lake column schema drifted between "
        "observability.ROUTING_DECISION_COLUMNS and "
        "data_lake_construct._COLUMNS -- they must be identical in names AND "
        f"order.\n  gateway: {gateway_names}\n  cdk:     {cdk_names}"
    )


def test_lake_columns_are_the_frozen_14() -> None:
    """Lock the count so neither side can add/drop a column unnoticed."""
    assert len(ROUTING_DECISION_COLUMNS) == 14
    assert len(_extract_cdk_column_names(_DATA_LAKE_CONSTRUCT)) == 14


@pytest.mark.parametrize(
    "drifted",
    [
        pytest.param(
            lambda names: names[:-1] + ["renamed_column"], id="rename-last-column"
        ),
        pytest.param(
            lambda names: [names[1], names[0], *names[2:]], id="swap-first-two-order"
        ),
        pytest.param(lambda names: names[:-1], id="drop-a-column"),
        pytest.param(lambda names: [*names, "extra_column"], id="add-a-column"),
    ],
)
def test_contract_would_fail_on_drift(drifted) -> None:
    """Negative control: the name+order equality WOULD fail if either side drifted.

    Proves the contract assertion is load-bearing rather than vacuously true --
    a rename, a reorder, a drop, or an addition each breaks equality.
    """
    cdk_names = _extract_cdk_column_names(_DATA_LAKE_CONSTRUCT)
    gateway_names = list(ROUTING_DECISION_COLUMNS)
    assert cdk_names == gateway_names  # sanity: in sync before we mutate
    assert drifted(cdk_names) != gateway_names
