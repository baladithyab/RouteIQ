"""Unit tests for the RouteIQ schema-bootstrap Lambda's SQL hardening (RouteIQ-4d7c).

The bootstrap Lambda interpolates the runtime DB user (a fixed deploy-time
construct value, NOT request-derived) into DDL because asyncpg cannot
parameterise an identifier. That is not a live SQLi vector, but it is hardened:
``_quote_identifier`` validates the name against the plain unquoted-identifier
charset and double-quotes it, so an unusual user string can never malform the
DDL and a malformed value fails CLOSED (raises) at the bootstrap.

The handler is loaded via ``importlib`` from its on-disk path (it is a Lambda
asset, not an installed package). Its ``asyncpg`` / ``boto3`` imports are lazy
(inside the functions that use them), so the module imports cleanly in the
gateway test venv that lacks those AWS deps -- which is what makes this
pure-python test runnable here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_HANDLER_PATH = (
    Path(__file__).resolve().parents[2]
    / "deploy"
    / "cdk"
    / "lambda"
    / "routeiq-schema-bootstrap"
    / "handler.py"
)


def _load_handler() -> ModuleType:
    """Import the Lambda handler from its asset path (not on sys.path)."""
    spec = importlib.util.spec_from_file_location(
        "routeiq_schema_bootstrap_handler", _HANDLER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def handler() -> ModuleType:
    return _load_handler()


def test_handler_file_present() -> None:
    assert _HANDLER_PATH.is_file(), _HANDLER_PATH


def test_handler_imports_without_aws_deps(handler: ModuleType) -> None:
    """The module imports in a venv lacking asyncpg/boto3 (lazy imports)."""
    assert hasattr(handler, "_quote_identifier")
    assert hasattr(handler, "lambda_handler")


def test_normal_runtime_user_is_double_quoted(handler: ModuleType) -> None:
    """The default ``routeiq`` user (and other normal names) quote cleanly."""
    assert handler._quote_identifier("routeiq") == '"routeiq"'
    assert handler._quote_identifier("routeiq_app") == '"routeiq_app"'
    assert handler._quote_identifier("_RouteIQ2") == '"_RouteIQ2"'


@pytest.mark.parametrize(
    "malformed",
    [
        "routeiq; DROP ROLE admin",  # statement break-out
        'routeiq" WITH SUPERUSER --',  # embedded double-quote
        "routeiq'--",  # embedded single-quote (DO-block literal break-out)
        "route iq",  # whitespace
        "1routeiq",  # leading digit (not a valid unquoted identifier)
        "route-iq",  # hyphen
        "",  # empty
        "x" * 64,  # exceeds NAMEDATALEN-1 (63)
    ],
)
def test_malformed_identifier_is_rejected(handler: ModuleType, malformed: str) -> None:
    """A malformed identifier fails CLOSED (ValueError), never emits bad DDL."""
    with pytest.raises(ValueError):
        handler._quote_identifier(malformed)


def test_non_string_identifier_is_rejected(handler: ModuleType) -> None:
    with pytest.raises(ValueError):
        handler._quote_identifier(None)  # type: ignore[arg-type]


def test_quoted_identifier_cannot_break_out_of_ddl(handler: ModuleType) -> None:
    """Any accepted identifier is a single double-quoted token with no breakers."""
    quoted = handler._quote_identifier("routeiq")
    inner = quoted[1:-1]
    assert quoted.startswith('"') and quoted.endswith('"')
    # No characters that could terminate the statement or the quoted literal.
    for forbidden in ('"', "'", ";", " ", "\n", "-"):
        assert forbidden not in inner


def test_delete_request_is_noop(handler: ModuleType) -> None:
    """Delete must not touch the DB (so it needs neither asyncpg nor boto3)."""
    result = handler.lambda_handler({"RequestType": "Delete"}, object())
    assert result == {"PhysicalResourceId": "routeiq-schema-bootstrap"}
