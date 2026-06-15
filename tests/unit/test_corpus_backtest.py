"""Offline corpus loader + backtest validation against the real preserved corpus.

De-risks the bandit math against the 241-row corpus OFFLINE (no DB, no live cache).
Skips gracefully if the corpus path is absent so CI stays portable. The backtest
imports ONLY the pure posterior primitives from the package.

Honest assertions, matching the documented corpus caveats (discover-corpus.md §7):
the reward is a CONSTRUCTED cost-efficiency proxy, the log is single-action with
one-sided positive cost-savings, and pseudo-regret is vs running empirical means.
We therefore assert posterior-fit convergence and finite/bounded regret, NOT a
strict sublinear-regret claim (which this exploration-free log cannot guarantee).
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

# The backtest tooling lives in the COMMITTED tree at
# ``scripts/p3_bandit_backtest/`` (NOT gitignored research/), so a fresh clone /
# CI can import it. If the package is somehow absent we skip the whole module
# rather than abort collection (a module-level import error would break the
# entire tests/unit/ run — see seed RouteIQ test-collection guard).
_BACKTEST_PKG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "p3_bandit_backtest")
)


def _load_module(name: str):
    """Import a backtest-tool module by path under a UNIQUELY-NAMESPACED name.

    Registered as ``_p3bt_<name>`` in ``sys.modules`` (not the bare
    ``corpus_loader``/``backtest``) to avoid the sys-modules-stub-leak hazard of
    shadowing a future generically-named module. ``backtest.py`` imports
    ``corpus_loader`` by bare name, so we also expose that alias on ``sys.path``
    via the package dir, scoped to this load.
    """
    namespaced = f"_p3bt_{name}"
    if namespaced in sys.modules:
        return sys.modules[namespaced]
    if _BACKTEST_PKG not in sys.path:
        sys.path.insert(0, _BACKTEST_PKG)
    path = os.path.join(_BACKTEST_PKG, f"{name}.py")
    spec = importlib.util.spec_from_file_location(namespaced, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # backtest.py does `from corpus_loader import ...` by bare name — satisfy it.
    sys.modules.setdefault(
        name, module if name == "corpus_loader" else sys.modules.get(name)
    )
    sys.modules[namespaced] = module
    spec.loader.exec_module(module)
    return module


try:
    corpus_loader = _load_module("corpus_loader")
    backtest = _load_module("backtest")
    _IMPORT_OK = True
except Exception as _exc:  # pragma: no cover - defensive: never abort collection
    corpus_loader = backtest = None
    _IMPORT_OK = False
    pytest.skip(
        f"p3_bandit_backtest tooling not importable ({_exc!r}); skipping backtest tests",
        allow_module_level=True,
    )

_CSV_PATH = corpus_loader.DEFAULT_CSV_PATH
_corpus_present = os.path.exists(_CSV_PATH)
_skip_no_corpus = pytest.mark.skipif(
    not _corpus_present, reason=f"preserved corpus not found at {_CSV_PATH}"
)


# ===========================================================================
# Loader correctness
# ===========================================================================


@_skip_no_corpus
def test_load_csv_row_count():
    rows = corpus_loader.load_csv(_CSV_PATH)
    assert len(rows) == 241


@_skip_no_corpus
def test_reward_in_unit_interval():
    rows = corpus_loader.load_csv(_CSV_PATH)
    for row in rows:
        assert 0.0 <= row.reward_costmin <= 1.0


@_skip_no_corpus
def test_failed_rows_have_zero_served():
    rows = corpus_loader.load_csv(_CSV_PATH)
    # 4 rows have empty response bodies + a couple of failed/jailbreak rows.
    served = sum(r.served for r in rows)
    assert served == 225  # per discover-corpus.md fill-rate analysis
    # Any unserved row contributes 0 reward (gated on served).
    for row in rows:
        if row.served == 0:
            assert row.reward_costmin == 0.0


@_skip_no_corpus
def test_distinct_arm_count():
    rows = corpus_loader.load_csv(_CSV_PATH)
    arms = {r.selected_model for r in rows}
    assert len(arms) == 31


def test_size_bucket_tiers():
    assert corpus_loader.size_bucket("gemma-3-4b") == "small"
    assert corpus_loader.size_bucket("qwen3-32b") == "mid"
    assert corpus_loader.size_bucket("qwen3-coder-480b") == "large"
    assert corpus_loader.size_bucket("nemotron-nano-9b") == "small"


# ===========================================================================
# Backtest — Mode A (posterior-fit convergence)
# ===========================================================================


@_skip_no_corpus
def test_mode_a_bucket_posterior_converges():
    report = backtest.run(_CSV_PATH)
    # Bucketed posteriors wash out the Beta(1,1) prior: every bucket's posterior
    # mean must track its empirical reward mean within tolerance.
    assert report.mode_a_buckets_checked >= 1
    assert report.mode_a_buckets_within_tol == report.mode_a_buckets_checked
    assert report.mode_a_max_abs_error < 0.05


@_skip_no_corpus
def test_mode_a_at_least_one_well_pulled_arm_close():
    report = backtest.run(_CSV_PATH)
    # Raw 31-arm fit is sparser (larger Laplace error expected), but at least the
    # diagnostic must report some well-pulled arms.
    assert report.mode_a_raw_well_pulled_total >= 1


# ===========================================================================
# Backtest — Mode B (Thompson agreement + bounded pseudo-regret)
# ===========================================================================


@_skip_no_corpus
def test_mode_b_regret_finite_and_bounded():
    report = backtest.run(_CSV_PATH, seed=1234)
    # Pseudo-regret is finite and bounded (NOT asserted sublinear: this is a
    # single-action, exploration-free, one-sided-reward log — see caveats).
    assert report.mode_b_pseudo_regret >= 0.0
    assert report.mode_b_pseudo_regret < report.n_rows  # < 1 regret/step on avg
    assert 0.0 <= report.mode_b_agreement_acc <= 1.0


@_skip_no_corpus
def test_mode_b_mean_reward_positive():
    report = backtest.run(_CSV_PATH, seed=1234)
    # Constructed cost-efficiency reward is one-sided positive across served rows.
    assert report.mode_b_thompson_mean_reward > 0.0


@_skip_no_corpus
def test_backtest_deterministic_under_seed():
    r1 = backtest.run(_CSV_PATH, seed=7)
    r2 = backtest.run(_CSV_PATH, seed=7)
    assert r1.mode_b_agreement_acc == r2.mode_b_agreement_acc
    assert r1.mode_b_pseudo_regret == r2.mode_b_pseudo_regret


@_skip_no_corpus
def test_report_serializable():
    report = backtest.run(_CSV_PATH)
    d = report.to_dict()
    import json

    json.dumps(d)  # must not raise
    assert d["n_rows"] == 241


def test_loader_importable_without_corpus():
    # The loader module imports with stdlib only — no RouteIQ package, no DB.
    assert hasattr(corpus_loader, "load_csv")
    assert hasattr(corpus_loader, "Row")
