"""Tests for multinode engine-affinity passthrough (RouteIQ-bdd0 + RouteIQ-3316)."""

from __future__ import annotations

import pytest

from litellm_llmrouter.engine_affinity import (
    AFFINITY_KEY_HEADER,
    WORKER_INSTANCE_HEADER,
    apply_engine_affinity,
    build_affinity_headers,
    build_kv_transfer_params,
    multinode_affinity_enabled,
)


def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTEIQ_MULTINODE_AFFINITY_ENABLED", raising=False)
    assert multinode_affinity_enabled() is False
    assert build_affinity_headers(affinity_key="c1", worker_instance_id="w1") == {}
    assert build_kv_transfer_params(do_remote_prefill=True) == {}


def test_headers_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_MULTINODE_AFFINITY_ENABLED", "true")
    headers = build_affinity_headers(affinity_key="conv-1", worker_instance_id="dec-7")
    assert headers[WORKER_INSTANCE_HEADER] == "dec-7"
    assert headers[AFFINITY_KEY_HEADER] == "conv-1"


def test_headers_affinity_key_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_MULTINODE_AFFINITY_ENABLED", "true")
    headers = build_affinity_headers(affinity_key="conv-1")
    assert headers == {AFFINITY_KEY_HEADER: "conv-1"}


def test_kv_transfer_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_MULTINODE_AFFINITY_ENABLED", "true")
    params = build_kv_transfer_params(
        do_remote_prefill=True,
        do_remote_decode=True,
        kv_transfer_params={"engine_id": "e1"},
    )
    assert params == {
        "do_remote_prefill": True,
        "do_remote_decode": True,
        "kv_transfer_params": {"engine_id": "e1"},
    }


def test_kv_transfer_only_truthy_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_MULTINODE_AFFINITY_ENABLED", "true")
    # non-disagg request: no flags -> empty (byte-stable)
    assert build_kv_transfer_params() == {}
    assert build_kv_transfer_params(do_remote_prefill=True) == {
        "do_remote_prefill": True
    }


def test_apply_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTEIQ_MULTINODE_AFFINITY_ENABLED", raising=False)
    kwargs = {"model": "m", "messages": []}
    out = apply_engine_affinity(
        kwargs, affinity_key="c1", do_remote_decode=True, worker_instance_id="w1"
    )
    assert out == kwargs
    assert out is not kwargs  # always a copy


def test_apply_merges_headers_and_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_MULTINODE_AFFINITY_ENABLED", "true")
    kwargs = {"model": "m"}
    out = apply_engine_affinity(
        kwargs,
        affinity_key="conv-1",
        worker_instance_id="dec-7",
        do_remote_prefill=True,
        kv_transfer_params={"x": 1},
    )
    assert out["extra_headers"][WORKER_INSTANCE_HEADER] == "dec-7"
    assert out["extra_headers"][AFFINITY_KEY_HEADER] == "conv-1"
    assert out["do_remote_prefill"] is True
    assert out["kv_transfer_params"] == {"x": 1}


def test_apply_never_clobbers_caller_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_MULTINODE_AFFINITY_ENABLED", "true")
    kwargs = {
        "extra_headers": {WORKER_INSTANCE_HEADER: "caller-worker"},
        "do_remote_prefill": False,
    }
    out = apply_engine_affinity(
        kwargs, worker_instance_id="affinity-worker", do_remote_prefill=True
    )
    # caller-supplied header + param win
    assert out["extra_headers"][WORKER_INSTANCE_HEADER] == "caller-worker"
    assert out["do_remote_prefill"] is False
