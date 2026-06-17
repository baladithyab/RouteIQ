"""Tests for S3 archival of full request/response logs (RouteIQ-6702).

Cred-free: a fake S3 client records put_object. Covers the default-off no-op,
the date-partitioned tier-friendly key, the enabled archive write (with storage
class + redaction), the fail-closed-when-no-bucket guard, and the PUT-failure
fail-safe.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from litellm_llmrouter.log_archival import LogArchiver, reset_log_archiver


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_log_archiver()
    yield
    reset_log_archiver()


def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTEIQ_LOG_ARCHIVAL_ENABLED", raising=False)
    arch = LogArchiver()
    assert arch.enabled is False
    assert arch.archive("req-1", {"m": "x"}, {"r": "y"}) is False


def test_enabled_but_no_bucket_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_LOG_ARCHIVAL_ENABLED", "true")
    monkeypatch.delenv("ROUTEIQ_LOG_ARCHIVAL_BUCKET", raising=False)
    arch = LogArchiver()
    assert arch.enabled is False  # fail-closed to no-op


def test_build_key_is_date_partitioned() -> None:
    arch = LogArchiver(bucket="b", prefix="logs")
    when = datetime(2026, 6, 17, 9, 5, tzinfo=timezone.utc)
    key = arch.build_key("req-42", when=when)
    assert key == "logs/dt=2026/06/17/09/req-42.json"


def test_storage_tier_normalised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_LOG_ARCHIVAL_ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_LOG_ARCHIVAL_BUCKET", "b")
    monkeypatch.setenv("ROUTEIQ_LOG_ARCHIVAL_TIER", "glacier")
    arch = LogArchiver()
    assert arch.storage_tier == "GLACIER"


def test_invalid_tier_falls_back_to_standard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_LOG_ARCHIVAL_ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_LOG_ARCHIVAL_BUCKET", "b")
    monkeypatch.setenv("ROUTEIQ_LOG_ARCHIVAL_TIER", "NONSENSE")
    arch = LogArchiver()
    assert arch.storage_tier == "STANDARD"


def test_archive_writes_object(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_LOG_ARCHIVAL_ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_LOG_ARCHIVAL_BUCKET", "my-bucket")
    client = MagicMock()
    arch = LogArchiver(prefix="logs", storage_tier="STANDARD_IA")
    when = datetime(2026, 6, 17, 9, 0, tzinfo=timezone.utc)
    with patch.object(arch, "_get_client", return_value=client):
        ok = arch.archive(
            "req-1",
            {"model": "gpt"},
            {"text": "hi"},
            metadata={"ws": "w1"},
            when=when,
        )
    assert ok is True
    call = client.put_object.call_args.kwargs
    assert call["Bucket"] == "my-bucket"
    assert call["Key"] == "logs/dt=2026/06/17/09/req-1.json"
    assert call["StorageClass"] == "STANDARD_IA"
    body = json.loads(call["Body"].decode("utf-8"))
    assert body["request"] == {"model": "gpt"}
    assert body["response"] == {"text": "hi"}
    assert body["metadata"] == {"ws": "w1"}


def test_archive_applies_redactor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_LOG_ARCHIVAL_ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_LOG_ARCHIVAL_BUCKET", "b")
    client = MagicMock()

    def _redactor(record: dict) -> dict:
        record["request"] = {"redacted": True}
        return record

    arch = LogArchiver(redactor=_redactor)
    with patch.object(arch, "_get_client", return_value=client):
        arch.archive("r", {"secret": "x"}, {})
    body = json.loads(client.put_object.call_args.kwargs["Body"].decode("utf-8"))
    assert body["request"] == {"redacted": True}


def test_archive_put_failure_is_failsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROUTEIQ_LOG_ARCHIVAL_ENABLED", "true")
    monkeypatch.setenv("ROUTEIQ_LOG_ARCHIVAL_BUCKET", "b")
    client = MagicMock()
    client.put_object.side_effect = RuntimeError("503 SlowDown")
    arch = LogArchiver()
    with patch.object(arch, "_get_client", return_value=client):
        assert arch.archive("r", {}, {}) is False
