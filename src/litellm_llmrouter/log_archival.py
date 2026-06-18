"""S3 archival of full request/response logs (RouteIQ-6702).

Streams full request/response payloads to an S3 bucket for long-term retention
(compliance / audit / offline eval corpus) while keeping the hot path's in-memory
/ Redis logging unchanged. Objects are written under a date-partitioned,
lifecycle-tiered key layout so an S3 lifecycle policy can transition older
partitions to Glacier / expire them.

Design (additive, gated, default-off, cred-free testable):

* DEFAULT-OFF. Disabled unless ``ROUTEIQ_LOG_ARCHIVAL_ENABLED=true``; the
  default deployment writes nothing to S3 and pays no boto3 cost.

* Date-partitioned, tier-friendly keys::

      <prefix>/dt=YYYY/MM/DD/HH/<request_id>.json

  The ``dt=YYYY/MM/DD/HH`` partition is the unit an S3 lifecycle rule transitions
  by tier (the ``storage_tier`` hint, default ``STANDARD``, is recorded in the
  object metadata so an operator's lifecycle policy + this module agree on the
  tiering intent). The module does NOT itself apply lifecycle (that is the
  bucket's CDK-managed lifecycle policy -- see deploy/cdk); it only writes the
  partitioned objects the policy acts on.

* Fail-safe + best-effort. A PUT failure logs and returns ``False`` -- archival
  is never allowed to break the request path.

* Redaction hook. Before serialization the payload is passed through an optional
  ``redactor`` callable so secret-bearing fields can be scrubbed.

The boto3 client is created lazily so importing this module has no AWS
dependency; unit tests inject a fake client without creds.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger("litellm_llmrouter.log_archival")

#: Recognised S3 storage tiers an operator may hint (recorded in object metadata).
VALID_STORAGE_TIERS = (
    "STANDARD",
    "STANDARD_IA",
    "INTELLIGENT_TIERING",
    "GLACIER",
    "DEEP_ARCHIVE",
)


def _archival_enabled() -> bool:
    """Whether request/response S3 archival is active (default OFF)."""
    return os.getenv("ROUTEIQ_LOG_ARCHIVAL_ENABLED", "false").lower() == "true"


class LogArchiver:
    """Archives full request/response payloads to S3 (gated, fail-safe).

    Disabled (no-op) unless ``ROUTEIQ_LOG_ARCHIVAL_ENABLED=true``. When enabled,
    requires ``ROUTEIQ_LOG_ARCHIVAL_BUCKET`` -- if the bucket is unset the
    archiver disables itself (logs once) so a half-configured deployment fails
    closed to no-op rather than erroring per request.
    """

    def __init__(
        self,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        region: Optional[str] = None,
        storage_tier: Optional[str] = None,
        redactor: Optional[Callable[[dict], dict]] = None,
    ) -> None:
        self._enabled = _archival_enabled()
        self._bucket = bucket or os.getenv("ROUTEIQ_LOG_ARCHIVAL_BUCKET", "")
        self._prefix = (
            prefix or os.getenv("ROUTEIQ_LOG_ARCHIVAL_PREFIX", "logs")
        ).strip("/")
        self._region = region or os.getenv("AWS_REGION", "us-east-1")
        tier = (
            storage_tier or os.getenv("ROUTEIQ_LOG_ARCHIVAL_TIER", "STANDARD")
        ).upper()
        self._storage_tier = tier if tier in VALID_STORAGE_TIERS else "STANDARD"
        self._redactor = redactor
        self._client: Any = None

        if self._enabled and not self._bucket:
            logger.warning(
                "Log archival enabled but ROUTEIQ_LOG_ARCHIVAL_BUCKET unset; "
                "disabling archival (no-op)."
            )
            self._enabled = False

    @property
    def enabled(self) -> bool:
        """True when archival is active (enabled + a bucket configured)."""
        return self._enabled

    @property
    def storage_tier(self) -> str:
        """The configured storage-tier hint recorded on archived objects."""
        return self._storage_tier

    def _get_client(self) -> Any:
        """Lazily build the boto3 S3 client (no creds at import time)."""
        if self._client is None:
            import boto3  # local import: optional dependency

            self._client = boto3.client("s3", region_name=self._region)
        return self._client

    def build_key(self, request_id: str, when: Optional[datetime] = None) -> str:
        """Build the date-partitioned, lifecycle-tier-friendly object key.

        ``<prefix>/dt=YYYY/MM/DD/HH/<request_id>.json`` -- the ``dt=`` partition
        is the unit an S3 lifecycle rule transitions / expires by.
        """
        ts = when or datetime.now(timezone.utc)
        partition = ts.strftime("dt=%Y/%m/%d/%H")
        safe_id = request_id.replace("/", "_") or "unknown"
        return f"{self._prefix}/{partition}/{safe_id}.json"

    def archive(
        self,
        request_id: str,
        request_payload: dict,
        response_payload: dict,
        *,
        metadata: Optional[dict] = None,
        when: Optional[datetime] = None,
    ) -> bool:
        """Archive one request/response pair to S3.

        No-op (returns ``False``) when disabled. On success returns ``True``.
        The body is the JSON of ``{request, response, metadata, archived_at}``;
        the optional ``redactor`` scrubs the combined record first.

        Never raises -- a PUT failure logs and returns ``False`` so the request
        path is unaffected.
        """
        if not self._enabled:
            return False

        ts = when or datetime.now(timezone.utc)
        record = {
            "request_id": request_id,
            "request": request_payload,
            "response": response_payload,
            "metadata": metadata or {},
            "archived_at": ts.isoformat(),
        }
        if self._redactor is not None:
            try:
                record = self._redactor(record)
            except Exception as exc:  # pragma: no cover - redactor is operator code
                logger.warning("Log archival: redactor failed: %s", exc)
                return False

        key = self.build_key(request_id, when=ts)
        try:
            self._get_client().put_object(
                Bucket=self._bucket,
                Key=key,
                Body=json.dumps(record, default=str).encode("utf-8"),
                ContentType="application/json",
                StorageClass=self._storage_tier,
                Metadata={"routeiq-tier": self._storage_tier},
            )
        except Exception as exc:
            logger.warning(
                "Log archival: PUT s3://%s/%s failed: %s", self._bucket, key, exc
            )
            return False
        logger.debug("Log archival: wrote s3://%s/%s", self._bucket, key)
        return True


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_archiver: Optional[LogArchiver] = None


def get_log_archiver() -> LogArchiver:
    """Get or create the global log archiver singleton."""
    global _archiver
    if _archiver is None:
        _archiver = LogArchiver()
    return _archiver


def reset_log_archiver() -> None:
    """Reset the log archiver singleton (for testing)."""
    global _archiver
    _archiver = None


# ---------------------------------------------------------------------------
# LiteLLM success-callback live callsite (RouteIQ-6702 wiring)
# ---------------------------------------------------------------------------


def _settings_archival_wiring_enabled() -> bool:
    """Whether the LIVE archival callback should be wired in.

    Requires BOTH ``settings.log_archival.enabled`` AND the explicit
    ``settings.log_archival.pii_acknowledged`` PII gate to be true. Any settings
    read failure degrades to ``False`` (default-off, byte-stable). This is the
    gate the app lifespan consults before registering the callback; the archiver
    itself additionally requires a configured bucket.
    """
    try:
        from litellm_llmrouter.settings import get_settings

        la = get_settings().log_archival
        return bool(
            getattr(la, "enabled", False) and getattr(la, "pii_acknowledged", False)
        )
    except Exception:  # pragma: no cover - defensive
        return False


def _custom_logger_base() -> type:
    """Resolve the litellm ``CustomLogger`` base lazily (no import side effects)."""
    try:
        from litellm.integrations.custom_logger import CustomLogger

        return CustomLogger
    except Exception:  # pragma: no cover - litellm always present in prod
        return object


class LogArchivalCallback(_custom_logger_base()):  # type: ignore[misc]
    """LiteLLM ``CustomLogger`` that archives each successful call to S3.

    Registered into ``litellm.callbacks`` only when archival is enabled +
    PII-acknowledged + a bucket is configured. On each success event it extracts
    a request id, the request kwargs, and the response payload and hands them to
    the :class:`LogArchiver`. Fail-safe: archival never raises into the hot path.
    """

    def __init__(self, archiver: Optional[LogArchiver] = None) -> None:
        try:
            super().__init__()
        except Exception:  # pragma: no cover - base may be ``object``
            pass
        self._archiver = archiver or get_log_archiver()

    @staticmethod
    def _request_id(kwargs: dict) -> str:
        meta = (kwargs.get("litellm_params") or {}).get("metadata") or {}
        return str(
            kwargs.get("litellm_call_id")
            or meta.get("request_id")
            or kwargs.get("id")
            or "unknown"
        )

    @staticmethod
    def _to_payload(obj: Any) -> dict:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        for attr in ("model_dump", "dict"):
            fn = getattr(obj, attr, None)
            if callable(fn):
                try:
                    out = fn()
                    if isinstance(out, dict):
                        return out
                except Exception:  # pragma: no cover - defensive
                    pass
        return {"repr": str(obj)}

    def _archive(self, kwargs: dict, response_obj: Any) -> None:
        if not self._archiver.enabled:
            return
        try:
            request_payload = {
                "model": kwargs.get("model", ""),
                "messages": kwargs.get("messages"),
                "input": kwargs.get("input"),
            }
            self._archiver.archive(
                self._request_id(kwargs),
                request_payload,
                self._to_payload(response_obj),
            )
        except Exception as exc:  # pragma: no cover - fail-safe
            logger.warning("Log archival callback failed: %s", exc)

    def log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self._archive(kwargs, response_obj)

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self._archive(kwargs, response_obj)


def register_log_archival_callback() -> Optional[LogArchivalCallback]:
    """Register the S3 archival success callback with LiteLLM (RouteIQ-6702).

    The LIVE callsite for the archiver: invoked from the app lifespan startup.
    No-op (returns ``None``) unless archival is enabled + PII-acknowledged
    (settings gate) AND the archiver has a configured bucket -- so a default
    deployment registers nothing, builds no boto3 client, and is byte-stable.
    Idempotent: an already-registered callback is reused.
    """
    if not _settings_archival_wiring_enabled():
        return None
    archiver = get_log_archiver()
    if not archiver.enabled:
        logger.warning(
            "Log archival enabled in settings but archiver inactive "
            "(no bucket?); skipping callback registration."
        )
        return None
    try:
        import litellm

        if not hasattr(litellm, "callbacks") or litellm.callbacks is None:
            litellm.callbacks = []
        for existing in litellm.callbacks:
            if isinstance(existing, LogArchivalCallback):
                return existing
        cb = LogArchivalCallback(archiver)
        litellm.callbacks.append(cb)
        logger.info("Registered LogArchivalCallback with LiteLLM (S3 archival ON)")
        return cb
    except ImportError:  # pragma: no cover - litellm always present in prod
        logger.warning("LiteLLM not available; cannot register log archival callback")
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to register log archival callback: %s", exc)
        return None


def unregister_log_archival_callback() -> None:
    """Remove the archival callback from ``litellm.callbacks`` (shutdown/test)."""
    try:
        import litellm

        callbacks = getattr(litellm, "callbacks", None)
        if not callbacks:
            return
        litellm.callbacks = [
            cb for cb in callbacks if not isinstance(cb, LogArchivalCallback)
        ]
    except Exception:  # pragma: no cover - defensive
        pass


__all__ = [
    "VALID_STORAGE_TIERS",
    "LogArchiver",
    "LogArchivalCallback",
    "get_log_archiver",
    "reset_log_archiver",
    "register_log_archival_callback",
    "unregister_log_archival_callback",
]
