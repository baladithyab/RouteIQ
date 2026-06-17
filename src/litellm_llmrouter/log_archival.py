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


__all__ = [
    "VALID_STORAGE_TIERS",
    "LogArchiver",
    "get_log_archiver",
    "reset_log_archiver",
]
