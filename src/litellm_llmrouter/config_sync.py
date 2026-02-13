"""
Configuration Sync and Hot Reload
==================================

Background sync from S3/GCS with file watching for hot reload.
Uses ETag-based caching to avoid unnecessary downloads.

In HA deployments with multiple replicas, leader election is used
to ensure only one replica performs the sync at a time, avoiding
thundering herd problems and conflicting updates.
"""

import dataclasses
import hashlib
import os
import signal
import threading
from pathlib import Path
from typing import Any, Callable

from litellm._logging import verbose_proxy_logger


@dataclasses.dataclass
class ConfigDiffResult:
    """Result of diffing two model config lists."""

    added: list[dict[str, Any]]
    removed: list[dict[str, Any]]
    changed: list[dict[str, Any]]


def diff_model_configs(
    old_models: list[dict[str, Any]],
    new_models: list[dict[str, Any]],
) -> ConfigDiffResult:
    """Diff two model config lists by model_name.

    Models are matched by model_name. A model is "changed" if its
    serialized dict differs but the name is the same.
    """
    old_by_name = {m.get("model_name"): m for m in old_models}
    new_by_name = {m.get("model_name"): m for m in new_models}

    old_names = set(old_by_name.keys())
    new_names = set(new_by_name.keys())

    added = [new_by_name[n] for n in sorted(new_names - old_names)]
    removed = [old_by_name[n] for n in sorted(old_names - new_names)]
    changed = [
        new_by_name[n]
        for n in sorted(old_names & new_names)
        if old_by_name[n] != new_by_name[n]
    ]

    return ConfigDiffResult(added=added, removed=removed, changed=changed)


@dataclasses.dataclass
class ConfigSyncStatus:
    """Status of the config sync system."""

    config_source: str | None
    sync_enabled: bool
    sync_interval_seconds: int
    last_sync_attempt: str | None
    last_sync_success: str | None
    last_sync_error: str | None
    config_version_hash: str | None
    model_count: int
    next_sync_at: str | None


def get_config_sync_status() -> ConfigSyncStatus:
    """Get current config sync status."""
    manager = get_sync_manager()
    source = None
    if manager.s3_sync_enabled:
        source = f"s3://{manager.s3_bucket}/{manager.s3_key}"
    elif manager.gcs_sync_enabled:
        source = f"gs://{manager.gcs_bucket}/{manager.gcs_key}"

    config_hash = None
    if manager._last_config_hash:
        config_hash = f"sha256:{manager._last_config_hash[:16]}"

    return ConfigSyncStatus(
        config_source=source,
        sync_enabled=manager.sync_enabled,
        sync_interval_seconds=manager.sync_interval,
        last_sync_attempt=None,
        last_sync_success=None,
        last_sync_error=None,
        config_version_hash=config_hash,
        model_count=len(getattr(manager, "_current_model_configs", [])),
        next_sync_at=None,
    )


class ConfigSyncManager:
    """Manages config synchronization from S3/GCS with hot reload support.

    Uses ETag-based change detection to minimize bandwidth and only
    download when remote config has actually changed.

    In HA mode with leader election enabled, only the leader replica
    performs the actual sync. Non-leaders skip quietly.
    """

    def __init__(
        self,
        local_config_path: str = "/app/config/config.yaml",
        sync_interval_seconds: int = 60,
        on_config_changed: Callable[[], None] | None = None,
    ):
        self.local_config_path = Path(local_config_path)
        self.sync_interval = sync_interval_seconds
        self.on_config_changed = on_config_changed
        self._last_config_hash: str | None = None
        self._last_s3_etag: str | None = None
        self._last_gcs_etag: str | None = None
        self._stop_event = threading.Event()
        self._sync_thread: threading.Thread | None = None
        self._reload_count = 0
        self._last_sync_time: float | None = None
        self._skipped_sync_count = 0  # Track skipped syncs (non-leader)

        # S3 config
        self.s3_bucket = os.getenv("CONFIG_S3_BUCKET")
        self.s3_key = os.getenv("CONFIG_S3_KEY")
        self.s3_sync_enabled = bool(self.s3_bucket and self.s3_key)

        # GCS config
        self.gcs_bucket = os.getenv("CONFIG_GCS_BUCKET")
        self.gcs_key = os.getenv("CONFIG_GCS_KEY")
        self.gcs_sync_enabled = bool(self.gcs_bucket and self.gcs_key)

        # Hot reload settings
        self.hot_reload_enabled = (
            os.getenv("CONFIG_HOT_RELOAD", "false").lower() == "true"
        )
        self.sync_enabled = os.getenv("CONFIG_SYNC_ENABLED", "true").lower() == "true"

        # Incremental reload tracking
        self._current_model_configs: list[dict[str, Any]] = []

        # Leader election (optional, for HA deployments)
        self._leader_election = None
        self._leader_election_enabled = False
        self._initialize_leader_election()

    def _initialize_leader_election(self):
        """Initialize leader election if enabled."""
        try:
            from litellm_llmrouter.leader_election import (
                get_leader_election,
                get_leader_election_config,
            )

            config = get_leader_election_config()
            self._ha_mode = config.get("ha_mode", "single")
            self._leader_election_enabled = config["enabled"]

            if self._leader_election_enabled:
                self._leader_election = get_leader_election()
                verbose_proxy_logger.info(
                    f"Config sync: HA mode '{self._ha_mode}', leader election enabled"
                )
            else:
                verbose_proxy_logger.debug(
                    f"Config sync: HA mode '{self._ha_mode}', leader election disabled"
                )

        except ImportError as e:
            verbose_proxy_logger.warning(
                f"Config sync: Leader election not available: {e}"
            )
            self._ha_mode = "single"
            self._leader_election_enabled = False

    def _is_leader(self) -> bool:
        """
        Check if this instance is the leader (or if leader election is disabled).

        Returns:
            True if this instance should perform sync, False otherwise
        """
        # If leader election is not enabled, always perform sync
        if not self._leader_election_enabled or self._leader_election is None:
            return True

        return self._leader_election.is_leader

    def _compute_file_hash(self, path: Path) -> str | None:
        """Compute MD5 hash of a file."""
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None

    def _get_s3_etag(self) -> str | None:
        """Get the current ETag of the S3 object without downloading."""
        if not self.s3_sync_enabled:
            return None
        try:
            import boto3

            s3_client = boto3.client("s3")
            response = s3_client.head_object(Bucket=self.s3_bucket, Key=self.s3_key)
            return str(response.get("ETag", "")).strip('"')
        except Exception as e:
            verbose_proxy_logger.warning(f"Failed to get S3 ETag: {e}")
            return None

    def _download_from_s3_if_changed(self) -> bool:
        """Download config from S3 only if ETag has changed.

        Returns True if config was downloaded and is different from before.
        """
        if not self.s3_sync_enabled:
            return False

        try:
            current_etag = self._get_s3_etag()
            if current_etag is None:
                return False

            # Skip download if ETag hasn't changed
            if current_etag == self._last_s3_etag:
                verbose_proxy_logger.debug(
                    f"S3 config unchanged (ETag: {current_etag[:8]}...)"
                )
                return False

            import boto3

            s3_client = boto3.client("s3")
            self.local_config_path.parent.mkdir(parents=True, exist_ok=True)

            # Compute hash before download
            old_hash = self._compute_file_hash(self.local_config_path)

            # Download the file
            s3_client.download_file(
                self.s3_bucket, self.s3_key, str(self.local_config_path)
            )

            # Update cached ETag
            self._last_s3_etag = current_etag

            # Check if content actually changed
            new_hash = self._compute_file_hash(self.local_config_path)
            if old_hash != new_hash:
                verbose_proxy_logger.info(
                    f"Config synced from s3://{self.s3_bucket}/{self.s3_key} "
                    f"(ETag: {current_etag[:8]}...)"
                )
                return True
            return False

        except Exception as e:
            verbose_proxy_logger.error(f"Failed to sync config from S3: {e}")
            return False

    def _sync_loop(self):
        """Background sync loop with ETag-based change detection."""
        import time

        verbose_proxy_logger.info(
            f"Config sync started (interval: {self.sync_interval}s, "
            f"hot_reload: {self.hot_reload_enabled}, "
            f"leader_election: {self._leader_election_enabled})"
        )

        while not self._stop_event.is_set():
            try:
                self._last_sync_time = time.time()

                # Check if we are the leader before syncing
                if not self._is_leader():
                    self._skipped_sync_count += 1
                    if self._skipped_sync_count % 10 == 1:  # Log every 10th skip
                        verbose_proxy_logger.debug(
                            f"Config sync: Skipping (not leader, skipped={self._skipped_sync_count})"
                        )
                else:
                    # Reset skipped count when we become leader
                    self._skipped_sync_count = 0

                    # Check S3 for updates using ETag
                    if self.s3_sync_enabled:
                        if (
                            self._download_from_s3_if_changed()
                            and self.hot_reload_enabled
                        ):
                            verbose_proxy_logger.info(
                                "Config changed, triggering reload..."
                            )
                            self._trigger_reload()
                            self._reload_count += 1

            except Exception as e:
                verbose_proxy_logger.error(f"Config sync error: {e}")

            # Wait for next sync interval
            self._stop_event.wait(self.sync_interval)

        verbose_proxy_logger.info("Config sync stopped")

    def _trigger_reload(self):
        """Trigger config reload by sending SIGHUP."""
        if self.on_config_changed:
            self.on_config_changed()
        else:
            # Send SIGHUP to trigger LiteLLM's built-in config reload
            try:
                os.kill(os.getpid(), signal.SIGHUP)
            except Exception as e:
                verbose_proxy_logger.error(f"Failed to signal reload: {e}")

    def start(self):
        """Start the background sync thread."""
        # Always initialize leader election if enabled, even if remote sync is disabled
        # This ensures HA mode works for other features beyond config sync
        if self._leader_election_enabled and self._leader_election is not None:
            # Initialize leader election table and try initial acquisition
            import asyncio

            try:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(self._leader_election.ensure_table_exists())
                    loop.run_until_complete(self._leader_election.try_acquire())
                finally:
                    loop.close()
                verbose_proxy_logger.info(
                    f"Config sync: Leader election initialized "
                    f"(is_leader={self._leader_election.is_leader})"
                )
            except Exception as e:
                verbose_proxy_logger.warning(
                    f"Config sync: Leader election init error: {e}"
                )

            # Start background lease renewal
            self._leader_election.start_renewal(
                on_leadership_change=self._on_leadership_change
            )

        if not self.sync_enabled or (
            not self.s3_sync_enabled and not self.gcs_sync_enabled
        ):
            verbose_proxy_logger.info(
                "Config sync disabled or no remote config configured"
            )
            return

        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

    def _on_leadership_change(self, is_leader: bool):
        """Callback when leadership status changes."""
        if is_leader:
            verbose_proxy_logger.info("Config sync: Became leader, will sync")
        else:
            verbose_proxy_logger.info("Config sync: Lost leadership, will skip sync")

    def stop(self):
        """Stop the background sync."""
        self._stop_event.set()

        # Stop leader election renewal
        if self._leader_election_enabled and self._leader_election is not None:
            self._leader_election.stop_renewal()

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5)

    def force_sync(self) -> bool:
        """Force an immediate sync from remote storage."""
        if self.s3_sync_enabled:
            return self._download_from_s3_if_changed()
        return False

    def get_status(self) -> dict:
        """Get the current sync status."""
        status = {
            "enabled": self.sync_enabled,
            "hot_reload_enabled": self.hot_reload_enabled,
            "sync_interval_seconds": self.sync_interval,
            "s3": (
                {
                    "enabled": self.s3_sync_enabled,
                    "bucket": self.s3_bucket,
                    "key": self.s3_key,
                    "last_etag": self._last_s3_etag,
                }
                if self.s3_sync_enabled
                else None
            ),
            "gcs": (
                {
                    "enabled": self.gcs_sync_enabled,
                    "bucket": self.gcs_bucket,
                    "key": self.gcs_key,
                }
                if self.gcs_sync_enabled
                else None
            ),
            "local_config_path": str(self.local_config_path),
            "local_config_hash": self._compute_file_hash(self.local_config_path),
            "reload_count": self._reload_count,
            "last_sync_time": self._last_sync_time,
            "running": self._sync_thread is not None and self._sync_thread.is_alive(),
            "ha_mode": getattr(self, "_ha_mode", "single"),
            "leader_election": {
                "enabled": self._leader_election_enabled,
                "is_leader": self._is_leader(),
                "skipped_sync_count": self._skipped_sync_count,
            },
        }

        # Add detailed leader election status if available
        if self._leader_election_enabled and self._leader_election is not None:
            status["leader_election"].update(self._leader_election.get_status())

        return status

    def _compute_reload_plan(
        self, new_model_configs: list[dict[str, Any]]
    ) -> ConfigDiffResult:
        """Compute what needs to change for an incremental reload.

        Does not mutate _current_model_configs. The caller is responsible
        for updating it after a successful reload.
        """
        return diff_model_configs(self._current_model_configs, new_model_configs)


# Singleton instance
_sync_manager: ConfigSyncManager | None = None


def get_sync_manager() -> ConfigSyncManager:
    """Get or create the global sync manager."""
    global _sync_manager
    if _sync_manager is None:
        _sync_manager = ConfigSyncManager()
    return _sync_manager


def start_config_sync():
    """Start background config synchronization."""
    manager = get_sync_manager()
    manager.start()


def stop_config_sync():
    """Stop background config synchronization."""
    if _sync_manager:
        _sync_manager.stop()


def reset_config_sync_manager() -> None:
    """Reset the global config sync manager (for testing)."""
    global _sync_manager
    if _sync_manager is not None:
        _sync_manager.stop()
    _sync_manager = None
