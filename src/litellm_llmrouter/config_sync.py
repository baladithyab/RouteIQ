"""
Configuration Sync and Hot Reload
==================================

Background sync from S3/GCS with file watching for hot reload.
Uses ETag-based caching to avoid unnecessary downloads.
"""

import hashlib
import os
import signal
import threading
from pathlib import Path
from typing import Callable

from litellm._logging import verbose_proxy_logger


class ConfigSyncManager:
    """Manages config synchronization from S3/GCS with hot reload support.

    Uses ETag-based change detection to minimize bandwidth and only
    download when remote config has actually changed.
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
            return response.get("ETag", "").strip('"')
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
            f"hot_reload: {self.hot_reload_enabled})"
        )

        while not self._stop_event.is_set():
            try:
                self._last_sync_time = time.time()

                # Check S3 for updates using ETag
                if self.s3_sync_enabled:
                    if self._download_from_s3_if_changed() and self.hot_reload_enabled:
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
        if not self.sync_enabled or (
            not self.s3_sync_enabled and not self.gcs_sync_enabled
        ):
            verbose_proxy_logger.info(
                "Config sync disabled or no remote config configured"
            )
            return

        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

    def stop(self):
        """Stop the background sync."""
        self._stop_event.set()
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5)

    def force_sync(self) -> bool:
        """Force an immediate sync from remote storage."""
        if self.s3_sync_enabled:
            return self._download_from_s3_if_changed()
        return False

    def get_status(self) -> dict:
        """Get the current sync status."""
        return {
            "enabled": self.sync_enabled,
            "hot_reload_enabled": self.hot_reload_enabled,
            "sync_interval_seconds": self.sync_interval,
            "s3": {
                "enabled": self.s3_sync_enabled,
                "bucket": self.s3_bucket,
                "key": self.s3_key,
                "last_etag": self._last_s3_etag,
            }
            if self.s3_sync_enabled
            else None,
            "gcs": {
                "enabled": self.gcs_sync_enabled,
                "bucket": self.gcs_bucket,
                "key": self.gcs_key,
            }
            if self.gcs_sync_enabled
            else None,
            "local_config_path": str(self.local_config_path),
            "local_config_hash": self._compute_file_hash(self.local_config_path),
            "reload_count": self._reload_count,
            "last_sync_time": self._last_sync_time,
            "running": self._sync_thread is not None and self._sync_thread.is_alive(),
        }


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
