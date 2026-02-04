"""
OTel Compose Test Helpers
=========================

Utilities for managing Docker Compose stacks in OTel E2E integration tests.

Features:
- Compose stack lifecycle management
- Jaeger trace query API client
- Deterministic polling with timeouts
- Guaranteed cleanup on success AND failure
"""

import contextlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Generator

import httpx


# =============================================================================
# Configuration
# =============================================================================

# Detect compose CLI (finch or docker)
COMPOSE_CMD: str | None = None
for cmd in ("finch", "docker"):
    if shutil.which(cmd):
        COMPOSE_CMD = cmd
        break


@dataclass
class OTelTestConfig:
    """Configuration for OTel E2E tests."""

    compose_file: str = "docker-compose.otel.yml"
    gateway_url: str = "http://localhost:4001"
    jaeger_url: str = "http://localhost:16686"
    master_key: str = "sk-dev-key"
    service_name: str = "litellm-gateway"
    startup_timeout: int = 120  # Max seconds to wait for stack startup
    trace_timeout: int = 30  # Max seconds to wait for traces to appear
    poll_interval: float = 2.0  # Seconds between polls


@dataclass
class JaegerTrace:
    """Parsed Jaeger trace data."""

    trace_id: str
    spans: list[dict[str, Any]] = field(default_factory=list)
    
    def get_span_attributes(self, span_name: str | None = None) -> dict[str, Any]:
        """
        Get attributes from a span by name, or from all spans if name is None.
        
        Returns a flattened dict of attribute key -> value.
        """
        attrs = {}
        for span in self.spans:
            if span_name and span.get("operationName") != span_name:
                continue
            for tag in span.get("tags", []):
                attrs[tag["key"]] = tag["value"]
        return attrs
    
    def has_attribute(self, key: str, value: Any = None) -> bool:
        """Check if any span has the given attribute (optionally matching value)."""
        attrs = self.get_span_attributes()
        if key not in attrs:
            return False
        if value is not None:
            return attrs[key] == value
        return True


# =============================================================================
# Jaeger Query Client
# =============================================================================


class JaegerQueryClient:
    """
    Client for querying Jaeger's HTTP API.
    
    Jaeger API: https://www.jaegertracing.io/docs/1.54/apis/
    """
    
    def __init__(self, base_url: str = "http://localhost:16686"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=10.0)
    
    def close(self):
        """Close the HTTP client."""
        self._client.close()
    
    def get_services(self) -> list[str]:
        """Get list of services that have reported traces."""
        resp = self._client.get(f"{self.base_url}/api/services")
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    
    def get_traces(
        self,
        service: str,
        operation: str | None = None,
        tags: dict[str, str] | None = None,
        limit: int = 20,
        lookback: str = "1h",
    ) -> list[JaegerTrace]:
        """
        Query traces from Jaeger.
        
        Args:
            service: Service name to query
            operation: Optional operation name filter
            tags: Optional tag filters (e.g., {"router.strategy": "knn"})
            limit: Maximum traces to return
            lookback: Time lookback (e.g., "1h", "30m")
        
        Returns:
            List of JaegerTrace objects
        """
        params = {
            "service": service,
            "limit": limit,
            "lookBack": lookback,
        }
        
        if operation:
            params["operation"] = operation
        
        if tags:
            # Jaeger expects tags as "key=value" format
            tag_strs = [f"{k}={v}" for k, v in tags.items()]
            params["tags"] = "{" + ",".join(tag_strs) + "}"
        
        resp = self._client.get(f"{self.base_url}/api/traces", params=params)
        resp.raise_for_status()
        data = resp.json()
        
        traces = []
        for trace_data in data.get("data", []):
            trace_id = trace_data.get("traceID", "")
            spans = trace_data.get("spans", [])
            traces.append(JaegerTrace(trace_id=trace_id, spans=spans))
        
        return traces
    
    def wait_for_service(
        self,
        service: str,
        timeout: float = 30.0,
        poll_interval: float = 2.0,
    ) -> bool:
        """
        Wait for a service to appear in Jaeger.
        
        Returns True if service appeared within timeout, False otherwise.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                services = self.get_services()
                if service in services:
                    return True
            except Exception:
                pass
            time.sleep(poll_interval)
        return False
    
    def wait_for_trace_with_attribute(
        self,
        service: str,
        attribute_key: str,
        attribute_value: Any = None,
        timeout: float = 30.0,
        poll_interval: float = 2.0,
    ) -> JaegerTrace | None:
        """
        Wait for a trace with a specific attribute to appear.
        
        Args:
            service: Service name to query
            attribute_key: The span attribute key to look for
            attribute_value: Optional value the attribute must match
            timeout: Max seconds to wait
            poll_interval: Seconds between polls
        
        Returns:
            The matching JaegerTrace, or None if timeout
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                traces = self.get_traces(service, limit=50)
                for trace in traces:
                    if trace.has_attribute(attribute_key, attribute_value):
                        return trace
            except Exception:
                pass
            time.sleep(poll_interval)
        return None


# =============================================================================
# Compose Stack Manager
# =============================================================================


class ComposeStackManager:
    """
    Manages Docker Compose stack lifecycle for tests.
    
    Ensures cleanup on both success and failure via context manager.
    """
    
    def __init__(self, config: OTelTestConfig):
        self.config = config
        self._compose_base: list[str] = []
        self._running = False
    
    def _ensure_compose_cmd(self) -> None:
        """Ensure compose CLI is available."""
        if COMPOSE_CMD is None:
            raise RuntimeError(
                "Docker/Finch CLI not found. "
                "Install Docker or Finch: https://github.com/runfinch/finch"
            )
    
    def start(self) -> None:
        """Start the compose stack."""
        self._ensure_compose_cmd()
        
        if not os.path.exists(self.config.compose_file):
            raise FileNotFoundError(
                f"Compose file not found: {self.config.compose_file}"
            )
        
        self._compose_base = [COMPOSE_CMD, "compose", "-f", self.config.compose_file]
        
        print(f"\n🚀 Starting OTel test stack with {COMPOSE_CMD}...")
        try:
            result = subprocess.run(
                self._compose_base + ["up", "-d", "--build"],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute build timeout
            )
            print(f"Compose up stdout: {result.stdout[:500] if result.stdout else '(none)'}")
            self._running = True
        except subprocess.CalledProcessError as e:
            print(f"Compose up failed: {e.stderr}")
            raise RuntimeError(f"Failed to start compose stack: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Compose up timed out after 5 minutes")
    
    def wait_for_health(self) -> None:
        """Wait for the gateway to become healthy."""
        print(f"⏳ Waiting for gateway health at {self.config.gateway_url}...")
        deadline = time.monotonic() + self.config.startup_timeout
        
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(
                    f"{self.config.gateway_url}/health",
                    headers={"Authorization": f"Bearer {self.config.master_key}"},
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    print("✅ Gateway healthy")
                    return
            except (httpx.RequestError, httpx.TimeoutException) as e:
                print(f"Waiting: {e}")
            time.sleep(3)
        
        # Timeout - get logs for debugging
        self._dump_logs()
        raise RuntimeError(
            f"Gateway did not become healthy within {self.config.startup_timeout}s"
        )
    
    def wait_for_jaeger(self) -> None:
        """Wait for Jaeger to be accessible."""
        print(f"⏳ Waiting for Jaeger at {self.config.jaeger_url}...")
        deadline = time.monotonic() + 60  # 60s for Jaeger
        
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(f"{self.config.jaeger_url}/api/services", timeout=5.0)
                if resp.status_code == 200:
                    print("✅ Jaeger accessible")
                    return
            except (httpx.RequestError, httpx.TimeoutException):
                pass
            time.sleep(2)
        
        raise RuntimeError("Jaeger did not become accessible within 60s")
    
    def stop(self) -> None:
        """Stop and teardown the compose stack."""
        if not self._running or not self._compose_base:
            return
        
        print("\n🧹 Tearing down OTel test stack...")
        try:
            subprocess.run(
                self._compose_base + ["down", "-v", "--remove-orphans"],
                capture_output=True,
                timeout=60,
            )
        except Exception as e:
            print(f"Warning: Teardown error: {e}")
        finally:
            self._running = False
    
    def _dump_logs(self) -> None:
        """Dump container logs for debugging."""
        if not self._compose_base:
            return
        
        try:
            result = subprocess.run(
                self._compose_base + ["logs", "--tail=100"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            print(f"\n📋 Container logs:\n{result.stdout}\n{result.stderr}")
        except Exception as e:
            print(f"Warning: Could not get logs: {e}")


@contextlib.contextmanager
def otel_compose_stack(
    config: OTelTestConfig | None = None,
) -> Generator[tuple[ComposeStackManager, JaegerQueryClient], None, None]:
    """
    Context manager for OTel compose stack with guaranteed cleanup.
    
    Usage:
        with otel_compose_stack() as (stack, jaeger):
            # Make requests to stack.config.gateway_url
            # Query traces via jaeger.get_traces(...)
    
    The stack is always torn down, even if an exception occurs.
    """
    if config is None:
        config = OTelTestConfig()
    
    stack = ComposeStackManager(config)
    jaeger = JaegerQueryClient(config.jaeger_url)
    
    try:
        stack.start()
        stack.wait_for_health()
        stack.wait_for_jaeger()
        yield stack, jaeger
    finally:
        jaeger.close()
        stack.stop()


# =============================================================================
# TG4.1 Router Decision Attributes
# =============================================================================

# Expected span attributes from TG4.1 implementation
ROUTER_DECISION_ATTRIBUTES = [
    "router.strategy",
    "router.model_selected",
    "router.score",
    "router.candidates_evaluated",
    "router.decision_outcome",
    "router.decision_reason",
    "router.latency_ms",
    "router.error_type",
    "router.error_message",
    "router.strategy_version",
    "router.fallback_triggered",
]

# Minimum required attributes for a valid router decision span
REQUIRED_ROUTER_ATTRIBUTES = [
    "router.strategy",
]


def validate_router_decision_trace(trace: JaegerTrace) -> tuple[bool, list[str]]:
    """
    Validate that a trace contains router decision attributes from TG4.1.
    
    Returns:
        Tuple of (is_valid, list_of_found_attributes)
    """
    attrs = trace.get_span_attributes()
    found = [attr for attr in ROUTER_DECISION_ATTRIBUTES if attr in attrs]
    
    # Check minimum required attributes
    has_required = all(attr in attrs for attr in REQUIRED_ROUTER_ATTRIBUTES)
    
    return has_required, found
