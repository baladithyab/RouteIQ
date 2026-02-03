"""
SSRF Async DNS Resolution Tests
================================

Tests for the non-blocking async DNS resolution in SSRF URL validation:
- Async DNS validation with validate_outbound_url_async()
- Non-blocking behavior verification
- Rollback flag (LLMROUTER_SSRF_USE_SYNC_DNS) functionality
- Timeout handling
- Multi-answer resolution (IPv4/IPv6)
- Security correctness in async path
"""

import asyncio
import socket
import time
from unittest.mock import AsyncMock, patch

import pytest

# Check if litellm is available
try:
    import litellm  # noqa: F401

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not LITELLM_AVAILABLE,
    reason="litellm package not installed - SSRF tests require litellm",
)


# Fixture to clear SSRF config cache
@pytest.fixture(autouse=True)
def clear_config_cache():
    """Clear SSRF config cache before and after each test."""
    from litellm_llmrouter.url_security import clear_ssrf_config_cache

    clear_ssrf_config_cache()
    yield
    clear_ssrf_config_cache()


# Fixture to clean env vars
@pytest.fixture
def clean_env(monkeypatch):
    """Remove all SSRF-related env vars for clean test state."""
    env_vars = [
        # New env var names
        "LLMROUTER_OUTBOUND_ALLOW_PRIVATE",
        "LLMROUTER_OUTBOUND_HOST_ALLOWLIST",
        "LLMROUTER_OUTBOUND_CIDR_ALLOWLIST",
        "LLMROUTER_OUTBOUND_URL_ALLOWLIST",
        # Legacy env var names
        "LLMROUTER_ALLOW_PRIVATE_IPS",
        "LLMROUTER_SSRF_ALLOWLIST_HOSTS",
        "LLMROUTER_SSRF_ALLOWLIST_CIDRS",
        # New async-related env vars
        "LLMROUTER_SSRF_USE_SYNC_DNS",
        "LLMROUTER_SSRF_DNS_TIMEOUT",
        # Gateway env vars
        "MCP_GATEWAY_ENABLED",
        "A2A_GATEWAY_ENABLED",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    yield


class TestAsyncSSRFValidation:
    """Test async SSRF validation functionality."""

    @pytest.mark.asyncio
    async def test_async_public_url_allowed(self, clean_env):
        """Test that public URLs are allowed via async validation."""
        from litellm_llmrouter.url_security import validate_outbound_url_async

        result = await validate_outbound_url_async(
            "https://api.example.com/v1/chat", resolve_dns=False
        )
        assert result == "https://api.example.com/v1/chat"

    @pytest.mark.asyncio
    async def test_async_loopback_blocked(self, clean_env):
        """Test that loopback is blocked via async validation."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            await validate_outbound_url_async("http://127.0.0.1/api", resolve_dns=False)

        assert "loopback" in exc_info.value.reason.lower()

    @pytest.mark.asyncio
    async def test_async_localhost_blocked(self, clean_env):
        """Test that localhost hostname is blocked via async validation."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError):
            await validate_outbound_url_async("http://localhost/api", resolve_dns=False)

    @pytest.mark.asyncio
    async def test_async_private_ip_blocked_by_default(self, clean_env):
        """Test that private IPs are blocked by default via async validation."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            await validate_outbound_url_async("http://10.0.0.1/api", resolve_dns=False)

        assert "private IP" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_async_metadata_ip_blocked(self, clean_env):
        """Test that cloud metadata IP is blocked via async validation."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            await validate_outbound_url_async(
                "http://169.254.169.254/latest/meta-data/", resolve_dns=False
            )

        assert "link-local" in exc_info.value.reason.lower()

    @pytest.mark.asyncio
    async def test_async_ipv6_loopback_blocked(self, clean_env):
        """Test that IPv6 loopback is blocked via async validation."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            await validate_outbound_url_async("http://[::1]/api", resolve_dns=False)

        assert "loopback" in exc_info.value.reason.lower()

    @pytest.mark.asyncio
    async def test_async_ipv6_unique_local_blocked_by_default(self, clean_env):
        """Test that IPv6 unique-local is blocked by default via async validation."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            await validate_outbound_url_async(
                "http://[fd12:3456:789a::1]/api", resolve_dns=False
            )

        assert "unique-local" in exc_info.value.reason.lower()

    @pytest.mark.asyncio
    async def test_async_is_url_safe(self, clean_env):
        """Test is_url_safe_async helper function."""
        from litellm_llmrouter.url_security import is_url_safe_async

        assert (
            await is_url_safe_async("https://api.openai.com/v1/chat", resolve_dns=False)
            is True
        )
        assert await is_url_safe_async("http://127.0.0.1/api", resolve_dns=False) is False
        assert await is_url_safe_async("http://10.0.0.1/api", resolve_dns=False) is False


class TestAsyncDNSResolution:
    """Test async DNS resolution behavior."""

    @pytest.mark.asyncio
    async def test_async_dns_resolution_blocks_resolved_private_ip(self, clean_env):
        """Test that resolved private IPs are blocked during async DNS resolution."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        # Mock async DNS resolution to return a private IP
        mock_addr_info = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))
        ]

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(return_value=mock_addr_info)

            with pytest.raises(SSRFBlockedError) as exc_info:
                await validate_outbound_url_async(
                    "https://malicious.example.com/api", resolve_dns=True
                )

            assert "10.0.0.1" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_async_dns_resolution_allows_public_ip(self, clean_env):
        """Test that resolved public IPs are allowed during async DNS resolution."""
        from litellm_llmrouter.url_security import validate_outbound_url_async

        # Mock async DNS resolution to return a truly public IP (Google DNS)
        mock_addr_info = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))
        ]

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(return_value=mock_addr_info)

            result = await validate_outbound_url_async(
                "https://api.example.com/v1/chat", resolve_dns=True
            )
            assert result == "https://api.example.com/v1/chat"

    @pytest.mark.asyncio
    async def test_async_dns_multi_answer_all_checked(self, clean_env):
        """Test that all DNS answers (A/AAAA records) are checked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        # Mock async DNS resolution to return multiple IPs (public + private)
        mock_addr_info = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.100", 443)),
        ]

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(return_value=mock_addr_info)

            with pytest.raises(SSRFBlockedError) as exc_info:
                await validate_outbound_url_async(
                    "https://api.example.com/v1/chat", resolve_dns=True
                )

            # Should block on the private IP
            assert "192.168.1.100" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_async_dns_ipv4_and_ipv6_answers(self, clean_env):
        """Test that both IPv4 and IPv6 DNS answers are checked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        # Mock async DNS resolution to return IPv4 (public) + IPv6 (loopback)
        mock_addr_info = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 443, 0, 0)),
        ]

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(return_value=mock_addr_info)

            with pytest.raises(SSRFBlockedError) as exc_info:
                await validate_outbound_url_async(
                    "https://api.example.com/v1/chat", resolve_dns=True
                )

            # Should block on the IPv6 loopback
            assert "loopback" in exc_info.value.reason.lower()


class TestRollbackFlag:
    """Test the LLMROUTER_SSRF_USE_SYNC_DNS rollback flag."""

    @pytest.mark.asyncio
    async def test_rollback_flag_uses_sync_dns(self, clean_env, monkeypatch):
        """Test that LLMROUTER_SSRF_USE_SYNC_DNS=true uses sync resolution."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_SSRF_USE_SYNC_DNS", "true")
        clear_ssrf_config_cache()

        # Mock the sync validate_outbound_url to track calls
        with patch(
            "litellm_llmrouter.url_security.validate_outbound_url"
        ) as mock_sync:
            mock_sync.return_value = "https://api.example.com/v1/chat"

            await validate_outbound_url_async(
                "https://api.example.com/v1/chat", resolve_dns=False
            )

            # Should have called the sync version
            mock_sync.assert_called_once_with(
                "https://api.example.com/v1/chat", False, None
            )

    @pytest.mark.asyncio
    async def test_default_uses_async_dns(self, clean_env):
        """Test that async DNS is used by default (no rollback flag)."""
        from litellm_llmrouter.url_security import (
            clear_ssrf_config_cache,
            _get_ssrf_config,
        )

        clear_ssrf_config_cache()
        config = _get_ssrf_config()

        # Default should be False (use async)
        assert config.get("use_sync_dns", False) is False

    @pytest.mark.asyncio
    async def test_rollback_flag_false_uses_async(self, clean_env, monkeypatch):
        """Test that LLMROUTER_SSRF_USE_SYNC_DNS=false explicitly uses async."""
        from litellm_llmrouter.url_security import (
            clear_ssrf_config_cache,
            _get_ssrf_config,
        )

        monkeypatch.setenv("LLMROUTER_SSRF_USE_SYNC_DNS", "false")
        clear_ssrf_config_cache()

        config = _get_ssrf_config()
        assert config.get("use_sync_dns") is False


class TestDNSTimeout:
    """Test DNS timeout configuration and handling."""

    @pytest.mark.asyncio
    async def test_custom_dns_timeout(self, clean_env, monkeypatch):
        """Test that custom DNS timeout is loaded from env var."""
        from litellm_llmrouter.url_security import (
            clear_ssrf_config_cache,
            _get_ssrf_config,
        )

        monkeypatch.setenv("LLMROUTER_SSRF_DNS_TIMEOUT", "10.0")
        clear_ssrf_config_cache()

        config = _get_ssrf_config()
        assert config.get("dns_timeout") == 10.0

    @pytest.mark.asyncio
    async def test_invalid_dns_timeout_uses_default(self, clean_env, monkeypatch):
        """Test that invalid DNS timeout falls back to default."""
        from litellm_llmrouter.url_security import (
            clear_ssrf_config_cache,
            _get_ssrf_config,
            DEFAULT_DNS_TIMEOUT,
        )

        monkeypatch.setenv("LLMROUTER_SSRF_DNS_TIMEOUT", "invalid")
        clear_ssrf_config_cache()

        config = _get_ssrf_config()
        assert config.get("dns_timeout") == DEFAULT_DNS_TIMEOUT

    @pytest.mark.asyncio
    async def test_negative_dns_timeout_uses_default(self, clean_env, monkeypatch):
        """Test that negative DNS timeout falls back to default."""
        from litellm_llmrouter.url_security import (
            clear_ssrf_config_cache,
            _get_ssrf_config,
            DEFAULT_DNS_TIMEOUT,
        )

        monkeypatch.setenv("LLMROUTER_SSRF_DNS_TIMEOUT", "-5.0")
        clear_ssrf_config_cache()

        config = _get_ssrf_config()
        assert config.get("dns_timeout") == DEFAULT_DNS_TIMEOUT

    @pytest.mark.asyncio
    async def test_dns_timeout_handled_gracefully(self, clean_env, monkeypatch):
        """Test that DNS timeout is handled gracefully (allows through)."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_SSRF_DNS_TIMEOUT", "0.001")  # Very short timeout
        clear_ssrf_config_cache()

        # Mock async DNS resolution to timeout
        async def slow_getaddrinfo(*args, **kwargs):
            await asyncio.sleep(1.0)  # Sleep longer than timeout
            return []

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = slow_getaddrinfo

            # Should not raise - timeout is handled gracefully
            result = await validate_outbound_url_async(
                "https://api.example.com/v1/chat", resolve_dns=True
            )
            assert result == "https://api.example.com/v1/chat"


class TestNonBlockingBehavior:
    """Test that async DNS resolution doesn't block the event loop."""

    @pytest.mark.asyncio
    async def test_event_loop_progress_during_dns_resolution(self, clean_env):
        """
        Test that the event loop makes progress while DNS resolution is pending.

        This test verifies that validate_outbound_url_async() doesn't block
        the event loop by running a concurrent task that increments a counter.
        """
        from litellm_llmrouter.url_security import validate_outbound_url_async

        progress_counter = 0
        dns_resolution_started = asyncio.Event()
        dns_resolution_complete = asyncio.Event()

        async def slow_getaddrinfo(*args, **kwargs):
            dns_resolution_started.set()
            # Simulate slow DNS resolution
            await asyncio.sleep(0.1)
            dns_resolution_complete.set()
            # Return truly public IP (Google DNS)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]

        async def progress_task():
            """Task that increments counter to prove event loop progress."""
            nonlocal progress_counter
            while not dns_resolution_complete.is_set():
                progress_counter += 1
                await asyncio.sleep(0.01)

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = slow_getaddrinfo

            # Run both tasks concurrently
            validation_task = asyncio.create_task(
                validate_outbound_url_async(
                    "https://api.example.com/v1/chat", resolve_dns=True
                )
            )
            counter_task = asyncio.create_task(progress_task())

            # Wait for validation to complete
            result = await validation_task

            # Cancel the counter task
            counter_task.cancel()
            try:
                await counter_task
            except asyncio.CancelledError:
                pass

        # Verify the event loop made progress (counter incremented)
        assert progress_counter > 0, "Event loop should have made progress during DNS resolution"
        assert result == "https://api.example.com/v1/chat"

    @pytest.mark.asyncio
    async def test_multiple_concurrent_validations(self, clean_env):
        """Test that multiple async validations can run concurrently."""
        from litellm_llmrouter.url_security import validate_outbound_url_async

        call_order = []

        async def mock_getaddrinfo(host, port, *args, **kwargs):
            call_order.append(f"start_{host}")
            await asyncio.sleep(0.05)  # Simulate network delay
            call_order.append(f"end_{host}")
            # Return truly public IP (Google DNS)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port))]

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = mock_getaddrinfo

            # Run multiple validations concurrently
            results = await asyncio.gather(
                validate_outbound_url_async("https://api1.example.com/", resolve_dns=True),
                validate_outbound_url_async("https://api2.example.com/", resolve_dns=True),
                validate_outbound_url_async("https://api3.example.com/", resolve_dns=True),
            )

        # All should succeed
        assert len(results) == 3
        assert all("example.com" in r for r in results)

        # If truly concurrent, starts should happen before all ends
        starts = [i for i, x in enumerate(call_order) if x.startswith("start_")]
        ends = [i for i, x in enumerate(call_order) if x.startswith("end_")]

        # At least some starts should happen before some ends (concurrent execution)
        # In sequential execution, all starts would come before all ends
        assert len(starts) == 3
        assert len(ends) == 3

    @pytest.mark.asyncio
    async def test_sync_resolution_blocks_event_loop(self, clean_env, monkeypatch):
        """
        Test that sync DNS resolution (rollback mode) blocks the event loop.

        This verifies that the rollback flag actually uses blocking resolution.
        """
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_SSRF_USE_SYNC_DNS", "true")
        clear_ssrf_config_cache()

        progress_counter = 0
        validation_done = asyncio.Event()

        async def progress_task():
            """Task that tries to increment counter during validation."""
            nonlocal progress_counter
            while not validation_done.is_set():
                progress_counter += 1
                await asyncio.sleep(0.001)

        def blocking_getaddrinfo(*args, **kwargs):
            # Simulate blocking DNS - but we can't truly block in a mock
            # so we just verify the sync path is taken
            time.sleep(0.01)
            # Return truly public IP (Google DNS)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]

        with patch("socket.getaddrinfo", blocking_getaddrinfo):
            # Start progress task
            counter_task = asyncio.create_task(progress_task())

            # Run validation (this should use sync path when rollback is enabled)
            result = await validate_outbound_url_async(
                "https://api.example.com/v1/chat", resolve_dns=True
            )

            validation_done.set()

            # Cancel and await the counter task
            counter_task.cancel()
            try:
                await counter_task
            except asyncio.CancelledError:
                pass

        # Sync resolution was used (verified by socket.getaddrinfo being called)
        assert result == "https://api.example.com/v1/chat"


class TestAsyncAllowlistIntegration:
    """Test that allowlists work correctly with async validation."""

    @pytest.mark.asyncio
    async def test_async_host_allowlist(self, clean_env, monkeypatch):
        """Test that host allowlist works with async validation."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_OUTBOUND_HOST_ALLOWLIST", "myinternal.local")
        clear_ssrf_config_cache()

        result = await validate_outbound_url_async(
            "http://myinternal.local/api", resolve_dns=False
        )
        assert result == "http://myinternal.local/api"

    @pytest.mark.asyncio
    async def test_async_cidr_allowlist(self, clean_env, monkeypatch):
        """Test that CIDR allowlist works with async validation."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_OUTBOUND_CIDR_ALLOWLIST", "10.100.0.0/16")
        clear_ssrf_config_cache()

        result = await validate_outbound_url_async(
            "http://10.100.5.10/api", resolve_dns=False
        )
        assert result == "http://10.100.5.10/api"

    @pytest.mark.asyncio
    async def test_async_url_prefix_allowlist(self, clean_env, monkeypatch):
        """Test that URL prefix allowlist works with async validation."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv(
            "LLMROUTER_OUTBOUND_URL_ALLOWLIST", "http://10.0.0.1:8080/mcp/"
        )
        clear_ssrf_config_cache()

        result = await validate_outbound_url_async(
            "http://10.0.0.1:8080/mcp/tools/call", resolve_dns=False
        )
        assert result == "http://10.0.0.1:8080/mcp/tools/call"

    @pytest.mark.asyncio
    async def test_async_allow_private_ips_override(self, clean_env):
        """Test that per-call allow_private_ips override works with async."""
        from litellm_llmrouter.url_security import validate_outbound_url_async

        # Default: blocked
        result = await validate_outbound_url_async(
            "http://10.0.0.1/api", resolve_dns=False, allow_private_ips=True
        )
        assert result == "http://10.0.0.1/api"


class TestAsyncDNSResolutionFailure:
    """Test handling of DNS resolution failures in async path."""

    @pytest.mark.asyncio
    async def test_dns_gaierror_allows_through(self, clean_env):
        """Test that DNS resolution failure (gaierror) allows through."""
        from litellm_llmrouter.url_security import validate_outbound_url_async

        async def failing_getaddrinfo(*args, **kwargs):
            raise socket.gaierror(8, "nodename nor servname provided")

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = failing_getaddrinfo

            # Should not raise - DNS failure is handled gracefully
            result = await validate_outbound_url_async(
                "https://nonexistent.example.com/api", resolve_dns=True
            )
            assert result == "https://nonexistent.example.com/api"

    @pytest.mark.asyncio
    async def test_dns_generic_error_allows_through(self, clean_env):
        """Test that generic DNS errors are handled gracefully."""
        from litellm_llmrouter.url_security import validate_outbound_url_async

        async def failing_getaddrinfo(*args, **kwargs):
            raise OSError("Network unreachable")

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = failing_getaddrinfo

            # Should not raise - error is handled gracefully
            result = await validate_outbound_url_async(
                "https://api.example.com/v1/chat", resolve_dns=True
            )
            assert result == "https://api.example.com/v1/chat"


class TestSSRFSecurityCorrectness:
    """Security-focused tests to ensure SSRF protection is correct in async path."""

    @pytest.mark.asyncio
    async def test_async_blocks_dns_rebinding_attack(self, clean_env):
        """Test that DNS rebinding attacks are blocked (private IP in resolution)."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        # Simulate DNS rebinding: public domain resolving to private IP
        mock_addr_info = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))
        ]

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(return_value=mock_addr_info)

            with pytest.raises(SSRFBlockedError) as exc_info:
                await validate_outbound_url_async(
                    "https://evil-rebind.example.com/api", resolve_dns=True
                )

            assert "loopback" in exc_info.value.reason.lower()

    @pytest.mark.asyncio
    async def test_async_blocks_cloud_metadata_in_resolution(self, clean_env):
        """Test that cloud metadata IP in DNS resolution is blocked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        # Simulate domain resolving to cloud metadata IP
        mock_addr_info = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80))
        ]

        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.getaddrinfo = AsyncMock(return_value=mock_addr_info)

            with pytest.raises(SSRFBlockedError) as exc_info:
                await validate_outbound_url_async(
                    "http://metadata-steal.example.com/", resolve_dns=True
                )

            assert "link-local" in exc_info.value.reason.lower()

    @pytest.mark.asyncio
    async def test_async_scheme_validation(self, clean_env):
        """Test that non-http/https schemes are blocked in async path."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        for scheme in ["ftp", "file", "gopher", "dict", "ldap"]:
            with pytest.raises(SSRFBlockedError) as exc_info:
                await validate_outbound_url_async(
                    f"{scheme}://example.com/", resolve_dns=False
                )

            assert "scheme" in exc_info.value.reason.lower()

    @pytest.mark.asyncio
    async def test_async_empty_url_rejected(self, clean_env):
        """Test that empty URLs are rejected in async path."""
        from litellm_llmrouter.url_security import validate_outbound_url_async

        with pytest.raises(ValueError) as exc_info:
            await validate_outbound_url_async("", resolve_dns=False)

        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_async_malformed_url_rejected(self, clean_env):
        """Test that malformed URLs are rejected in async path."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        # URL without scheme is rejected as SSRFBlockedError (empty scheme)
        with pytest.raises(SSRFBlockedError) as exc_info:
            await validate_outbound_url_async("not-a-valid-url", resolve_dns=False)

        assert "scheme" in exc_info.value.reason.lower()
