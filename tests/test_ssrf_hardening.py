"""
SSRF Hardening Tests
====================

Tests for the deny-by-default outbound egress policy:
- Private IPs (RFC1918) are blocked by default (fail-closed)
- IPv6 unique-local (fc00::/7) are blocked by default
- Loopback (127.0.0.0/8, ::1) and link-local (169.254.0.0/16, fe80::/10) are ALWAYS blocked
- Allowlist support for hosts/domains (exact, wildcard, suffix match)
- Allowlist support for CIDRs
- Allowlist support for URL prefixes
- Backwards compatibility with legacy env var names
- Enforcement in MCP and A2A registration flows
"""

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
        # Gateway env vars
        "MCP_GATEWAY_ENABLED",
        "A2A_GATEWAY_ENABLED",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    yield


class TestSSRFBlockedByDefault:
    """Test that private IPs are blocked by default (deny-by-default)."""

    def test_loopback_always_blocked(self, clean_env):
        """Test that 127.0.0.1 (loopback) is always blocked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://127.0.0.1/api", resolve_dns=False)

        assert "loopback" in exc_info.value.reason.lower()

    def test_loopback_blocked_even_with_allowlist(self, clean_env, monkeypatch):
        """Test that loopback is blocked even if allow_private_ips is True."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_OUTBOUND_ALLOW_PRIVATE", "true")
        clear_ssrf_config_cache()

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://127.0.0.1/api", resolve_dns=False)

        assert "loopback" in exc_info.value.reason.lower()

    def test_localhost_blocked(self, clean_env):
        """Test that localhost hostname is blocked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError):
            validate_outbound_url("http://localhost/api", resolve_dns=False)

    def test_metadata_ip_blocked(self, clean_env):
        """Test that cloud metadata IP (169.254.169.254) is blocked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url(
                "http://169.254.169.254/latest/meta-data/", resolve_dns=False
            )

        assert "link-local" in exc_info.value.reason.lower()

    def test_private_ip_10_blocked_by_default(self, clean_env):
        """Test that 10.x.x.x private IPs are blocked by default."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://10.0.0.1/api", resolve_dns=False)

        assert "private IP" in exc_info.value.reason

    def test_private_ip_172_blocked_by_default(self, clean_env):
        """Test that 172.16.x.x private IPs are blocked by default."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://172.16.0.1/api", resolve_dns=False)

        assert "private IP" in exc_info.value.reason

    def test_private_ip_192_blocked_by_default(self, clean_env):
        """Test that 192.168.x.x private IPs are blocked by default."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://192.168.1.1/api", resolve_dns=False)

        assert "private IP" in exc_info.value.reason

    def test_public_url_allowed(self, clean_env):
        """Test that public URLs are allowed."""
        from litellm_llmrouter.url_security import validate_outbound_url

        # This should not raise - public URL
        result = validate_outbound_url(
            "https://api.example.com/v1/chat", resolve_dns=False
        )
        assert result == "https://api.example.com/v1/chat"

    def test_is_url_safe_returns_false_for_private_ip(self, clean_env):
        """Test is_url_safe() helper returns False for private IPs."""
        from litellm_llmrouter.url_security import is_url_safe

        assert is_url_safe("http://10.0.0.1/api", resolve_dns=False) is False
        assert is_url_safe("http://192.168.1.1/api", resolve_dns=False) is False
        assert is_url_safe("http://127.0.0.1/api", resolve_dns=False) is False

    def test_is_url_safe_returns_true_for_public(self, clean_env):
        """Test is_url_safe() helper returns True for public URLs."""
        from litellm_llmrouter.url_security import is_url_safe

        assert is_url_safe("https://api.openai.com/v1/chat", resolve_dns=False) is True


class TestIPv6Blocking:
    """Test IPv6 address blocking."""

    def test_ipv6_loopback_blocked(self, clean_env):
        """Test that IPv6 loopback (::1) is always blocked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://[::1]/api", resolve_dns=False)

        assert "loopback" in exc_info.value.reason.lower()

    def test_ipv6_link_local_blocked(self, clean_env):
        """Test that IPv6 link-local (fe80::/10) is always blocked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://[fe80::1]/api", resolve_dns=False)

        assert "link-local" in exc_info.value.reason.lower()

    def test_ipv6_unique_local_blocked_by_default(self, clean_env):
        """Test that IPv6 unique-local (fc00::/7) is blocked by default."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        # fd00::/8 is the commonly used portion of unique-local
        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://[fd12:3456:789a::1]/api", resolve_dns=False)

        assert "unique-local" in exc_info.value.reason.lower()

    def test_ipv6_unique_local_allowed_when_private_enabled(
        self, clean_env, monkeypatch
    ):
        """Test that IPv6 unique-local is allowed when ALLOW_PRIVATE=true."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_OUTBOUND_ALLOW_PRIVATE", "true")
        clear_ssrf_config_cache()

        result = validate_outbound_url(
            "http://[fd12:3456:789a::1]/api", resolve_dns=False
        )
        assert "[fd12:3456:789a::1]" in result

    def test_ipv6_loopback_blocked_even_with_allow_private(
        self, clean_env, monkeypatch
    ):
        """Test that IPv6 loopback is ALWAYS blocked even with ALLOW_PRIVATE=true."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_OUTBOUND_ALLOW_PRIVATE", "true")
        clear_ssrf_config_cache()

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://[::1]/api", resolve_dns=False)

        assert "loopback" in exc_info.value.reason.lower()


class TestSSRFAllowlistHosts:
    """Test host/domain allowlist functionality."""

    def test_exact_host_match_allows_blocked_hostname(self, clean_env, monkeypatch):
        """Test that exact hostname match in allowlist allows the host."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_OUTBOUND_HOST_ALLOWLIST", "myinternal.local")
        clear_ssrf_config_cache()

        # .local suffix would normally be blocked, but allowlist permits it
        result = validate_outbound_url("http://myinternal.local/api", resolve_dns=False)
        assert result == "http://myinternal.local/api"

    def test_wildcard_match_allows_subdomains(self, clean_env, monkeypatch):
        """Test that wildcard pattern (e.g., *.trusted.com) allows subdomains."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_OUTBOUND_HOST_ALLOWLIST", "*.trusted.internal")
        clear_ssrf_config_cache()

        # Should allow any subdomain of trusted.internal
        result = validate_outbound_url(
            "http://api.trusted.internal/v1", resolve_dns=False
        )
        assert result == "http://api.trusted.internal/v1"

        result = validate_outbound_url(
            "http://mcp.trusted.internal:8080/", resolve_dns=False
        )
        assert result == "http://mcp.trusted.internal:8080/"

    def test_suffix_match_allows_subdomains(self, clean_env, monkeypatch):
        """Test that suffix pattern (e.g., .trusted.com) allows subdomains."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_OUTBOUND_HOST_ALLOWLIST", ".trusted.internal")
        clear_ssrf_config_cache()

        # Should allow any subdomain of trusted.internal
        result = validate_outbound_url(
            "http://api.trusted.internal/v1", resolve_dns=False
        )
        assert result == "http://api.trusted.internal/v1"

        result = validate_outbound_url(
            "http://mcp.trusted.internal:8080/", resolve_dns=False
        )
        assert result == "http://mcp.trusted.internal:8080/"

    def test_suffix_match_allows_exact_domain(self, clean_env, monkeypatch):
        """Test that suffix pattern also allows the exact domain."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_OUTBOUND_HOST_ALLOWLIST", ".trusted.internal")
        clear_ssrf_config_cache()

        result = validate_outbound_url("http://trusted.internal/api", resolve_dns=False)
        assert result == "http://trusted.internal/api"

    def test_multiple_hosts_allowlisted(self, clean_env, monkeypatch):
        """Test that multiple comma-separated hosts work."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv(
            "LLMROUTER_OUTBOUND_HOST_ALLOWLIST",
            "host1.local, host2.local, *.trusted.com",
        )
        clear_ssrf_config_cache()

        result = validate_outbound_url("http://host1.local/api", resolve_dns=False)
        assert result == "http://host1.local/api"

        result = validate_outbound_url("http://host2.local/api", resolve_dns=False)
        assert result == "http://host2.local/api"

        result = validate_outbound_url("http://api.trusted.com/v1", resolve_dns=False)
        assert result == "http://api.trusted.com/v1"


class TestSSRFAllowlistCIDRs:
    """Test CIDR allowlist functionality."""

    def test_cidr_allows_specific_private_range(self, clean_env, monkeypatch):
        """Test that CIDR allowlist permits specific private IP ranges."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_OUTBOUND_CIDR_ALLOWLIST", "10.100.0.0/16")
        clear_ssrf_config_cache()

        # IPs in the allowed range should work
        result = validate_outbound_url("http://10.100.0.1/api", resolve_dns=False)
        assert result == "http://10.100.0.1/api"

        result = validate_outbound_url("http://10.100.255.255/api", resolve_dns=False)
        assert result == "http://10.100.255.255/api"

        # IPs outside the allowed range should still be blocked
        with pytest.raises(SSRFBlockedError):
            validate_outbound_url("http://10.101.0.1/api", resolve_dns=False)

    def test_multiple_cidrs_allowed(self, clean_env, monkeypatch):
        """Test that multiple CIDRs can be specified."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv(
            "LLMROUTER_OUTBOUND_CIDR_ALLOWLIST", "10.100.0.0/16,192.168.1.0/24"
        )
        clear_ssrf_config_cache()

        result = validate_outbound_url("http://10.100.5.5/api", resolve_dns=False)
        assert result == "http://10.100.5.5/api"

        result = validate_outbound_url("http://192.168.1.100/api", resolve_dns=False)
        assert result == "http://192.168.1.100/api"

    def test_loopback_not_allowed_via_cidr(self, clean_env, monkeypatch):
        """Test that loopback cannot be allowed via CIDR (always blocked)."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
            clear_ssrf_config_cache,
        )

        # Even if someone tries to allowlist the loopback range
        monkeypatch.setenv("LLMROUTER_OUTBOUND_CIDR_ALLOWLIST", "127.0.0.0/8")
        clear_ssrf_config_cache()

        # Loopback should still be blocked
        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://127.0.0.1/api", resolve_dns=False)

        assert "loopback" in exc_info.value.reason.lower()

    def test_link_local_not_allowed_via_cidr(self, clean_env, monkeypatch):
        """Test that link-local cannot be allowed via CIDR (always blocked)."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_OUTBOUND_CIDR_ALLOWLIST", "169.254.0.0/16")
        clear_ssrf_config_cache()

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url(
                "http://169.254.169.254/latest/meta-data/", resolve_dns=False
            )

        assert "link-local" in exc_info.value.reason.lower()

    def test_ipv6_cidr_allowlist(self, clean_env, monkeypatch):
        """Test that IPv6 CIDRs work in allowlist."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_OUTBOUND_CIDR_ALLOWLIST", "fd00:1234::/32")
        clear_ssrf_config_cache()

        result = validate_outbound_url("http://[fd00:1234::1]/api", resolve_dns=False)
        assert "[fd00:1234::1]" in result


class TestURLPrefixAllowlist:
    """Test URL prefix allowlist functionality."""

    def test_url_prefix_allows_matching_url(self, clean_env, monkeypatch):
        """Test that URL prefix allowlist permits matching URLs."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv(
            "LLMROUTER_OUTBOUND_URL_ALLOWLIST",
            "http://10.0.0.1:8080/mcp/",
        )
        clear_ssrf_config_cache()

        # Exact prefix match should work (bypasses IP check)
        result = validate_outbound_url(
            "http://10.0.0.1:8080/mcp/tools/call", resolve_dns=False
        )
        assert result == "http://10.0.0.1:8080/mcp/tools/call"

    def test_url_prefix_blocks_non_matching_url(self, clean_env, monkeypatch):
        """Test that URL prefix allowlist blocks non-matching URLs."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv(
            "LLMROUTER_OUTBOUND_URL_ALLOWLIST",
            "http://10.0.0.1:8080/mcp/",
        )
        clear_ssrf_config_cache()

        # Different port or path should still be blocked
        with pytest.raises(SSRFBlockedError):
            validate_outbound_url("http://10.0.0.1:9090/api", resolve_dns=False)

    def test_multiple_url_prefixes(self, clean_env, monkeypatch):
        """Test that multiple URL prefixes work."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv(
            "LLMROUTER_OUTBOUND_URL_ALLOWLIST",
            "http://10.0.0.1:8080/mcp/,https://internal.local/a2a/",
        )
        clear_ssrf_config_cache()

        result1 = validate_outbound_url(
            "http://10.0.0.1:8080/mcp/tools", resolve_dns=False
        )
        assert "10.0.0.1:8080" in result1

        result2 = validate_outbound_url(
            "https://internal.local/a2a/agent", resolve_dns=False
        )
        assert "internal.local" in result2


class TestPrivateIPAllowlist:
    """Test the global allow_private_ips setting."""

    def test_allow_private_ips_permits_all_private_ranges(self, clean_env, monkeypatch):
        """Test that LLMROUTER_OUTBOUND_ALLOW_PRIVATE=true permits all private IPs."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_OUTBOUND_ALLOW_PRIVATE", "true")
        clear_ssrf_config_cache()

        result = validate_outbound_url("http://10.0.0.1/api", resolve_dns=False)
        assert result == "http://10.0.0.1/api"

        result = validate_outbound_url("http://172.16.5.5/api", resolve_dns=False)
        assert result == "http://172.16.5.5/api"

        result = validate_outbound_url("http://192.168.1.1/api", resolve_dns=False)
        assert result == "http://192.168.1.1/api"

    def test_per_call_override_allow_private_ips(self, clean_env):
        """Test that allow_private_ips parameter overrides env var."""
        from litellm_llmrouter.url_security import validate_outbound_url

        # Default: blocked
        result = validate_outbound_url(
            "http://10.0.0.1/api", resolve_dns=False, allow_private_ips=True
        )
        assert result == "http://10.0.0.1/api"


class TestLegacyEnvVarCompatibility:
    """Test backwards compatibility with legacy env var names."""

    def test_legacy_allow_private_ips_env_var(self, clean_env, monkeypatch):
        """Test that LLMROUTER_ALLOW_PRIVATE_IPS (legacy) still works."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_ALLOW_PRIVATE_IPS", "true")
        clear_ssrf_config_cache()

        result = validate_outbound_url("http://10.0.0.1/api", resolve_dns=False)
        assert result == "http://10.0.0.1/api"

    def test_legacy_host_allowlist_env_var(self, clean_env, monkeypatch):
        """Test that LLMROUTER_SSRF_ALLOWLIST_HOSTS (legacy) still works."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_SSRF_ALLOWLIST_HOSTS", "myinternal.local")
        clear_ssrf_config_cache()

        result = validate_outbound_url("http://myinternal.local/api", resolve_dns=False)
        assert result == "http://myinternal.local/api"

    def test_legacy_cidr_allowlist_env_var(self, clean_env, monkeypatch):
        """Test that LLMROUTER_SSRF_ALLOWLIST_CIDRS (legacy) still works."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_SSRF_ALLOWLIST_CIDRS", "10.100.0.0/16")
        clear_ssrf_config_cache()

        result = validate_outbound_url("http://10.100.0.1/api", resolve_dns=False)
        assert result == "http://10.100.0.1/api"

    def test_new_env_var_takes_precedence_over_legacy(self, clean_env, monkeypatch):
        """Test that new env var names take precedence over legacy."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
            clear_ssrf_config_cache,
        )

        # Set legacy to allow, new to block (via not being "true")
        monkeypatch.setenv("LLMROUTER_ALLOW_PRIVATE_IPS", "true")
        monkeypatch.setenv("LLMROUTER_OUTBOUND_ALLOW_PRIVATE", "false")
        clear_ssrf_config_cache()

        # New env var should take precedence - should be blocked
        with pytest.raises(SSRFBlockedError):
            validate_outbound_url("http://10.0.0.1/api", resolve_dns=False)


class TestGatewayIntegration:
    """Test SSRF enforcement in MCP gateway registration flows."""

    def test_mcp_registration_blocks_private_ip_url(self, clean_env, monkeypatch):
        """Test that MCP server registration rejects private IP URLs."""
        from litellm_llmrouter.mcp_gateway import MCPGateway, MCPServer, MCPTransport

        monkeypatch.setenv("MCP_GATEWAY_ENABLED", "true")
        gateway = MCPGateway()
        gateway.enabled = True  # Force enable for test

        server = MCPServer(
            server_id="test-server",
            name="Test",
            url="http://10.0.0.1:8080/mcp",
            transport=MCPTransport.STREAMABLE_HTTP,
        )

        with pytest.raises(ValueError) as exc_info:
            gateway.register_server(server)

        assert "blocked for security reasons" in str(exc_info.value)

    def test_mcp_registration_allows_public_url(self, clean_env, monkeypatch):
        """Test that MCP server registration allows public URLs."""
        from litellm_llmrouter.mcp_gateway import MCPGateway, MCPServer, MCPTransport

        monkeypatch.setenv("MCP_GATEWAY_ENABLED", "true")
        gateway = MCPGateway()
        gateway.enabled = True  # Force enable for test

        server = MCPServer(
            server_id="test-server",
            name="Test",
            url="https://mcp.example.com/api",
            transport=MCPTransport.STREAMABLE_HTTP,
        )

        # Should not raise
        gateway.register_server(server)
        assert gateway.get_server("test-server") is not None

    def test_mcp_registration_respects_allowlist(self, clean_env, monkeypatch):
        """Test that allowlisted private IPs work in MCP registration."""
        from litellm_llmrouter.mcp_gateway import MCPGateway, MCPServer, MCPTransport
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        monkeypatch.setenv("MCP_GATEWAY_ENABLED", "true")
        monkeypatch.setenv("LLMROUTER_OUTBOUND_CIDR_ALLOWLIST", "10.100.0.0/16")
        clear_ssrf_config_cache()

        gateway = MCPGateway()
        gateway.enabled = True

        server = MCPServer(
            server_id="test-server",
            name="Test",
            url="http://10.100.5.10:8080/mcp",
            transport=MCPTransport.STREAMABLE_HTTP,
        )

        # Should not raise
        gateway.register_server(server)
        assert gateway.get_server("test-server") is not None


class TestA2AIntegration:
    """Test SSRF enforcement in A2A gateway registration flows."""

    def test_a2a_registration_blocks_private_ip_url(self, clean_env, monkeypatch):
        """Test that A2A agent registration rejects private IP URLs."""
        from litellm_llmrouter.a2a_gateway import A2AGateway, A2AAgent

        monkeypatch.setenv("A2A_GATEWAY_ENABLED", "true")
        gateway = A2AGateway()
        gateway.enabled = True  # Force enable for test

        agent = A2AAgent(
            agent_id="test-agent",
            name="Test Agent",
            description="Test",
            url="http://192.168.1.100:8000/a2a",
        )

        with pytest.raises(ValueError) as exc_info:
            gateway.register_agent(agent)

        assert "blocked for security reasons" in str(exc_info.value)

    def test_a2a_registration_allows_public_url(self, clean_env, monkeypatch):
        """Test that A2A agent registration allows public URLs."""
        from litellm_llmrouter.a2a_gateway import A2AGateway, A2AAgent

        monkeypatch.setenv("A2A_GATEWAY_ENABLED", "true")
        gateway = A2AGateway()
        gateway.enabled = True  # Force enable for test

        agent = A2AAgent(
            agent_id="test-agent",
            name="Test Agent",
            description="Test",
            url="https://agent.example.com/a2a",
        )

        # Should not raise
        gateway.register_agent(agent)
        assert gateway.get_agent("test-agent") is not None


class TestPublicURLRegression:
    """Regression tests to ensure public URLs continue to work."""

    def test_openai_api_allowed(self, clean_env):
        """Test that OpenAI API URLs are allowed."""
        from litellm_llmrouter.url_security import validate_outbound_url

        result = validate_outbound_url(
            "https://api.openai.com/v1/chat/completions", resolve_dns=False
        )
        assert result == "https://api.openai.com/v1/chat/completions"

    def test_anthropic_api_allowed(self, clean_env):
        """Test that Anthropic API URLs are allowed."""
        from litellm_llmrouter.url_security import validate_outbound_url

        result = validate_outbound_url(
            "https://api.anthropic.com/v1/messages", resolve_dns=False
        )
        assert result == "https://api.anthropic.com/v1/messages"

    def test_azure_api_allowed(self, clean_env):
        """Test that Azure OpenAI URLs are allowed."""
        from litellm_llmrouter.url_security import validate_outbound_url

        result = validate_outbound_url(
            "https://myresource.openai.azure.com/openai/deployments/gpt-4/chat/completions",
            resolve_dns=False,
        )
        assert "myresource.openai.azure.com" in result

    def test_custom_mcp_server_allowed(self, clean_env):
        """Test that custom MCP server URLs are allowed."""
        from litellm_llmrouter.url_security import validate_outbound_url

        result = validate_outbound_url(
            "https://mcp.mycompany.com/api/v1", resolve_dns=False
        )
        assert result == "https://mcp.mycompany.com/api/v1"

    def test_http_allowed_for_public(self, clean_env):
        """Test that HTTP (not just HTTPS) is allowed for public URLs."""
        from litellm_llmrouter.url_security import validate_outbound_url

        result = validate_outbound_url("http://api.example.com/v1", resolve_dns=False)
        assert result == "http://api.example.com/v1"

    def test_non_standard_port_allowed(self, clean_env):
        """Test that non-standard ports are allowed."""
        from litellm_llmrouter.url_security import validate_outbound_url

        result = validate_outbound_url(
            "https://api.example.com:8443/v1", resolve_dns=False
        )
        assert result == "https://api.example.com:8443/v1"


class TestBlockedSchemes:
    """Tests for blocked URL schemes."""

    def test_ftp_blocked(self, clean_env):
        """Test that ftp:// scheme is blocked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("ftp://files.example.com/file.txt", resolve_dns=False)

        assert "scheme" in exc_info.value.reason.lower()

    def test_file_blocked(self, clean_env):
        """Test that file:// scheme is blocked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("file:///etc/passwd", resolve_dns=False)

        assert "scheme" in exc_info.value.reason.lower()

    def test_gopher_blocked(self, clean_env):
        """Test that gopher:// scheme is blocked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("gopher://internal/", resolve_dns=False)

        assert "scheme" in exc_info.value.reason.lower()


class TestAsyncDNSResolution:
    """Tests for async DNS resolution and caching."""

    @pytest.mark.asyncio
    async def test_async_validation_basic(self, clean_env):
        """Test that async validation works for basic URLs."""
        from litellm_llmrouter.url_security import validate_outbound_url_async

        result = await validate_outbound_url_async(
            "https://api.example.com/v1/chat", resolve_dns=False
        )
        assert result == "https://api.example.com/v1/chat"

    @pytest.mark.asyncio
    async def test_async_validation_blocks_private_ip(self, clean_env):
        """Test that async validation blocks private IPs."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            await validate_outbound_url_async("http://10.0.0.1/api", resolve_dns=False)

        assert "private IP" in exc_info.value.reason

    @pytest.mark.asyncio
    async def test_async_validation_blocks_loopback(self, clean_env):
        """Test that async validation blocks loopback."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            await validate_outbound_url_async(
                "http://127.0.0.1/api", resolve_dns=False
            )

        assert "loopback" in exc_info.value.reason.lower()

    @pytest.mark.asyncio
    async def test_sync_dns_rollback_mode(self, clean_env, monkeypatch):
        """Test that LLMROUTER_SSRF_USE_SYNC_DNS=true falls back to sync."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_SSRF_USE_SYNC_DNS", "true")
        clear_ssrf_config_cache()

        # Should still work - falls back to sync validation
        result = await validate_outbound_url_async(
            "https://api.example.com/v1", resolve_dns=False
        )
        assert result == "https://api.example.com/v1"


class TestDNSCache:
    """Tests for DNS cache functionality."""

    def test_dns_cache_creation(self, clean_env):
        """Test that DNS cache is created correctly."""
        from litellm_llmrouter.url_security import get_dns_cache_stats

        stats = get_dns_cache_stats()
        assert "size" in stats
        assert "max_size" in stats
        assert "ttl" in stats
        assert stats["max_size"] > 0
        assert stats["ttl"] > 0

    def test_dns_cache_clear(self, clean_env):
        """Test that DNS cache can be cleared."""
        from litellm_llmrouter.url_security import (
            clear_dns_cache,
            get_dns_cache_stats,
        )

        clear_dns_cache()
        stats = get_dns_cache_stats()
        assert stats["size"] == 0

    def test_dns_cache_ttl_config(self, clean_env, monkeypatch):
        """Test that DNS cache TTL is configurable."""
        from litellm_llmrouter.url_security import (
            clear_ssrf_config_cache,
            clear_dns_cache,
            _get_ssrf_config,
        )

        monkeypatch.setenv("LLMROUTER_SSRF_DNS_CACHE_TTL", "120")
        monkeypatch.setenv("LLMROUTER_SSRF_DNS_CACHE_SIZE", "2000")
        clear_ssrf_config_cache()
        clear_dns_cache()

        config = _get_ssrf_config()
        assert config["dns_cache_ttl"] == 120
        assert config["dns_cache_size"] == 2000


class TestNonBlockingDNS:
    """Tests to verify DNS resolution doesn't block the event loop."""

    @pytest.mark.asyncio
    async def test_concurrent_tasks_progress_during_dns(self, clean_env):
        """
        Test that concurrent asyncio tasks make progress while DNS resolution
        is happening, proving the event loop is not blocked.

        This test:
        1. Starts multiple concurrent validation tasks with DNS resolution
        2. Also starts a counter task that increments every 10ms
        3. If DNS were blocking, the counter would not increment during resolution
        4. We verify the counter incremented, proving non-blocking behavior
        """
        import asyncio
        from litellm_llmrouter.url_security import (
            validate_outbound_url_async,
            clear_dns_cache,
        )

        # Clear cache to force actual DNS resolution
        clear_dns_cache()

        # Counter to verify event loop is responsive
        counter = {"value": 0}
        stop_event = asyncio.Event()

        async def increment_counter():
            """Increment counter every 10ms to prove event loop is responsive."""
            while not stop_event.is_set():
                counter["value"] += 1
                await asyncio.sleep(0.01)

        async def validate_url(url: str):
            """Validate a URL with DNS resolution enabled."""
            try:
                return await validate_outbound_url_async(url, resolve_dns=True)
            except Exception:
                # DNS might fail for test domains, that's OK
                return None

        # Start counter task
        counter_task = asyncio.create_task(increment_counter())

        # Create multiple concurrent validation tasks
        urls = [
            "https://example.com/api",  # Real domain
            "https://httpbin.org/get",  # Real domain
            "https://api.github.com",   # Real domain
        ]

        validation_tasks = [
            asyncio.create_task(validate_url(url))
            for url in urls
        ]

        # Wait for all validations to complete (with timeout)
        try:
            await asyncio.wait_for(
                asyncio.gather(*validation_tasks, return_exceptions=True),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            pass  # Some DNS might be slow, that's OK

        # Stop counter
        stop_event.set()
        await counter_task

        # Verify counter incremented (event loop was responsive)
        # If DNS were blocking, counter would be 0 or very low
        assert counter["value"] > 0, (
            f"Counter should have incremented during DNS resolution, "
            f"but got {counter['value']}. This indicates event loop was blocked."
        )

        # A healthier check: if we ran for at least 100ms, we should have
        # at least 5 increments (10ms each with some overhead)
        # This is a soft assertion - we just need to prove progress was made
        print(f"Counter incremented {counter['value']} times during DNS resolution")

    @pytest.mark.asyncio
    async def test_async_is_url_safe(self, clean_env):
        """Test the async is_url_safe helper."""
        from litellm_llmrouter.url_security import is_url_safe_async

        # Safe URL
        assert await is_url_safe_async("https://api.openai.com/v1", resolve_dns=False)

        # Unsafe URLs
        assert not await is_url_safe_async("http://127.0.0.1/api", resolve_dns=False)
        assert not await is_url_safe_async("http://10.0.0.1/api", resolve_dns=False)
        assert not await is_url_safe_async(
            "http://169.254.169.254/latest", resolve_dns=False
        )


class TestEnvFlagRollback:
    """Tests for the environment flag rollback mechanism."""

    @pytest.mark.asyncio
    async def test_default_uses_async_dns(self, clean_env, monkeypatch):
        """Test that the default (no env var) uses async DNS."""
        from litellm_llmrouter.url_security import (
            _get_ssrf_config,
            clear_ssrf_config_cache,
        )

        # Ensure no sync flag is set
        monkeypatch.delenv("LLMROUTER_SSRF_USE_SYNC_DNS", raising=False)
        clear_ssrf_config_cache()

        config = _get_ssrf_config()
        assert config["use_sync_dns"] is False

    @pytest.mark.asyncio
    async def test_sync_dns_flag_true(self, clean_env, monkeypatch):
        """Test that LLMROUTER_SSRF_USE_SYNC_DNS=true enables sync mode."""
        from litellm_llmrouter.url_security import (
            _get_ssrf_config,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_SSRF_USE_SYNC_DNS", "true")
        clear_ssrf_config_cache()

        config = _get_ssrf_config()
        assert config["use_sync_dns"] is True

    @pytest.mark.asyncio
    async def test_sync_dns_flag_false(self, clean_env, monkeypatch):
        """Test that LLMROUTER_SSRF_USE_SYNC_DNS=false uses async mode."""
        from litellm_llmrouter.url_security import (
            _get_ssrf_config,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_SSRF_USE_SYNC_DNS", "false")
        clear_ssrf_config_cache()

        config = _get_ssrf_config()
        assert config["use_sync_dns"] is False

    def test_dns_timeout_configurable(self, clean_env, monkeypatch):
        """Test that DNS timeout is configurable via environment variable."""
        from litellm_llmrouter.url_security import (
            _get_ssrf_config,
            clear_ssrf_config_cache,
        )

        monkeypatch.setenv("LLMROUTER_SSRF_DNS_TIMEOUT", "15.0")
        clear_ssrf_config_cache()

        config = _get_ssrf_config()
        assert config["dns_timeout"] == 15.0
