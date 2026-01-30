"""
SSRF Hardening Tests
====================

Tests for the secure-by-default outbound egress policy:
- Private IPs are blocked by default (fail-closed)
- Loopback and link-local addresses are always blocked
- Allowlist support for hosts/domains (exact and suffix match)
- Allowlist support for CIDRs
- Enforcement in MCP and A2A registration flows
"""

import os
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


class TestSSRFBlockedByDefault:
    """Test that private IPs are blocked by default (secure-by-default)."""

    def setup_method(self):
        """Clear config cache before each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        clear_ssrf_config_cache()

    def teardown_method(self):
        """Clean up environment after each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        # Clear any test env vars
        for key in [
            "LLMROUTER_ALLOW_PRIVATE_IPS",
            "LLMROUTER_SSRF_ALLOWLIST_HOSTS",
            "LLMROUTER_SSRF_ALLOWLIST_CIDRS",
        ]:
            if key in os.environ:
                del os.environ[key]
        clear_ssrf_config_cache()

    def test_loopback_always_blocked(self):
        """Test that 127.0.0.1 (loopback) is always blocked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://127.0.0.1/api", resolve_dns=False)

        assert "loopback" in exc_info.value.reason.lower()

    def test_loopback_blocked_even_with_allowlist(self):
        """Test that loopback is blocked even if allow_private_ips is True."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        os.environ["LLMROUTER_ALLOW_PRIVATE_IPS"] = "true"
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        clear_ssrf_config_cache()

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://127.0.0.1/api", resolve_dns=False)

        assert "loopback" in exc_info.value.reason.lower()

    def test_localhost_blocked(self):
        """Test that localhost hostname is blocked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError):
            validate_outbound_url("http://localhost/api", resolve_dns=False)

    def test_metadata_ip_blocked(self):
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

    def test_private_ip_10_blocked_by_default(self):
        """Test that 10.x.x.x private IPs are blocked by default."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://10.0.0.1/api", resolve_dns=False)

        assert "private IP" in exc_info.value.reason

    def test_private_ip_172_blocked_by_default(self):
        """Test that 172.16.x.x private IPs are blocked by default."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://172.16.0.1/api", resolve_dns=False)

        assert "private IP" in exc_info.value.reason

    def test_private_ip_192_blocked_by_default(self):
        """Test that 192.168.x.x private IPs are blocked by default."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://192.168.1.1/api", resolve_dns=False)

        assert "private IP" in exc_info.value.reason

    def test_public_url_allowed(self):
        """Test that public URLs are allowed."""
        from litellm_llmrouter.url_security import validate_outbound_url

        # This should not raise - public URL
        result = validate_outbound_url(
            "https://api.example.com/v1/chat", resolve_dns=False
        )
        assert result == "https://api.example.com/v1/chat"

    def test_is_url_safe_returns_false_for_private_ip(self):
        """Test is_url_safe() helper returns False for private IPs."""
        from litellm_llmrouter.url_security import is_url_safe

        assert is_url_safe("http://10.0.0.1/api", resolve_dns=False) is False
        assert is_url_safe("http://192.168.1.1/api", resolve_dns=False) is False
        assert is_url_safe("http://127.0.0.1/api", resolve_dns=False) is False

    def test_is_url_safe_returns_true_for_public(self):
        """Test is_url_safe() helper returns True for public URLs."""
        from litellm_llmrouter.url_security import is_url_safe

        assert is_url_safe("https://api.openai.com/v1/chat", resolve_dns=False) is True


class TestSSRFAllowlistHosts:
    """Test host/domain allowlist functionality."""

    def setup_method(self):
        """Clear config cache before each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        clear_ssrf_config_cache()

    def teardown_method(self):
        """Clean up environment after each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        for key in [
            "LLMROUTER_ALLOW_PRIVATE_IPS",
            "LLMROUTER_SSRF_ALLOWLIST_HOSTS",
            "LLMROUTER_SSRF_ALLOWLIST_CIDRS",
        ]:
            if key in os.environ:
                del os.environ[key]
        clear_ssrf_config_cache()

    def test_exact_host_match_allows_blocked_hostname(self):
        """Test that exact hostname match in allowlist allows the host."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        os.environ["LLMROUTER_SSRF_ALLOWLIST_HOSTS"] = "myinternal.local"
        clear_ssrf_config_cache()

        # .local suffix would normally be blocked, but allowlist permits it
        result = validate_outbound_url("http://myinternal.local/api", resolve_dns=False)
        assert result == "http://myinternal.local/api"

    def test_suffix_match_allows_subdomains(self):
        """Test that suffix pattern (e.g., .trusted.com) allows subdomains."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        os.environ["LLMROUTER_SSRF_ALLOWLIST_HOSTS"] = ".trusted.internal"
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

    def test_suffix_match_allows_exact_domain(self):
        """Test that suffix pattern also allows the exact domain."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        os.environ["LLMROUTER_SSRF_ALLOWLIST_HOSTS"] = ".trusted.internal"
        clear_ssrf_config_cache()

        result = validate_outbound_url("http://trusted.internal/api", resolve_dns=False)
        assert result == "http://trusted.internal/api"

    def test_multiple_hosts_allowlisted(self):
        """Test that multiple comma-separated hosts work."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        os.environ["LLMROUTER_SSRF_ALLOWLIST_HOSTS"] = (
            "host1.local, host2.local, .trusted.com"
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

    def setup_method(self):
        """Clear config cache before each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        clear_ssrf_config_cache()

    def teardown_method(self):
        """Clean up environment after each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        for key in [
            "LLMROUTER_ALLOW_PRIVATE_IPS",
            "LLMROUTER_SSRF_ALLOWLIST_HOSTS",
            "LLMROUTER_SSRF_ALLOWLIST_CIDRS",
        ]:
            if key in os.environ:
                del os.environ[key]
        clear_ssrf_config_cache()

    def test_cidr_allows_specific_private_range(self):
        """Test that CIDR allowlist permits specific private IP ranges."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
            clear_ssrf_config_cache,
        )

        os.environ["LLMROUTER_SSRF_ALLOWLIST_CIDRS"] = "10.100.0.0/16"
        clear_ssrf_config_cache()

        # IPs in the allowed range should work
        result = validate_outbound_url("http://10.100.0.1/api", resolve_dns=False)
        assert result == "http://10.100.0.1/api"

        result = validate_outbound_url("http://10.100.255.255/api", resolve_dns=False)
        assert result == "http://10.100.255.255/api"

        # IPs outside the allowed range should still be blocked
        with pytest.raises(SSRFBlockedError):
            validate_outbound_url("http://10.101.0.1/api", resolve_dns=False)

    def test_multiple_cidrs_allowed(self):
        """Test that multiple CIDRs can be specified."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        os.environ["LLMROUTER_SSRF_ALLOWLIST_CIDRS"] = "10.100.0.0/16,192.168.1.0/24"
        clear_ssrf_config_cache()

        result = validate_outbound_url("http://10.100.5.5/api", resolve_dns=False)
        assert result == "http://10.100.5.5/api"

        result = validate_outbound_url("http://192.168.1.100/api", resolve_dns=False)
        assert result == "http://192.168.1.100/api"

    def test_loopback_not_allowed_via_cidr(self):
        """Test that loopback cannot be allowed via CIDR (always blocked)."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
            clear_ssrf_config_cache,
        )

        # Even if someone tries to allowlist the loopback range
        os.environ["LLMROUTER_SSRF_ALLOWLIST_CIDRS"] = "127.0.0.0/8"
        clear_ssrf_config_cache()

        # Loopback should still be blocked
        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("http://127.0.0.1/api", resolve_dns=False)

        assert "loopback" in exc_info.value.reason.lower()

    def test_link_local_not_allowed_via_cidr(self):
        """Test that link-local cannot be allowed via CIDR (always blocked)."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
            clear_ssrf_config_cache,
        )

        os.environ["LLMROUTER_SSRF_ALLOWLIST_CIDRS"] = "169.254.0.0/16"
        clear_ssrf_config_cache()

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url(
                "http://169.254.169.254/latest/meta-data/", resolve_dns=False
            )

        assert "link-local" in exc_info.value.reason.lower()


class TestPrivateIPAllowlist:
    """Test the global allow_private_ips setting."""

    def setup_method(self):
        """Clear config cache before each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        clear_ssrf_config_cache()

    def teardown_method(self):
        """Clean up environment after each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        for key in [
            "LLMROUTER_ALLOW_PRIVATE_IPS",
            "LLMROUTER_SSRF_ALLOWLIST_HOSTS",
            "LLMROUTER_SSRF_ALLOWLIST_CIDRS",
        ]:
            if key in os.environ:
                del os.environ[key]
        clear_ssrf_config_cache()

    def test_allow_private_ips_permits_all_private_ranges(self):
        """Test that LLMROUTER_ALLOW_PRIVATE_IPS=true permits all private IPs."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            clear_ssrf_config_cache,
        )

        os.environ["LLMROUTER_ALLOW_PRIVATE_IPS"] = "true"
        clear_ssrf_config_cache()

        result = validate_outbound_url("http://10.0.0.1/api", resolve_dns=False)
        assert result == "http://10.0.0.1/api"

        result = validate_outbound_url("http://172.16.5.5/api", resolve_dns=False)
        assert result == "http://172.16.5.5/api"

        result = validate_outbound_url("http://192.168.1.1/api", resolve_dns=False)
        assert result == "http://192.168.1.1/api"

    def test_per_call_override_allow_private_ips(self):
        """Test that allow_private_ips parameter overrides env var."""
        from litellm_llmrouter.url_security import validate_outbound_url

        # Default: blocked
        result = validate_outbound_url(
            "http://10.0.0.1/api", resolve_dns=False, allow_private_ips=True
        )
        assert result == "http://10.0.0.1/api"


class TestGatewayIntegration:
    """Test SSRF enforcement in MCP gateway registration flows."""

    def setup_method(self):
        """Clear config cache before each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        clear_ssrf_config_cache()

    def teardown_method(self):
        """Clean up environment after each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        for key in [
            "LLMROUTER_ALLOW_PRIVATE_IPS",
            "LLMROUTER_SSRF_ALLOWLIST_HOSTS",
            "LLMROUTER_SSRF_ALLOWLIST_CIDRS",
            "MCP_GATEWAY_ENABLED",
        ]:
            if key in os.environ:
                del os.environ[key]
        clear_ssrf_config_cache()

    def test_mcp_registration_blocks_private_ip_url(self):
        """Test that MCP server registration rejects private IP URLs."""
        from litellm_llmrouter.mcp_gateway import MCPGateway, MCPServer, MCPTransport

        os.environ["MCP_GATEWAY_ENABLED"] = "true"
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

    def test_mcp_registration_allows_public_url(self):
        """Test that MCP server registration allows public URLs."""
        from litellm_llmrouter.mcp_gateway import MCPGateway, MCPServer, MCPTransport

        os.environ["MCP_GATEWAY_ENABLED"] = "true"
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

    def test_mcp_registration_respects_allowlist(self):
        """Test that allowlisted private IPs work in MCP registration."""
        from litellm_llmrouter.mcp_gateway import MCPGateway, MCPServer, MCPTransport
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        os.environ["MCP_GATEWAY_ENABLED"] = "true"
        os.environ["LLMROUTER_SSRF_ALLOWLIST_CIDRS"] = "10.100.0.0/16"
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

    def setup_method(self):
        """Clear config cache before each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        clear_ssrf_config_cache()

    def teardown_method(self):
        """Clean up environment after each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        for key in [
            "LLMROUTER_ALLOW_PRIVATE_IPS",
            "LLMROUTER_SSRF_ALLOWLIST_HOSTS",
            "LLMROUTER_SSRF_ALLOWLIST_CIDRS",
            "A2A_GATEWAY_ENABLED",
        ]:
            if key in os.environ:
                del os.environ[key]
        clear_ssrf_config_cache()

    def test_a2a_registration_blocks_private_ip_url(self):
        """Test that A2A agent registration rejects private IP URLs."""
        from litellm_llmrouter.a2a_gateway import A2AGateway, A2AAgent

        os.environ["A2A_GATEWAY_ENABLED"] = "true"
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

    def test_a2a_registration_allows_public_url(self):
        """Test that A2A agent registration allows public URLs."""
        from litellm_llmrouter.a2a_gateway import A2AGateway, A2AAgent

        os.environ["A2A_GATEWAY_ENABLED"] = "true"
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

    def setup_method(self):
        """Clear config cache before each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        clear_ssrf_config_cache()

    def teardown_method(self):
        """Clean up environment after each test."""
        from litellm_llmrouter.url_security import clear_ssrf_config_cache

        for key in [
            "LLMROUTER_ALLOW_PRIVATE_IPS",
            "LLMROUTER_SSRF_ALLOWLIST_HOSTS",
            "LLMROUTER_SSRF_ALLOWLIST_CIDRS",
        ]:
            if key in os.environ:
                del os.environ[key]
        clear_ssrf_config_cache()

    def test_openai_api_allowed(self):
        """Test that OpenAI API URLs are allowed."""
        from litellm_llmrouter.url_security import validate_outbound_url

        result = validate_outbound_url(
            "https://api.openai.com/v1/chat/completions", resolve_dns=False
        )
        assert result == "https://api.openai.com/v1/chat/completions"

    def test_anthropic_api_allowed(self):
        """Test that Anthropic API URLs are allowed."""
        from litellm_llmrouter.url_security import validate_outbound_url

        result = validate_outbound_url(
            "https://api.anthropic.com/v1/messages", resolve_dns=False
        )
        assert result == "https://api.anthropic.com/v1/messages"

    def test_azure_api_allowed(self):
        """Test that Azure OpenAI URLs are allowed."""
        from litellm_llmrouter.url_security import validate_outbound_url

        result = validate_outbound_url(
            "https://myresource.openai.azure.com/openai/deployments/gpt-4/chat/completions",
            resolve_dns=False,
        )
        assert "myresource.openai.azure.com" in result

    def test_custom_mcp_server_allowed(self):
        """Test that custom MCP server URLs are allowed."""
        from litellm_llmrouter.url_security import validate_outbound_url

        result = validate_outbound_url(
            "https://mcp.mycompany.com/api/v1", resolve_dns=False
        )
        assert result == "https://mcp.mycompany.com/api/v1"

    def test_http_allowed_for_public(self):
        """Test that HTTP (not just HTTPS) is allowed for public URLs."""
        from litellm_llmrouter.url_security import validate_outbound_url

        result = validate_outbound_url("http://api.example.com/v1", resolve_dns=False)
        assert result == "http://api.example.com/v1"

    def test_non_standard_port_allowed(self):
        """Test that non-standard ports are allowed."""
        from litellm_llmrouter.url_security import validate_outbound_url

        result = validate_outbound_url(
            "https://api.example.com:8443/v1", resolve_dns=False
        )
        assert result == "https://api.example.com:8443/v1"


class TestBlockedSchemes:
    """Tests for blocked URL schemes."""

    def test_ftp_blocked(self):
        """Test that ftp:// scheme is blocked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("ftp://files.example.com/file.txt", resolve_dns=False)

        assert "scheme" in exc_info.value.reason.lower()

    def test_file_blocked(self):
        """Test that file:// scheme is blocked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("file:///etc/passwd", resolve_dns=False)

        assert "scheme" in exc_info.value.reason.lower()

    def test_gopher_blocked(self):
        """Test that gopher:// scheme is blocked."""
        from litellm_llmrouter.url_security import (
            validate_outbound_url,
            SSRFBlockedError,
        )

        with pytest.raises(SSRFBlockedError) as exc_info:
            validate_outbound_url("gopher://internal/", resolve_dns=False)

        assert "scheme" in exc_info.value.reason.lower()
