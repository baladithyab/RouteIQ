"""
URL Security Utilities for SSRF Prevention
===========================================

This module provides URL validation utilities to prevent Server-Side Request
Forgery (SSRF) attacks when making outbound HTTP requests to user-configured URLs.

Security Focus (Fail-Closed / Deny-by-Default):
- Block loopback addresses (127.0.0.0/8, ::1) - ALWAYS blocked
- Block link-local addresses (169.254.0.0/16, fe80::/10) - includes cloud metadata
- Block private network IPs (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) by default
- Block IPv6 unique-local addresses (fc00::/7) by default
- Reject direct IP literal hosts by default (requires hostname)
- Allow only http:// and https:// schemes
- Explicit allowlists for hosts/domains, CIDRs, and URL prefixes

Configuration (Environment Variables):
- LLMROUTER_OUTBOUND_ALLOW_PRIVATE: Set to "true" to allow private/unique-local IPs
  (default: false / blocked). Legacy alias: LLMROUTER_ALLOW_PRIVATE_IPS
- LLMROUTER_OUTBOUND_HOST_ALLOWLIST: Comma-separated list of allowed hosts/domains
  - Exact match (e.g., "myserver.internal")
  - Wildcard match (e.g., "*.trusted.com" matches "api.trusted.com")
  - Suffix match (e.g., ".trusted.com" matches "api.trusted.com") [legacy format]
  Legacy alias: LLMROUTER_SSRF_ALLOWLIST_HOSTS
- LLMROUTER_OUTBOUND_CIDR_ALLOWLIST: Comma-separated CIDR ranges (e.g., "10.100.0.0/16")
  Legacy alias: LLMROUTER_SSRF_ALLOWLIST_CIDRS
- LLMROUTER_OUTBOUND_URL_ALLOWLIST: Comma-separated URL prefixes that bypass all checks
  (e.g., "http://internal-mcp.local:8080/,https://trusted-api.internal/")

Backwards Compatibility:
- Old env vars (LLMROUTER_ALLOW_PRIVATE_IPS, LLMROUTER_SSRF_ALLOWLIST_HOSTS,
  LLMROUTER_SSRF_ALLOWLIST_CIDRS) remain supported and are checked if new vars are unset.

Usage:
    from litellm_llmrouter.url_security import validate_outbound_url

    # Raises SSRFBlockedError if URL is dangerous
    validate_outbound_url("https://user-configured-endpoint.com/api")
"""

import ipaddress
import os
import socket
from functools import lru_cache
from urllib.parse import urlparse

from litellm._logging import verbose_proxy_logger


class SSRFBlockedError(Exception):
    """Raised when a URL is blocked due to SSRF risk."""

    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"SSRF blocked: {reason} (URL: {url})")


# Dangerous hostname patterns (exact match, case-insensitive)
BLOCKED_HOSTNAMES = frozenset(
    [
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        # AWS metadata endpoints
        "instance-data",
        "metadata",
        "metadata.google.internal",
        "metadata.gke.internal",
        # Azure metadata
        "metadata.azure.com",
        # GCP metadata
        "computeMetadata",
    ]
)

# Allowed URL schemes
ALLOWED_SCHEMES = frozenset(["http", "https"])

# IPv6 unique-local address range (fc00::/7)
# This covers both fd00::/8 (locally assigned) and fc00::/8 (globally assigned)
IPV6_UNIQUE_LOCAL_NETWORK = ipaddress.ip_network("fc00::/7")


def _get_env_with_fallback(primary: str, fallback: str, default: str = "") -> str:
    """
    Get environment variable with fallback to legacy name.

    Args:
        primary: Primary (new) env var name
        fallback: Fallback (legacy) env var name
        default: Default value if neither is set

    Returns:
        Value from primary env var, or fallback, or default
    """
    value = os.getenv(primary)
    if value is not None:
        return value
    value = os.getenv(fallback)
    if value is not None:
        verbose_proxy_logger.debug(
            f"SSRF: Using legacy env var {fallback} (prefer {primary})"
        )
        return value
    return default


@lru_cache(maxsize=1)
def _get_ssrf_config() -> dict:
    """
    Load SSRF configuration from environment variables.

    Returns a cached configuration dict with:
    - allow_private_ips: bool (default: False - blocked)
    - allowlist_hosts: set of allowed host patterns
    - allowlist_cidrs: list of ipaddress.ip_network objects
    - allowlist_urls: list of URL prefixes that bypass checks

    Env vars support legacy names for backwards compatibility.
    """
    # Default: private IPs are BLOCKED (deny-by-default)
    allow_private_str = _get_env_with_fallback(
        "LLMROUTER_OUTBOUND_ALLOW_PRIVATE",
        "LLMROUTER_ALLOW_PRIVATE_IPS",
        "false",
    )
    allow_private_ips = allow_private_str.lower() == "true"

    # Parse allowlisted hosts (exact match, wildcard with *, or suffix match with leading dot)
    allowlist_hosts_str = _get_env_with_fallback(
        "LLMROUTER_OUTBOUND_HOST_ALLOWLIST",
        "LLMROUTER_SSRF_ALLOWLIST_HOSTS",
        "",
    )
    allowlist_hosts = set()
    for host in allowlist_hosts_str.split(","):
        host = host.strip().lower()
        if host:
            allowlist_hosts.add(host)

    # Parse allowlisted CIDRs
    allowlist_cidrs_str = _get_env_with_fallback(
        "LLMROUTER_OUTBOUND_CIDR_ALLOWLIST",
        "LLMROUTER_SSRF_ALLOWLIST_CIDRS",
        "",
    )
    allowlist_cidrs = []
    for cidr in allowlist_cidrs_str.split(","):
        cidr = cidr.strip()
        if cidr:
            try:
                network = ipaddress.ip_network(cidr, strict=False)
                allowlist_cidrs.append(network)
            except ValueError as e:
                verbose_proxy_logger.warning(
                    f"SSRF: Invalid CIDR in allowlist: {cidr} - {e}"
                )

    # Parse URL prefix allowlist (new feature - no legacy alias)
    allowlist_urls_str = os.getenv("LLMROUTER_OUTBOUND_URL_ALLOWLIST", "")
    allowlist_urls = []
    for url_prefix in allowlist_urls_str.split(","):
        url_prefix = url_prefix.strip()
        if url_prefix:
            # Normalize: ensure trailing slash for prefix matching
            allowlist_urls.append(url_prefix)

    verbose_proxy_logger.debug(
        f"SSRF config loaded: allow_private_ips={allow_private_ips}, "
        f"allowlist_hosts={allowlist_hosts}, allowlist_cidrs={len(allowlist_cidrs)} networks, "
        f"allowlist_urls={len(allowlist_urls)} prefixes"
    )

    return {
        "allow_private_ips": allow_private_ips,
        "allowlist_hosts": allowlist_hosts,
        "allowlist_cidrs": allowlist_cidrs,
        "allowlist_urls": allowlist_urls,
    }


def clear_ssrf_config_cache() -> None:
    """Clear the cached SSRF configuration. Useful for testing."""
    _get_ssrf_config.cache_clear()


def _is_url_allowlisted(url: str, config: dict) -> bool:
    """
    Check if a URL matches any allowlisted URL prefix.

    Args:
        url: The full URL to check
        config: The SSRF configuration dict

    Returns:
        True if the URL starts with any allowlisted prefix, False otherwise
    """
    url_lower = url.lower()
    for prefix in config.get("allowlist_urls", []):
        if url_lower.startswith(prefix.lower()):
            return True
    return False


def _is_host_allowlisted(hostname: str, config: dict) -> bool:
    """
    Check if a hostname is in the allowlist.

    Supports:
    - Exact match: "myserver.internal"
    - Wildcard match: "*.trusted.com" matches "api.trusted.com", "x.y.trusted.com"
    - Suffix match (legacy): ".trusted.com" matches "api.trusted.com"

    Args:
        hostname: The hostname to check (case-insensitive)
        config: The SSRF configuration dict

    Returns:
        True if the hostname is allowlisted, False otherwise
    """
    hostname_lower = hostname.lower().strip(".")
    allowlist = config.get("allowlist_hosts", set())

    # Check exact match
    if hostname_lower in allowlist:
        return True

    for pattern in allowlist:
        # Wildcard match: "*.example.com" matches "api.example.com"
        if pattern.startswith("*."):
            suffix = pattern[1:]  # ".example.com"
            if hostname_lower.endswith(suffix) or hostname_lower == suffix[1:]:
                return True
        # Legacy suffix match: ".example.com" matches "api.example.com"
        elif pattern.startswith("."):
            if hostname_lower.endswith(pattern) or hostname_lower == pattern[1:]:
                return True

    return False


def _is_ip_allowlisted(ip_str: str, config: dict) -> bool:
    """
    Check if an IP address is in the allowlisted CIDRs.

    Args:
        ip_str: The IP address string
        config: The SSRF configuration dict

    Returns:
        True if the IP is in any allowlisted CIDR, False otherwise
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False

    for network in config.get("allowlist_cidrs", []):
        if ip in network:
            return True

    return False


def _is_ip_literal(hostname: str) -> bool:
    """
    Check if a hostname is actually an IP address literal.

    Args:
        hostname: The hostname to check

    Returns:
        True if it's an IP literal, False if it's a domain name
    """
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _check_ip_address(ip_str: str, config: dict = None) -> tuple[bool, str]:
    """
    Check if an IP address should be blocked.

    Order of checks (deny-by-default):
    1. Always block loopback (127.0.0.0/8, ::1) - NEVER allowlisted
    2. Always block link-local (169.254.0.0/16, fe80::/10) - includes cloud metadata
    3. Check allowlist for private IPs and IPv6 unique-local
    4. Block private IPs (RFC1918) and IPv6 unique-local (fc00::/7) unless:
       - allow_private_ips=True, OR
       - IP is in CIDR allowlist

    Args:
        ip_str: IP address string to check
        config: SSRF configuration dict (auto-loaded if None)

    Returns:
        Tuple of (should_block, reason) where reason explains why blocked
    """
    if config is None:
        config = _get_ssrf_config()

    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False, ""  # Not a valid IP, let it through for hostname resolution

    # Block loopback - ALWAYS blocked, no allowlist override
    if ip.is_loopback:
        return True, f"loopback address {ip_str} is blocked"

    # Block link-local (includes cloud metadata IP 169.254.169.254) - ALWAYS blocked
    if ip.is_link_local:
        return (
            True,
            f"link-local address {ip_str} is blocked (potential cloud metadata endpoint)",
        )

    # IPv6 specific checks
    if isinstance(ip, ipaddress.IPv6Address):
        # Block IPv4-mapped loopback/link-local - ALWAYS blocked
        if ip.ipv4_mapped:
            ipv4 = ip.ipv4_mapped
            if ipv4.is_loopback or ipv4.is_link_local:
                return True, f"IPv4-mapped loopback/link-local {ip_str} is blocked"

        # Check IPv6 unique-local (fc00::/7) - blocked by default like private IPv4
        if ip in IPV6_UNIQUE_LOCAL_NETWORK:
            # Check CIDR allowlist first
            if _is_ip_allowlisted(ip_str, config):
                verbose_proxy_logger.debug(
                    f"SSRF: IPv6 unique-local {ip_str} allowed by CIDR allowlist"
                )
                return False, ""

            # Check global allow_private_ips setting
            if config.get("allow_private_ips", False):
                return False, ""

            # Block unique-local IPv6
            return (
                True,
                f"Access to IPv6 unique-local address {ip_str} is blocked for security reasons",
            )

    # Check if IP is private (RFC1918 for IPv4)
    if ip.is_private and not ip.is_loopback:  # loopback already caught above
        # Check CIDR allowlist first
        if _is_ip_allowlisted(ip_str, config):
            verbose_proxy_logger.debug(
                f"SSRF: Private IP {ip_str} allowed by CIDR allowlist"
            )
            return False, ""

        # Check global allow_private_ips setting
        if config.get("allow_private_ips", False):
            return False, ""

        # Block private IP
        return (
            True,
            f"Access to private IP {ip_str} is blocked for security reasons",
        )

    return False, ""


def _check_hostname(hostname: str, config: dict = None) -> tuple[bool, str]:
    """
    Check if a hostname should be blocked.

    Args:
        hostname: Hostname to check (case-insensitive)
        config: SSRF configuration dict (auto-loaded if None)

    Returns:
        Tuple of (should_block, reason)
    """
    if config is None:
        config = _get_ssrf_config()

    hostname_lower = hostname.lower().strip(".")

    # Check if hostname is in allowlist first
    if _is_host_allowlisted(hostname, config):
        verbose_proxy_logger.debug(
            f"SSRF: Hostname '{hostname}' allowed by host allowlist"
        )
        return False, ""

    # Check exact matches against blocked hostnames
    if hostname_lower in BLOCKED_HOSTNAMES:
        return (
            True,
            f"hostname '{hostname}' is blocked (potential loopback/metadata endpoint)",
        )

    # Check if hostname ends with blocked suffixes
    blocked_suffixes = [".localhost", ".local"]
    for suffix in blocked_suffixes:
        if hostname_lower.endswith(suffix):
            return True, f"hostname '{hostname}' with suffix '{suffix}' is blocked"

    return False, ""


def validate_outbound_url(
    url: str,
    resolve_dns: bool = True,
    allow_private_ips: bool | None = None,
) -> str:
    """
    Validate a URL for safe outbound HTTP requests.

    This function checks URLs against SSRF attack patterns and raises
    SSRFBlockedError if the URL targets a potentially dangerous endpoint.

    **DENY-BY-DEFAULT**: Private IPs and IPv6 unique-local are blocked unless
    explicitly allowed via configuration.

    Blocked targets (always, cannot be overridden):
    - Non-http/https schemes (file://, ftp://, etc.)
    - Localhost and loopback addresses (127.0.0.1, ::1)
    - Link-local addresses including cloud metadata (169.254.0.0/16, fe80::/10)
    - Hostnames like "localhost", "metadata.google.internal"

    Blocked by default (can be allowed via config):
    - Private network IPs (10.x.x.x, 172.16-31.x.x, 192.168.x.x)
    - IPv6 unique-local addresses (fc00::/7)
    - Configure LLMROUTER_OUTBOUND_ALLOW_PRIVATE=true to allow all private IPs
    - Configure LLMROUTER_OUTBOUND_HOST_ALLOWLIST to allow specific hosts/domains
    - Configure LLMROUTER_OUTBOUND_CIDR_ALLOWLIST to allow specific IP ranges
    - Configure LLMROUTER_OUTBOUND_URL_ALLOWLIST to allow specific URL prefixes

    Args:
        url: The URL to validate
        resolve_dns: If True, resolve hostname to IP and check the IP too
        allow_private_ips: Override env var; if True, allow RFC1918/ULA IPs for this call

    Returns:
        The original URL if validation passes

    Raises:
        SSRFBlockedError: If the URL is blocked due to SSRF risk
        ValueError: If the URL is malformed
    """
    if not url:
        raise ValueError("URL cannot be empty")

    # Load configuration
    config = _get_ssrf_config()

    # Check URL allowlist first - bypasses all other checks
    if _is_url_allowlisted(url, config):
        verbose_proxy_logger.debug(f"SSRF: URL '{url}' allowed by URL prefix allowlist")
        return url

    # Apply per-call override if provided
    if allow_private_ips is not None:
        config = dict(config)  # Create a copy to avoid modifying cached config
        config["allow_private_ips"] = allow_private_ips

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {e}") from e

    # Check scheme
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise SSRFBlockedError(
            url, f"scheme '{scheme}' not allowed; only http/https permitted"
        )

    # Get hostname
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must have a hostname")

    # Check blocked hostnames (considers allowlist)
    blocked, reason = _check_hostname(hostname, config)
    if blocked:
        verbose_proxy_logger.warning(f"SSRF: Blocked URL due to hostname: {url}")
        raise SSRFBlockedError(url, reason)

    # Check if hostname is an IP address directly
    blocked, reason = _check_ip_address(hostname, config)
    if blocked:
        verbose_proxy_logger.warning(f"SSRF: Blocked URL due to IP address: {url}")
        raise SSRFBlockedError(url, reason)

    # Optionally resolve DNS and check resolved IP
    if resolve_dns:
        try:
            # Get all IPs for the hostname
            addr_info = socket.getaddrinfo(
                hostname, parsed.port or (443 if scheme == "https" else 80)
            )
            for family, type_, proto, canonname, sockaddr in addr_info:
                ip_str = sockaddr[0]
                blocked, reason = _check_ip_address(ip_str, config)
                if blocked:
                    verbose_proxy_logger.warning(
                        f"SSRF: Blocked URL due to resolved IP {ip_str}: {url}"
                    )
                    raise SSRFBlockedError(
                        url, f"resolved IP {ip_str} is blocked: {reason}"
                    )

        except socket.gaierror:
            # DNS resolution failed - let it through, will fail on actual connection
            verbose_proxy_logger.debug(
                f"SSRF: DNS resolution failed for {hostname}, allowing"
            )
            pass
        except SSRFBlockedError:
            raise  # Re-raise SSRF errors
        except Exception as e:
            # Other socket errors - log and allow
            verbose_proxy_logger.debug(f"SSRF: DNS check failed for {hostname}: {e}")
            pass

    verbose_proxy_logger.debug(f"SSRF: URL validated as safe: {url}")
    return url


def is_url_safe(url: str, resolve_dns: bool = True) -> bool:
    """
    Check if a URL is safe for outbound requests without raising exceptions.

    Args:
        url: The URL to check
        resolve_dns: If True, also resolve and check the IP

    Returns:
        True if URL is safe, False otherwise
    """
    try:
        validate_outbound_url(url, resolve_dns=resolve_dns)
        return True
    except (SSRFBlockedError, ValueError):
        return False
