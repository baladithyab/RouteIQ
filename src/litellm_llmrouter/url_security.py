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
- LLMROUTER_SSRF_USE_SYNC_DNS: Set to "true" to use synchronous (blocking) DNS resolution.
  (default: false = non-blocking async DNS). Use as rollback if async DNS causes issues.
- LLMROUTER_SSRF_DNS_TIMEOUT: Timeout in seconds for async DNS resolution (default: 5.0)
- LLMROUTER_SSRF_DNS_CACHE_TTL: TTL in seconds for DNS cache entries (default: 60)
- LLMROUTER_SSRF_DNS_CACHE_SIZE: Maximum number of DNS cache entries (default: 1000)

Backwards Compatibility:
- Old env vars (LLMROUTER_ALLOW_PRIVATE_IPS, LLMROUTER_SSRF_ALLOWLIST_HOSTS,
  LLMROUTER_SSRF_ALLOWLIST_CIDRS) remain supported and are checked if new vars are unset.

Usage:
    from litellm_llmrouter.url_security import validate_outbound_url

    # Sync usage (raises SSRFBlockedError if URL is dangerous)
    validate_outbound_url("https://user-configured-endpoint.com/api")

    # Async usage (non-blocking DNS resolution)
    await validate_outbound_url_async("https://user-configured-endpoint.com/api")
"""

import asyncio
import ipaddress
import os
import socket
import threading
import time
from functools import lru_cache
from typing import Any, cast
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

# Default DNS resolution timeout (seconds)
DEFAULT_DNS_TIMEOUT = 5.0

# DNS cache configuration defaults
DEFAULT_DNS_CACHE_TTL = 60  # seconds
DEFAULT_DNS_CACHE_SIZE = 1000  # max entries


# =============================================================================
# DNS Cache Implementation (Thread-Safe TTL Cache)
# =============================================================================


class DNSCacheEntry:
    """A cached DNS resolution result with TTL."""

    __slots__ = ("addresses", "expires_at")

    def __init__(self, addresses: list[tuple[int, int, int, str, tuple]], ttl: float):
        self.addresses = addresses
        self.expires_at = time.monotonic() + ttl

    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at


class DNSCache:
    """
    Thread-safe TTL cache for DNS resolution results.

    Uses a simple dict with periodic cleanup to avoid memory leaks.
    Thread safety is achieved via a threading.Lock.
    """

    def __init__(
        self, max_size: int = DEFAULT_DNS_CACHE_SIZE, ttl: float = DEFAULT_DNS_CACHE_TTL
    ):
        self._cache: dict[str, DNSCacheEntry] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl = ttl
        self._cleanup_threshold = max_size // 10  # Cleanup when 10% over capacity

    def get(self, key: str) -> list[tuple[int, int, int, str, tuple]] | None:
        """Get cached DNS result if not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._cache[key]
                return None
            return entry.addresses

    def set(self, key: str, addresses: list[tuple[int, int, int, str, tuple]]) -> None:
        """Cache DNS result with TTL."""
        with self._lock:
            # Cleanup if over capacity
            if len(self._cache) >= self._max_size + self._cleanup_threshold:
                self._cleanup_expired_locked()

            self._cache[key] = DNSCacheEntry(addresses, self._ttl)

    def _cleanup_expired_locked(self) -> None:
        """Remove expired entries. Must be called with lock held."""
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for k in expired_keys:
            del self._cache[k]

        # If still over capacity, remove oldest entries (LRU-ish)
        if len(self._cache) >= self._max_size:
            # Sort by expiry time and remove oldest
            sorted_keys = sorted(
                self._cache.keys(), key=lambda k: self._cache[k].expires_at
            )
            excess = len(self._cache) - self._max_size + self._cleanup_threshold
            for k in sorted_keys[:excess]:
                del self._cache[k]

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            expired_count = sum(1 for v in self._cache.values() if v.is_expired())
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl": self._ttl,
                "expired_pending_cleanup": expired_count,
            }


# Global DNS cache instance (lazy initialized via _get_dns_cache)
_dns_cache: DNSCache | None = None
_dns_cache_lock = threading.Lock()


def _get_dns_cache() -> DNSCache:
    """Get or create the global DNS cache instance."""
    global _dns_cache
    if _dns_cache is None:
        with _dns_cache_lock:
            if _dns_cache is None:
                config = _get_ssrf_config()
                _dns_cache = DNSCache(
                    max_size=config.get("dns_cache_size", DEFAULT_DNS_CACHE_SIZE),
                    ttl=config.get("dns_cache_ttl", DEFAULT_DNS_CACHE_TTL),
                )
    return _dns_cache


def clear_dns_cache() -> None:
    """Clear the DNS resolution cache. Useful for testing."""
    global _dns_cache
    if _dns_cache is not None:
        _dns_cache.clear()


def get_dns_cache_stats() -> dict[str, Any]:
    """Get DNS cache statistics."""
    cache = _get_dns_cache()
    return cache.stats()


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
    - use_sync_dns: bool (default: False - use async DNS)
    - dns_timeout: float (default: 5.0 seconds)
    - dns_cache_ttl: float (default: 60 seconds)
    - dns_cache_size: int (default: 1000 entries)

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

    # Rollback flag: use sync DNS resolution (default: False = use async)
    use_sync_dns_str = os.getenv("LLMROUTER_SSRF_USE_SYNC_DNS", "false")
    use_sync_dns = use_sync_dns_str.lower() == "true"

    # DNS resolution timeout
    dns_timeout_str = os.getenv("LLMROUTER_SSRF_DNS_TIMEOUT", str(DEFAULT_DNS_TIMEOUT))
    try:
        dns_timeout = float(dns_timeout_str)
        if dns_timeout <= 0:
            dns_timeout = DEFAULT_DNS_TIMEOUT
    except ValueError:
        dns_timeout = DEFAULT_DNS_TIMEOUT

    # DNS cache TTL
    dns_cache_ttl_str = os.getenv(
        "LLMROUTER_SSRF_DNS_CACHE_TTL", str(DEFAULT_DNS_CACHE_TTL)
    )
    try:
        dns_cache_ttl = float(dns_cache_ttl_str)
        if dns_cache_ttl <= 0:
            dns_cache_ttl = DEFAULT_DNS_CACHE_TTL
    except ValueError:
        dns_cache_ttl = DEFAULT_DNS_CACHE_TTL

    # DNS cache size
    dns_cache_size_str = os.getenv(
        "LLMROUTER_SSRF_DNS_CACHE_SIZE", str(DEFAULT_DNS_CACHE_SIZE)
    )
    try:
        dns_cache_size = int(dns_cache_size_str)
        if dns_cache_size <= 0:
            dns_cache_size = DEFAULT_DNS_CACHE_SIZE
    except ValueError:
        dns_cache_size = DEFAULT_DNS_CACHE_SIZE

    verbose_proxy_logger.debug(
        f"SSRF config loaded: allow_private_ips={allow_private_ips}, "
        f"allowlist_hosts={allowlist_hosts}, allowlist_cidrs={len(allowlist_cidrs)} networks, "
        f"allowlist_urls={len(allowlist_urls)} prefixes, "
        f"use_sync_dns={use_sync_dns}, dns_timeout={dns_timeout}s, "
        f"dns_cache_ttl={dns_cache_ttl}s, dns_cache_size={dns_cache_size}"
    )

    return {
        "allow_private_ips": allow_private_ips,
        "allowlist_hosts": allowlist_hosts,
        "allowlist_cidrs": allowlist_cidrs,
        "allowlist_urls": allowlist_urls,
        "use_sync_dns": use_sync_dns,
        "dns_timeout": dns_timeout,
        "dns_cache_ttl": dns_cache_ttl,
        "dns_cache_size": dns_cache_size,
    }


def clear_ssrf_config_cache() -> None:
    """Clear the cached SSRF configuration. Useful for testing."""
    _get_ssrf_config.cache_clear()
    # Also reset DNS cache when config changes
    global _dns_cache
    with _dns_cache_lock:
        _dns_cache = None


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


def _check_ip_address(ip_str: str, config: dict | None = None) -> tuple[bool, str]:
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


def _check_hostname(hostname: str, config: dict | None = None) -> tuple[bool, str]:
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

    **NOTE**: This function uses synchronous DNS resolution by default when
    LLMROUTER_SSRF_USE_SYNC_DNS=true (rollback mode), which blocks the event loop.
    For async contexts, use validate_outbound_url_async() instead.

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
        _resolve_and_check_dns_sync(url, hostname, parsed, scheme, config)

    verbose_proxy_logger.debug(f"SSRF: URL validated as safe: {url}")
    return url


def _resolve_and_check_dns_sync(
    url: str, hostname: str, parsed, scheme: str, config: dict
) -> None:
    """
    Resolve DNS synchronously and check all resolved IPs.

    This is the legacy blocking DNS resolution path.

    Args:
        url: Original URL (for error messages)
        hostname: Hostname to resolve
        parsed: Parsed URL object
        scheme: URL scheme (for default port)
        config: SSRF configuration dict

    Raises:
        SSRFBlockedError: If any resolved IP is blocked
    """
    try:
        # Get all IPs for the hostname
        addr_info = socket.getaddrinfo(
            hostname, parsed.port or (443 if scheme == "https" else 80)
        )
        for family, type_, proto, canonname, sockaddr in addr_info:
            resolved_ip = str(sockaddr[0])
            blocked, reason = _check_ip_address(resolved_ip, config)
            if blocked:
                verbose_proxy_logger.warning(
                    f"SSRF: Blocked URL due to resolved IP {resolved_ip}: {url}"
                )
                raise SSRFBlockedError(
                    url, f"resolved IP {resolved_ip} is blocked: {reason}"
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


async def _resolve_dns_async(
    hostname: str, port: int, timeout: float
) -> list[tuple[int, int, int, str, tuple]]:
    """
    Resolve DNS asynchronously without blocking the event loop.

    Uses asyncio.get_running_loop().getaddrinfo() for non-blocking resolution.
    Results are cached with TTL to avoid repeated resolution under load.

    Args:
        hostname: Hostname to resolve
        port: Port number for resolution context
        timeout: Timeout in seconds

    Returns:
        List of address info tuples from getaddrinfo

    Raises:
        asyncio.TimeoutError: If resolution exceeds timeout
        socket.gaierror: If DNS resolution fails
    """
    # Check cache first
    cache_key = f"{hostname}:{port}"
    cache = _get_dns_cache()
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        verbose_proxy_logger.debug(f"SSRF: DNS cache hit for {hostname}:{port}")
        return cached_result

    # Perform async DNS resolution
    loop = asyncio.get_running_loop()
    raw_result = await asyncio.wait_for(
        loop.getaddrinfo(hostname, port, family=0, type=socket.SOCK_STREAM),
        timeout=timeout,
    )
    result = cast(list[tuple[int, int, int, str, tuple[Any, ...]]], raw_result)

    # Cache the result
    cache.set(cache_key, result)
    verbose_proxy_logger.debug(f"SSRF: DNS resolved and cached for {hostname}:{port}")

    return result


async def validate_outbound_url_async(
    url: str,
    resolve_dns: bool = True,
    allow_private_ips: bool | None = None,
) -> str:
    """
    Validate a URL for safe outbound HTTP requests (async version).

    This function is the async-safe version of validate_outbound_url() that
    does NOT block the event loop during DNS resolution. It uses
    asyncio.get_running_loop().getaddrinfo() for non-blocking DNS lookups.

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

    Configuration:
    - LLMROUTER_SSRF_DNS_TIMEOUT: Timeout in seconds for DNS resolution (default: 5.0)
    - LLMROUTER_SSRF_USE_SYNC_DNS: If "true", falls back to sync resolution (default: false)

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

    # If rollback flag is set, use synchronous resolution
    if config.get("use_sync_dns", False):
        verbose_proxy_logger.debug(
            "SSRF: Using sync DNS resolution (rollback mode enabled)"
        )
        return validate_outbound_url(url, resolve_dns, allow_private_ips)

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

    # Optionally resolve DNS and check resolved IP (async)
    if resolve_dns:
        await _resolve_and_check_dns_async(url, hostname, parsed, scheme, config)

    verbose_proxy_logger.debug(f"SSRF: URL validated as safe: {url}")
    return url


async def _resolve_and_check_dns_async(
    url: str, hostname: str, parsed, scheme: str, config: dict
) -> None:
    """
    Resolve DNS asynchronously and check all resolved IPs.

    This is the non-blocking DNS resolution path that doesn't block the event loop.

    Args:
        url: Original URL (for error messages)
        hostname: Hostname to resolve
        parsed: Parsed URL object
        scheme: URL scheme (for default port)
        config: SSRF configuration dict

    Raises:
        SSRFBlockedError: If any resolved IP is blocked
    """
    dns_timeout = config.get("dns_timeout", DEFAULT_DNS_TIMEOUT)
    port = parsed.port or (443 if scheme == "https" else 80)

    try:
        # Get all IPs for the hostname asynchronously
        addr_info = await _resolve_dns_async(hostname, port, dns_timeout)

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

    except asyncio.TimeoutError:
        # DNS resolution timed out - log and allow (will fail on actual connection)
        verbose_proxy_logger.warning(
            f"SSRF: Async DNS resolution timed out for {hostname} after {dns_timeout}s, allowing"
        )
        pass
    except socket.gaierror:
        # DNS resolution failed - let it through, will fail on actual connection
        verbose_proxy_logger.debug(
            f"SSRF: DNS resolution failed for {hostname}, allowing"
        )
        pass
    except SSRFBlockedError:
        raise  # Re-raise SSRF errors
    except Exception as e:
        # Other errors - log and allow
        verbose_proxy_logger.debug(f"SSRF: Async DNS check failed for {hostname}: {e}")
        pass


async def is_url_safe_async(url: str, resolve_dns: bool = True) -> bool:
    """
    Check if a URL is safe for outbound requests without raising exceptions (async version).

    Args:
        url: The URL to check
        resolve_dns: If True, also resolve and check the IP

    Returns:
        True if URL is safe, False otherwise
    """
    try:
        await validate_outbound_url_async(url, resolve_dns=resolve_dns)
        return True
    except (SSRFBlockedError, ValueError):
        return False


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
