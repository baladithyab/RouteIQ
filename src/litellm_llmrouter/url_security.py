"""
URL Security Utilities for SSRF Prevention
===========================================

This module provides URL validation utilities to prevent Server-Side Request
Forgery (SSRF) attacks when making outbound HTTP requests to user-configured URLs.

Security Focus:
- Block localhost and loopback addresses (127.0.0.0/8, ::1)
- Block link-local addresses (169.254.0.0/16, fe80::/10)
- Block AWS/cloud metadata endpoints (169.254.169.254)
- Allow only http:// and https:// schemes
- Configurable via environment variable to allow private IPs for testing

Usage:
    from litellm_llmrouter.url_security import validate_outbound_url

    # Raises SSRFBlockedError if URL is dangerous
    validate_outbound_url("https://user-configured-endpoint.com/api")
"""

import ipaddress
import os
import socket
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
    ]
)

# Allowed URL schemes
ALLOWED_SCHEMES = frozenset(["http", "https"])


def _is_ip_blocked(ip_str: str) -> tuple[bool, str]:
    """
    Check if an IP address should be blocked.

    Blocks:
    - Loopback (127.0.0.0/8, ::1)
    - Link-local (169.254.0.0/16, fe80::/10)
    - AWS/cloud metadata IP (169.254.169.254)

    Does NOT block by default (to maintain backwards compatibility):
    - Private networks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)

    Args:
        ip_str: IP address string to check

    Returns:
        Tuple of (should_block, reason) where reason explains why blocked
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False, ""  # Not a valid IP, let it through for hostname resolution

    # Block loopback
    if ip.is_loopback:
        return True, f"loopback address {ip_str} is blocked"

    # Block link-local (includes cloud metadata IP 169.254.169.254)
    if ip.is_link_local:
        return (
            True,
            f"link-local address {ip_str} is blocked (potential cloud metadata endpoint)",
        )

    # IPv6 specific: block loopback variations
    if isinstance(ip, ipaddress.IPv6Address):
        # ::1 is covered by is_loopback, but check mapped IPv4 loopback
        if ip.ipv4_mapped:
            ipv4 = ip.ipv4_mapped
            if ipv4.is_loopback or ipv4.is_link_local:
                return True, f"IPv4-mapped loopback/link-local {ip_str} is blocked"

    return False, ""


def _is_hostname_blocked(hostname: str) -> tuple[bool, str]:
    """
    Check if a hostname should be blocked.

    Args:
        hostname: Hostname to check (case-insensitive)

    Returns:
        Tuple of (should_block, reason)
    """
    hostname_lower = hostname.lower().strip(".")

    # Check exact matches
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

    Blocked targets:
    - Non-http/https schemes (file://, ftp://, etc.)
    - Localhost and loopback addresses (127.0.0.1, ::1)
    - Link-local addresses including cloud metadata (169.254.0.0/16)
    - Hostnames like "localhost", "metadata.google.internal"

    NOT blocked by default (for backwards compatibility):
    - Private network IPs (10.x.x.x, 172.16-31.x.x, 192.168.x.x)
    - Set LLMROUTER_BLOCK_PRIVATE_IPS=true to also block these

    Args:
        url: The URL to validate
        resolve_dns: If True, resolve hostname to IP and check the IP too
        allow_private_ips: Override env var; if False, also block RFC1918 IPs

    Returns:
        The original URL if validation passes

    Raises:
        SSRFBlockedError: If the URL is blocked due to SSRF risk
        ValueError: If the URL is malformed
    """
    if not url:
        raise ValueError("URL cannot be empty")

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

    # Check blocked hostnames
    blocked, reason = _is_hostname_blocked(hostname)
    if blocked:
        verbose_proxy_logger.warning(f"SSRF: Blocked URL due to hostname: {url}")
        raise SSRFBlockedError(url, reason)

    # Check if hostname is an IP address directly
    blocked, reason = _is_ip_blocked(hostname)
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
                blocked, reason = _is_ip_blocked(ip_str)
                if blocked:
                    verbose_proxy_logger.warning(
                        f"SSRF: Blocked URL due to resolved IP {ip_str}: {url}"
                    )
                    raise SSRFBlockedError(
                        url, f"resolved IP {ip_str} is blocked: {reason}"
                    )

                # Check private IPs if configured to block them
                if allow_private_ips is None:
                    allow_private_ips = (
                        os.getenv("LLMROUTER_BLOCK_PRIVATE_IPS", "false").lower()
                        != "true"
                    )

                if not allow_private_ips:
                    try:
                        ip = ipaddress.ip_address(ip_str)
                        if (
                            ip.is_private and not ip.is_loopback
                        ):  # loopback already caught above
                            verbose_proxy_logger.warning(
                                f"SSRF: Blocked URL due to private IP {ip_str}: {url}"
                            )
                            raise SSRFBlockedError(
                                url,
                                f"private IP {ip_str} is blocked (LLMROUTER_BLOCK_PRIVATE_IPS=true)",
                            )
                    except ValueError:
                        pass  # Not an IP, skip

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
