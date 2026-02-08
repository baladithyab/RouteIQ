"""
Kubernetes health probe endpoints (/_health/*) and A2A Agent Card discovery.

- /_health/live: Liveness probe - doesn't check external deps (DB/Redis)
- /_health/ready: Readiness probe - checks optional deps with short timeouts
- /.well-known/agent.json: A2A Agent Card discovery (gateway-level)
- /.well-known/agents/{agent_id}.json: A2A Agent Card discovery (per-agent)
"""

import asyncio
import os

from fastapi import HTTPException

from ..auth import get_request_id, sanitize_error_response  # noqa: F401
from ..mcp_gateway import get_mcp_gateway
from ..resilience import get_drain_manager, get_circuit_breaker_manager
from . import health_router


# =============================================================================
# Kubernetes Health Probe Endpoints (/_health/*)
# =============================================================================
# These are minimal, unauthenticated endpoints for K8s probes.
# - /_health/live: Liveness probe - doesn't check external deps (DB/Redis)
# - /_health/ready: Readiness probe - checks optional deps with short timeouts
#
# Use these in K8s manifests instead of /health/* which may be auth-protected.


@health_router.get("/_health/live")
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint.

    This endpoint verifies the application process is alive and responsive.
    It does NOT check external dependencies (database, Redis, etc.) because
    liveness failures trigger pod restarts, not traffic rerouting.

    Returns:
        200 OK if the process is alive
    """
    return {"status": "alive", "service": "litellm-llmrouter"}


@health_router.get("/_health/ready")
async def readiness_probe():
    """
    Kubernetes readiness probe endpoint.

    This endpoint verifies the application is ready to accept traffic.
    It checks optional external dependencies (database, Redis) with short
    timeouts (2s) so the probe doesn't hang.

    If a dependency is not configured, it's not checked (still returns ready).
    If a dependency is configured but unreachable, returns 503.
    If the server is draining (graceful shutdown), returns 503.
    If any circuit breaker is open, returns 200 but with degraded status.

    Returns:
        200 OK if all configured dependencies are healthy and not draining
        503 Service Unavailable if any configured dependency is unhealthy or draining
    """
    request_id = get_request_id() or "unknown"
    checks = {}
    is_ready = True
    is_degraded = False

    # Check drain status first (highest priority)
    drain_manager = get_drain_manager()
    if drain_manager.is_draining:
        checks["drain_status"] = {
            "status": "draining",
            "active_requests": drain_manager.active_requests,
        }
        is_ready = False
    else:
        checks["drain_status"] = {
            "status": "accepting",
            "active_requests": drain_manager.active_requests,
        }

    # Check circuit breakers - degraded mode detection
    cb_manager = get_circuit_breaker_manager()
    cb_status = cb_manager.get_status()
    if cb_status["is_degraded"]:
        is_degraded = True
        checks["circuit_breakers"] = {
            "status": "degraded",
            "open_breakers": cb_status["degraded_components"],
            "breakers": {
                name: {
                    "state": info["state"],
                    "failure_count": info["failure_count"],
                    "time_until_retry": info["time_until_retry"],
                }
                for name, info in cb_status["breakers"].items()
            },
        }
    else:
        checks["circuit_breakers"] = {
            "status": "healthy",
            "breakers": {
                name: {"state": info["state"]}
                for name, info in cb_status["breakers"].items()
            }
            if cb_status["breakers"]
            else {},
        }

    # Check database if configured
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        db_breaker = cb_manager.database
        # If circuit breaker is open, skip the actual check
        if db_breaker.is_open:
            checks["database"] = {
                "status": "degraded",
                "circuit_breaker": "open",
                "time_until_retry": round(db_breaker.time_until_retry, 1),
            }
            # Degraded is not a readiness failure - service can still work with cache
            is_degraded = True
        else:
            try:
                # Import here to avoid circular imports and optional dependency
                import asyncpg

                # Use short timeout for health check
                conn = await asyncio.wait_for(
                    asyncpg.connect(db_url, timeout=2.0),
                    timeout=2.0,
                )
                await asyncio.wait_for(conn.execute("SELECT 1"), timeout=1.0)
                await conn.close()
                checks["database"] = {"status": "healthy"}
                # Record success in circuit breaker
                await db_breaker.record_success()
            except asyncio.TimeoutError:
                await db_breaker.record_failure("connection timeout")
                checks["database"] = {
                    "status": "unhealthy",
                    "error": "connection timeout",
                }
                is_ready = False
            except ImportError:
                # asyncpg not installed, try basic connectivity via litellm
                checks["database"] = {
                    "status": "skipped",
                    "reason": "asyncpg not installed",
                }
            except Exception as e:
                await db_breaker.record_failure(str(e))
                # Sanitize: don't leak exception details in health check response
                checks["database"] = {
                    "status": "unhealthy",
                    "error": "connection failed",
                }
                is_ready = False

    # Check Redis if configured
    redis_host = os.getenv("REDIS_HOST")
    if redis_host:
        redis_breaker = cb_manager.redis
        # If circuit breaker is open, skip the actual check
        if redis_breaker.is_open:
            checks["redis"] = {
                "status": "degraded",
                "circuit_breaker": "open",
                "time_until_retry": round(redis_breaker.time_until_retry, 1),
            }
            is_degraded = True
        else:
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            try:
                import redis.asyncio as aioredis

                r = aioredis.Redis(
                    host=redis_host,
                    port=redis_port,
                    socket_connect_timeout=2.0,
                    socket_timeout=2.0,
                )
                await asyncio.wait_for(r.ping(), timeout=2.0)
                await r.aclose()
                checks["redis"] = {"status": "healthy"}
                await redis_breaker.record_success()
            except asyncio.TimeoutError:
                await redis_breaker.record_failure("connection timeout")
                checks["redis"] = {"status": "unhealthy", "error": "connection timeout"}
                is_ready = False
            except ImportError:
                checks["redis"] = {
                    "status": "skipped",
                    "reason": "redis package not installed",
                }
            except Exception as e:
                await redis_breaker.record_failure(str(e))
                # Sanitize: don't leak exception details in health check response
                checks["redis"] = {"status": "unhealthy", "error": "connection failed"}
                is_ready = False

    # Check MCP gateway health if enabled
    if os.getenv("MCP_GATEWAY_ENABLED", "false").lower() == "true":
        try:
            gateway = get_mcp_gateway()
            if gateway.is_enabled():
                checks["mcp_gateway"] = {
                    "status": "healthy",
                    "servers": len(gateway.list_servers()),
                }
            else:
                checks["mcp_gateway"] = {"status": "disabled"}
        except Exception:
            # Sanitize: don't leak exception details
            checks["mcp_gateway"] = {"status": "unhealthy", "error": "check failed"}
            # MCP gateway failure is non-fatal for readiness
            # is_ready = False

    # Determine overall status
    if is_degraded and is_ready:
        status = "degraded"
    elif is_ready:
        status = "ready"
    else:
        status = "not_ready"

    response = {
        "status": status,
        "service": "litellm-llmrouter",
        "is_degraded": is_degraded,
        "checks": checks,
        "request_id": request_id,
    }

    if not is_ready:
        raise HTTPException(status_code=503, detail=response)

    return response


# =============================================================================
# A2A Agent Card Discovery (/.well-known/agent.json)
# =============================================================================
# These are unauthenticated endpoints per the A2A specification.
# They enable agent discovery by returning Agent Card metadata.


@health_router.get("/.well-known/agent.json")
async def get_gateway_agent_card():
    """
    Return the gateway-level A2A Agent Card.

    This is a public discovery endpoint per the A2A specification.
    No authentication required.

    Returns the gateway's Agent Card with capabilities, skills (derived
    from registered agents), and authentication requirements.
    """
    from ..a2a_gateway import get_a2a_gateway

    gateway = get_a2a_gateway()

    base_url = os.getenv("A2A_BASE_URL", "")

    return gateway.get_gateway_agent_card(base_url=base_url)


@health_router.get("/.well-known/agents/{agent_id}.json")
async def get_agent_card_by_id(agent_id: str):
    """
    Return the A2A Agent Card for a specific registered agent.

    This is a public discovery endpoint per the A2A specification.
    No authentication required.

    Args:
        agent_id: The ID of the agent to get the card for.
    """
    from ..a2a_gateway import get_a2a_gateway

    gateway = get_a2a_gateway()
    card = gateway.get_agent_card(agent_id)

    if card is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "agent_not_found",
                "message": f"Agent '{agent_id}' not found",
            },
        )

    return card
