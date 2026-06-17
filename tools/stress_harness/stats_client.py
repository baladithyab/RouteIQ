"""Async client for RouteIQ's control-plane stats surfaces (RouteIQ-4f19 (d)).

The harness reads the ACTIVE strategy + org-wide distributions from RouteIQ's
own admin surfaces — NOT from Envoy headers (RouteIQ has none). Three endpoints:

  * ``GET /api/v1/routeiq/routing/config`` -> ``active_strategy`` +
    ``available_strategies`` (the surface that NAMES the running strategy).
  * ``GET /api/v1/routeiq/stats/global``   -> ``model_distribution`` +
    ``strategy_distribution`` + ``total_decisions`` (org-wide rollups).
  * ``GET /api/v1/routeiq/routing/stats``  -> ``total_decisions`` +
    ``strategy_distribution`` (routing-decision accumulator totals).

These are admin-tier endpoints, so they take an ``X-Admin-API-Key`` header when
an admin key is supplied (separate from the data-plane Bearer key the
``RouterClient`` uses). The reader is DEFENSIVE: it tolerates each endpoint being
absent, unauthorized, or shaped slightly differently across RouteIQ versions
(e.g. ``active_strategy`` living on ``/routing/config`` vs ``/routing/stats``),
and merges whatever it can into one ``RouteIQStats`` snapshot, annotating what it
could not read rather than raising. Fully exercised through ``httpx.MockTransport``
in tests — no live control plane, no credentials.
"""

from __future__ import annotations

from typing import Any

import httpx

from .models import RouteIQStats

ROUTING_CONFIG_PATH = "/api/v1/routeiq/routing/config"
GLOBAL_STATS_PATH = "/api/v1/routeiq/stats/global"
ROUTING_STATS_PATH = "/api/v1/routeiq/routing/stats"
#: Caller-scoped per-user stats (RouteIQ-2bbe). Returns the AUTHENTICATED
#: caller's own ``recent_models`` + ``decision_count`` for the key on the
#: request, so the per-user routing view is read with each user's own token.
ME_STATS_PATH = "/api/v1/routeiq/me/stats"

DEFAULT_TIMEOUT_S = 30.0


class RouteIQStatsClient:
    """Reads RouteIQ's control-plane stats surfaces into a ``RouteIQStats``.

    Takes an injectable ``httpx.AsyncClient`` so tests drive the whole parse path
    through a ``MockTransport`` with no live endpoint.
    """

    def __init__(
        self,
        base_url: str,
        *,
        admin_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        http_client: httpx.AsyncClient | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._admin_key = admin_key
        self._timeout = timeout
        self._injected = http_client

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if self._admin_key:
            # RouteIQ admin-tier auth header (control-plane, fail-closed).
            headers["x-admin-api-key"] = self._admin_key
        return headers

    async def _get_json(
        self, client: httpx.AsyncClient, path: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        """GET ``path`` and return ``(json_dict_or_None, error_note_or_None)``.

        Never raises: a non-2xx, a non-JSON body, or a transport error all return
        ``(None, note)`` so a partially-reachable control plane still yields a
        merged snapshot.
        """
        url = f"{self._base_url}{path}"
        try:
            resp = await client.get(url, headers=self._headers(), timeout=self._timeout)
        except Exception as exc:  # noqa: BLE001 — degrade, never abort
            return None, f"{path}: transport error ({type(exc).__name__}: {exc})"
        if resp.status_code < 200 or resp.status_code >= 300:
            return None, f"{path}: HTTP {resp.status_code}"
        try:
            data = resp.json()
        except (ValueError, UnicodeDecodeError):
            return None, f"{path}: non-JSON body"
        if not isinstance(data, dict):
            return None, f"{path}: unexpected JSON shape"
        return data, None

    async def fetch(self) -> RouteIQStats:
        """Fetch + merge all three surfaces into one ``RouteIQStats`` snapshot."""
        owns_client = self._injected is None
        client = self._injected or httpx.AsyncClient()
        try:
            config, config_err = await self._get_json(client, ROUTING_CONFIG_PATH)
            glob, glob_err = await self._get_json(client, GLOBAL_STATS_PATH)
            routing, routing_err = await self._get_json(client, ROUTING_STATS_PATH)
        finally:
            if owns_client:
                await client.aclose()
        return self._merge(
            config, glob, routing, errors=[config_err, glob_err, routing_err]
        )

    async def fetch_per_user_recent_models(
        self, user_tokens: dict[str, str]
    ) -> dict[str, list[str]]:
        """Read each user's AUTHORITATIVE ``recent_models`` from ``/me/stats``.

        ``/me/stats`` is caller-scoped: it returns ONLY the key on the request,
        so the harness reads the per-user routing view by GETting it once per
        synthetic user, each with that user's own data-plane bearer token
        (``user_tokens`` maps ``user_id -> token``). Returns ``user_id ->
        recent_models``; a user whose stats are unreachable / malformed is simply
        omitted (never raises). Empty ``user_tokens`` -> ``{}`` (no /me/stats
        call), so a single-tenant run pays nothing here (RouteIQ-2bbe).
        """
        if not user_tokens:
            return {}
        owns_client = self._injected is None
        client = self._injected or httpx.AsyncClient()
        out: dict[str, list[str]] = {}
        try:
            for user_id, token in user_tokens.items():
                models = await self._fetch_one_user(client, token)
                if models is not None:
                    out[user_id] = models
        finally:
            if owns_client:
                await client.aclose()
        return out

    async def _fetch_one_user(
        self, client: httpx.AsyncClient, token: str
    ) -> list[str] | None:
        """GET ``/me/stats`` with ``token`` and return its ``recent_models`` list
        (or None on any failure / non-list shape). Never raises."""
        url = f"{self._base_url}{ME_STATS_PATH}"
        headers = {"accept": "application/json", "authorization": f"Bearer {token}"}
        try:
            resp = await client.get(url, headers=headers, timeout=self._timeout)
        except Exception:  # noqa: BLE001 — degrade, never abort
            return None
        if resp.status_code < 200 or resp.status_code >= 300:
            return None
        try:
            data = resp.json()
        except (ValueError, UnicodeDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        recent = data.get("recent_models")
        if not isinstance(recent, list):
            return None
        return [str(m) for m in recent if isinstance(m, str)]

    def _merge(
        self,
        config: dict[str, Any] | None,
        glob: dict[str, Any] | None,
        routing: dict[str, Any] | None,
        *,
        errors: list[str | None],
    ) -> RouteIQStats:
        """Merge the three (possibly-absent) payloads, preferring the most
        authoritative source for each field.

        active_strategy: ``/routing/config`` first, then either stats payload
        (some RouteIQ versions surface it on the stats responses too).
        model_distribution: ``/stats/global`` (the only surface that carries it).
        strategy_distribution: ``/stats/global`` first, then ``/routing/stats``.
        total_decisions: ``/stats/global`` first, then ``/routing/stats``.
        """
        stats = RouteIQStats()
        stats.raw = {
            "routing_config": config or {},
            "stats_global": glob or {},
            "routing_stats": routing or {},
        }
        for note in errors:
            if note:
                stats.note(note)

        # active strategy — config is canonical, fall back to stats payloads.
        for source in (config, glob, routing):
            value = _opt_str(source.get("active_strategy")) if source else None
            if value:
                stats.active_strategy = value
                break
        if stats.active_strategy is None:
            stats.note(
                "active_strategy not reported by any surface "
                "(routing/config unreachable or field absent); "
                "report names the run-level strategy as <unknown>."
            )

        # available strategies — config carries the full set.
        if config:
            avail = config.get("available_strategies")
            if isinstance(avail, list):
                stats.available_strategies = [str(s) for s in avail]

        # model distribution — stats/global only.
        if glob:
            stats.model_distribution = _int_map(glob.get("model_distribution"))
        if not stats.model_distribution:
            stats.note(
                "model_distribution not reported by /stats/global "
                "(unreachable or empty); server-side model rollup unavailable, "
                "client-observed body-model distribution still computed."
            )

        # strategy distribution — stats/global first, then routing/stats.
        for source in (glob, routing):
            sd = _int_map(source.get("strategy_distribution")) if source else {}
            if sd:
                stats.strategy_distribution = sd
                break

        # total decisions — stats/global first, then routing/stats.
        for source in (glob, routing):
            td = _opt_int(source.get("total_decisions")) if source else None
            if td is not None:
                stats.total_decisions = td
                break

        return stats


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _opt_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _int_map(value: Any) -> dict[str, int]:
    """Coerce a ``{name: count}`` payload to ``dict[str, int]``, dropping junk."""
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        count = _opt_int(raw)
        if isinstance(key, str) and count is not None:
            out[key] = count
    return out
