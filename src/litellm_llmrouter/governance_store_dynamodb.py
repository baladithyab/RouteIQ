"""DynamoDB-backed governance/key/quota state store (RouteIQ-a865).

An alternative durable backend for the RouteIQ control plane that implements the
same surface as the Aurora-backed :class:`~litellm_llmrouter.governance_store.GovernanceStore`
(orgs / workspaces / keys / usage-policies / guardrails / prompts CRUD + load),
but persists rows to a single DynamoDB table instead of Postgres. This lets a
serverless / NoSQL deployment carry governance state without provisioning
Aurora.

Design (additive, gated, default-off, cred-free testable):

* DEFAULT-OFF. Selected only when ``ROUTEIQ_GOVERNANCE_BACKEND=dynamodb``; the
  default (``file``) keeps the file-backed in-memory path unchanged, so a
  default deployment is byte-stable.

* Single-table design. One table (``ROUTEIQ_GOVERNANCE_DDB_TABLE``, default
  ``routeiq-governance``) with a composite key:
    - ``pk`` = entity type (``ORG`` / ``WORKSPACE`` / ``KEY`` / ``POLICY`` /
      ``GUARDRAIL`` / ``PROMPT``)
    - ``sk`` = the entity's id
  Each item stores the full Pydantic model as a JSON ``payload`` attribute
  (drift-free ``model_validate`` round-trip on load) plus a ``workspace_id``
  attribute for scope filtering.

* Same disabled contract as the Aurora store: when not selected, ``enabled`` is
  ``False`` and every method is a fail-open no-op (callers degrade to the
  in-memory + JSON path ``governance.py`` already ships).

* Fail-open. Every read/write swallows errors (log, never raise) so a backend
  blip cannot wedge a CRUD mutation -- identical to the Aurora store.

The boto3 resource is created lazily so importing this module has no AWS
dependency; unit tests inject a fake table without creds.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("litellm_llmrouter.governance_store_dynamodb")

if TYPE_CHECKING:  # avoid import cycle at runtime; only needed for typing
    from datetime import datetime

    from .governance import KeyGovernance, OrgConfig, WorkspaceConfig
    from .guardrail_policies import GuardrailPolicy
    from .prompt_management import PromptDefinition
    from .usage_policies import UsagePolicy

# Partition-key (entity-type) constants.
PK_ORG = "ORG"
PK_WORKSPACE = "WORKSPACE"
PK_KEY = "KEY"
PK_POLICY = "POLICY"
PK_GUARDRAIL = "GUARDRAIL"
PK_PROMPT = "PROMPT"
# Spend partitions: one item per (scope, period, period_start) for the scope-level
# rollup, and one per (model, scope, period, period_start) for the per-model
# chargeback breakdown -- the DynamoDB mirror of the Aurora ``governance_spend`` /
# ``governance_spend_model`` ledgers. Distinct pk values keep the spend ledger out
# of the entity (ORG/WORKSPACE/...) partitions a CRUD ``load_all_*`` query scans.
PK_SPEND = "SPEND"
PK_SPEND_MODEL = "SPEND_MODEL"

_DEFAULT_TABLE = "routeiq-governance"


def governance_backend() -> str:
    """The selected governance backend: ``file`` (default) or ``dynamodb``."""
    return os.getenv("ROUTEIQ_GOVERNANCE_BACKEND", "file").strip().lower()


def dynamodb_backend_enabled() -> bool:
    """True when the DynamoDB governance backend is selected (default OFF)."""
    return governance_backend() == "dynamodb"


class DynamoDBGovernanceStore:
    """DynamoDB-backed durable governance store (same surface as GovernanceStore).

    Disabled (fail-open no-op) unless ``ROUTEIQ_GOVERNANCE_BACKEND=dynamodb``.
    """

    def __init__(
        self, table_name: str | None = None, region: str | None = None
    ) -> None:
        self._enabled = dynamodb_backend_enabled()
        self._table_name = table_name or os.getenv(
            "ROUTEIQ_GOVERNANCE_DDB_TABLE", _DEFAULT_TABLE
        )
        self._region = region or os.getenv("AWS_REGION", "us-east-1")
        self._table: Any = None

    @property
    def enabled(self) -> bool:
        """True when the DynamoDB backend is selected."""
        return self._enabled

    # -- internal -----------------------------------------------------------

    def _get_table(self) -> Any:
        """Lazily build the boto3 DynamoDB Table resource (no creds at import)."""
        if self._table is None:
            import boto3  # local import: optional dependency

            resource = boto3.resource("dynamodb", region_name=self._region)
            self._table = resource.Table(self._table_name)
        return self._table

    def _put(self, pk: str, sk: str, payload: dict, workspace_id: Any = None) -> None:
        """Upsert one item, fail-open (log, never raise)."""
        if not self._enabled:
            return
        try:
            self._get_table().put_item(
                Item={
                    "pk": pk,
                    "sk": sk,
                    "payload": json.dumps(payload),
                    "workspace_id": workspace_id or "",
                }
            )
        except Exception as exc:
            logger.error("DynamoDBGovernanceStore: put %s/%s failed: %s", pk, sk, exc)

    def _delete(self, pk: str, sk: str) -> None:
        """Delete one item, fail-open (log, never raise)."""
        if not self._enabled:
            return
        try:
            self._get_table().delete_item(Key={"pk": pk, "sk": sk})
        except Exception as exc:
            logger.error(
                "DynamoDBGovernanceStore: delete %s/%s failed: %s", pk, sk, exc
            )

    def _pk_condition(self, pk: str) -> Any:
        """Build the ``KeyConditionExpression`` for a partition query.

        Isolated so the boto3 ``conditions.Key`` import is the only place the
        boto3 dynamodb extension is touched -- unit tests inject a fake table +
        patch this to a plain value without needing boto3 installed.
        """
        from boto3.dynamodb.conditions import Key

        return Key("pk").eq(pk)

    def _query_items(self, pk: str) -> list[dict]:
        """Query all raw items for a partition (fail-open -> [])."""
        if not self._enabled:
            return []
        try:
            resp = self._get_table().query(
                KeyConditionExpression=self._pk_condition(pk)
            )
        except Exception as exc:
            logger.error("DynamoDBGovernanceStore: query %s failed: %s", pk, exc)
            return []
        return list(resp.get("Items", []))

    def _query_payloads(self, pk: str) -> list[dict]:
        """Query all items for a partition; return decoded payload dicts."""
        out: list[dict] = []
        for item in self._query_items(pk):
            payload = item.get("payload")
            if payload is None:
                continue
            data = json.loads(payload) if isinstance(payload, str) else payload
            out.append(data)
        return out

    def _add_counters(self, pk: str, sk: str, deltas: dict[str, float]) -> None:
        """Atomically accumulate numeric counters on one spend item (fail-open).

        Uses an ``ADD`` ``UpdateExpression`` so the accumulation is atomic in
        DynamoDB (no read-modify-write race) -- the NoSQL analogue of the Aurora
        store's ``ON CONFLICT DO UPDATE SET col = col + EXCLUDED.col``. ADD on an
        absent attribute treats it as ``0``, so the first write creates the item.
        Logs and swallows on any error (never raises) so a spend-table blip cannot
        wedge the response path -- identical to the Aurora store.
        """
        if not self._enabled:
            return
        # Drop zero deltas: an empty ADD set is invalid and a no-op write is waste.
        nonzero = {k: v for k, v in deltas.items() if v}
        if not nonzero:
            return
        try:
            expr = "ADD " + ", ".join(f"#{k} :{k}" for k in nonzero)
            names = {f"#{k}": k for k in nonzero}
            values = {f":{k}": v for k, v in nonzero.items()}
            self._get_table().update_item(
                Key={"pk": pk, "sk": sk},
                UpdateExpression=expr,
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=values,
            )
        except Exception as exc:
            logger.error(
                "DynamoDBGovernanceStore: spend add %s/%s failed: %s", pk, sk, exc
            )

    @staticmethod
    def _spend_sk(scope: str, period: str, period_start: "datetime") -> str:
        """Composite sort key for a scope-level spend bucket."""
        return f"{scope}#{period}#{period_start.isoformat()}"

    @staticmethod
    def _model_spend_sk(
        model: str, scope: str, period: str, period_start: "datetime"
    ) -> str:
        """Composite sort key for a per-model spend bucket."""
        return f"{model}#{scope}#{period}#{period_start.isoformat()}"

    # -- Org ops ------------------------------------------------------------

    async def upsert_org(self, org: "OrgConfig") -> None:
        self._put(PK_ORG, org.org_id, org.model_dump(mode="json"))

    async def delete_org(self, org_id: str) -> None:
        self._delete(PK_ORG, org_id)

    async def load_all_orgs(self) -> list["OrgConfig"]:
        if not self._enabled:
            return []
        from .governance import OrgConfig

        return [OrgConfig.model_validate(d) for d in self._query_payloads(PK_ORG)]

    # -- Workspace ops ------------------------------------------------------

    async def upsert_workspace(self, ws: "WorkspaceConfig") -> None:
        self._put(
            PK_WORKSPACE, ws.workspace_id, ws.model_dump(mode="json"), ws.workspace_id
        )

    async def delete_workspace(self, workspace_id: str) -> None:
        self._delete(PK_WORKSPACE, workspace_id)

    async def load_all_workspaces(self) -> list["WorkspaceConfig"]:
        if not self._enabled:
            return []
        from .governance import WorkspaceConfig

        return [
            WorkspaceConfig.model_validate(d)
            for d in self._query_payloads(PK_WORKSPACE)
        ]

    # -- Key ops ------------------------------------------------------------

    async def upsert_key(self, kg: "KeyGovernance") -> None:
        self._put(PK_KEY, kg.key_id, kg.model_dump(mode="json"), kg.workspace_id)

    async def delete_key(self, key_id: str) -> None:
        self._delete(PK_KEY, key_id)

    async def load_all_keys(self) -> list["KeyGovernance"]:
        if not self._enabled:
            return []
        from .governance import KeyGovernance

        return [KeyGovernance.model_validate(d) for d in self._query_payloads(PK_KEY)]

    # -- Policy ops ---------------------------------------------------------

    async def upsert_policy(self, p: "UsagePolicy") -> None:
        self._put(PK_POLICY, p.policy_id, p.model_dump(mode="json"), p.workspace_id)

    async def delete_policy(self, policy_id: str) -> None:
        self._delete(PK_POLICY, policy_id)

    async def load_all_policies(self) -> list["UsagePolicy"]:
        if not self._enabled:
            return []
        from .usage_policies import UsagePolicy

        return [UsagePolicy.model_validate(d) for d in self._query_payloads(PK_POLICY)]

    # -- Guardrail policy ops -----------------------------------------------

    async def upsert_guardrail(self, g: "GuardrailPolicy") -> None:
        self._put(
            PK_GUARDRAIL, g.guardrail_id, g.model_dump(mode="json"), g.workspace_id
        )

    async def delete_guardrail(self, guardrail_id: str) -> None:
        self._delete(PK_GUARDRAIL, guardrail_id)

    async def load_all_guardrails(self) -> list["GuardrailPolicy"]:
        if not self._enabled:
            return []
        from .guardrail_policies import GuardrailPolicy

        return [
            GuardrailPolicy.model_validate(d)
            for d in self._query_payloads(PK_GUARDRAIL)
        ]

    # -- Prompt ops ---------------------------------------------------------

    async def upsert_prompt(self, storage_key: str, p: "PromptDefinition") -> None:
        self._put(PK_PROMPT, storage_key, p.model_dump(mode="json"), p.workspace_id)

    async def delete_prompt(self, storage_key: str) -> None:
        self._delete(PK_PROMPT, storage_key)

    async def load_all_prompts(self) -> list[tuple[str, "PromptDefinition"]]:
        if not self._enabled:
            return []
        from .prompt_management import PromptDefinition

        out: list[tuple[str, PromptDefinition]] = []
        for item in self._query_items(PK_PROMPT):
            payload = item.get("payload")
            if payload is None:
                continue
            data = json.loads(payload) if isinstance(payload, str) else payload
            out.append((item["sk"], PromptDefinition.model_validate(data)))
        return out

    # -- Spend ops (the durable budget system-of-record; DynamoDB mirror) ----
    #
    # These mirror the Aurora ``GovernanceStore`` spend interface 1:1 so the
    # spend write/read path (``governance.record_governance_spend`` ->
    # ``store.record_spend`` / ``GovernanceEngine._get_durable_spend`` ->
    # ``store.get_spend`` / ``routes/config.get_spend_report`` ->
    # ``store.aggregate_spend_report``) works against either backend. With
    # ``ROUTEIQ_GOVERNANCE_BACKEND=dynamodb`` those callsites see ``enabled is
    # True`` and INVOKE these methods; without them the store would raise
    # ``AttributeError`` and break (not degrade) the spend path. Every method is
    # fail-open (never raises; returns safe empties when the table/feature is
    # unavailable) -- identical to the Aurora store's no-op-when-disabled contract.

    async def record_spend(
        self,
        scope: str,
        scope_type: str,
        period: str,
        period_start: "datetime",
        *,
        cost: float = 0.0,
        tokens: int = 0,
        requests: int = 1,
    ) -> None:
        """Atomically accumulate spend/tokens/requests for a scope+period.

        No-op when the store is disabled. Mirrors the Aurora
        :meth:`~litellm_llmrouter.governance_store.GovernanceStore.record_spend`
        signature; the atomic ``ADD`` keeps the accumulation race-free.
        """
        if not self._enabled:
            return
        sk = self._spend_sk(scope, period, period_start)
        # scope/scope_type live as plain attributes so a future scan can read
        # them back; the SET of them is folded into the same atomic update via a
        # separate, idempotent put-of-metadata is avoided -- ADD-only keeps it
        # race-free, and the report reconstructs scope/scope_type from the sk + a
        # stored attribute written on first touch below.
        self._add_counters(
            PK_SPEND,
            sk,
            {
                "spend_usd": float(cost or 0.0),
                "request_count": int(requests or 0),
                "total_tokens": int(tokens or 0),
            },
        )
        # Stamp the descriptive (non-counter) attributes so aggregate_spend_report
        # can group by them. Fail-open like every other write.
        self._stamp_spend_attrs(
            PK_SPEND, sk, {"scope": scope, "scope_type": scope_type, "period": period}
        )

    def _stamp_spend_attrs(self, pk: str, sk: str, attrs: dict[str, Any]) -> None:
        """Idempotently set descriptive (non-counter) attributes (fail-open).

        Separate from the atomic counter ADD so the SET of immutable descriptors
        (scope / scope_type / model) does not interfere with the race-free ADD.
        """
        if not self._enabled or not attrs:
            return
        try:
            expr = "SET " + ", ".join(f"#{k} = :{k}" for k in attrs)
            names = {f"#{k}": k for k in attrs}
            values = {f":{k}": v for k, v in attrs.items()}
            self._get_table().update_item(
                Key={"pk": pk, "sk": sk},
                UpdateExpression=expr,
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=values,
            )
        except Exception as exc:
            logger.error(
                "DynamoDBGovernanceStore: spend stamp %s/%s failed: %s", pk, sk, exc
            )

    async def get_spend(
        self, scope: str, period: str, period_start: "datetime"
    ) -> float:
        """Read durable spend for a scope+period. Returns 0.0 when disabled/absent."""
        if not self._enabled:
            return 0.0
        sk = self._spend_sk(scope, period, period_start)
        try:
            resp = self._get_table().get_item(Key={"pk": PK_SPEND, "sk": sk})
        except Exception as exc:
            logger.error(
                "DynamoDBGovernanceStore: get_spend %s/%s failed: %s",
                PK_SPEND,
                sk,
                exc,
            )
            return 0.0
        item = resp.get("Item") or {}
        try:
            return float(item.get("spend_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    async def record_model_spend(
        self,
        model: str,
        scope: str,
        scope_type: str,
        period: str,
        period_start: "datetime",
        *,
        cost: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        requests: int = 1,
    ) -> None:
        """Atomically accumulate per-model spend for the chargeback breakdown.

        The companion of :meth:`record_spend` carrying the model dimension the
        scope-level rollup lacks. No-op when disabled; atomic ``ADD`` keeps the
        accumulation race-free. Mirrors the Aurora store signature.
        """
        if not self._enabled:
            return
        sk = self._model_spend_sk(model, scope, period, period_start)
        self._add_counters(
            PK_SPEND_MODEL,
            sk,
            {
                "spend_usd": float(cost or 0.0),
                "request_count": int(requests or 0),
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
            },
        )
        self._stamp_spend_attrs(
            PK_SPEND_MODEL,
            sk,
            {
                "model": model,
                "scope": scope,
                "scope_type": scope_type,
                "period": period,
            },
        )

    async def aggregate_spend_report(self, *, since: "datetime | None" = None) -> dict:
        """Aggregate the durable spend ledger into a chargeback report.

        Returns real totals plus per-scope and per-model breakdowns and daily
        rollups, computed in Python from the ``PK_SPEND`` / ``PK_SPEND_MODEL``
        partition items. Returns an empty (zero) report when disabled or when no
        spend has been recorded -- byte-identical in shape to the Aurora store's
        report so the caller (``get_spend_report``) renders it either way.

        Args:
            since: Optional lower bound on ``period_start`` -- when provided only
                buckets at/after this instant are summed (the sk encodes the
                ISO ``period_start`` as its trailing segment).
        """
        empty = {
            "total_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_requests": 0,
            "by_scope": [],
            "by_model": [],
            "daily": [],
        }
        if not self._enabled:
            return empty

        since_iso = since.isoformat() if since is not None else None

        def _ps_from_sk(sk: str) -> str:
            # period_start is the trailing ISO segment of the composite sk.
            return sk.rsplit("#", 1)[-1] if sk else ""

        def _num(v: Any) -> float:
            try:
                return float(v or 0)
            except (TypeError, ValueError):
                return 0.0

        # -- Scope-level rollup -> totals + by_scope + daily. -------------------
        by_scope_acc: dict[tuple[str, str], dict] = {}
        daily_acc: dict[str, dict] = {}
        for item in self._query_items(PK_SPEND):
            ps_iso = _ps_from_sk(str(item.get("sk", "")))
            if since_iso is not None and ps_iso < since_iso:
                continue
            scope = str(item.get("scope", "") or "")
            scope_type = str(item.get("scope_type", "") or "")
            spend = _num(item.get("spend_usd"))
            reqs = int(_num(item.get("request_count")))
            toks = int(_num(item.get("total_tokens")))
            bucket = by_scope_acc.setdefault(
                (scope, scope_type),
                {
                    "scope": scope,
                    "scope_type": scope_type,
                    "spend_usd": 0.0,
                    "request_count": 0,
                    "total_tokens": 0,
                },
            )
            bucket["spend_usd"] += spend
            bucket["request_count"] += reqs
            bucket["total_tokens"] += toks
            # Daily rollup: calendar-day truncation of the ISO period_start.
            day = ps_iso[:10] if ps_iso else ""
            drow = daily_acc.setdefault(
                day, {"date": day, "spend_usd": 0.0, "request_count": 0}
            )
            drow["spend_usd"] += spend
            drow["request_count"] += reqs

        # -- Per-model rollup -> by_model. --------------------------------------
        by_model_acc: dict[str, dict] = {}
        for item in self._query_items(PK_SPEND_MODEL):
            ps_iso = _ps_from_sk(str(item.get("sk", "")))
            if since_iso is not None and ps_iso < since_iso:
                continue
            model = str(item.get("model", "") or "")
            mbucket = by_model_acc.setdefault(
                model,
                {
                    "model": model,
                    "spend_usd": 0.0,
                    "request_count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            )
            mbucket["spend_usd"] += _num(item.get("spend_usd"))
            mbucket["request_count"] += int(_num(item.get("request_count")))
            mbucket["input_tokens"] += int(_num(item.get("input_tokens")))
            mbucket["output_tokens"] += int(_num(item.get("output_tokens")))

        by_scope = sorted(
            by_scope_acc.values(), key=lambda r: r["spend_usd"], reverse=True
        )
        by_model = sorted(
            by_model_acc.values(), key=lambda r: r["spend_usd"], reverse=True
        )
        daily = sorted(daily_acc.values(), key=lambda r: r["date"])

        return {
            "total_cost_usd": round(sum(s["spend_usd"] for s in by_scope), 6),
            "total_input_tokens": sum(m["input_tokens"] for m in by_model),
            "total_output_tokens": sum(m["output_tokens"] for m in by_model),
            "total_requests": sum(s["request_count"] for s in by_scope),
            "by_scope": by_scope,
            "by_model": by_model,
            "daily": daily,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_ddb_store: DynamoDBGovernanceStore | None = None


def get_dynamodb_governance_store() -> DynamoDBGovernanceStore:
    """Get or create the DynamoDB governance store singleton."""
    global _ddb_store
    if _ddb_store is None:
        _ddb_store = DynamoDBGovernanceStore()
    return _ddb_store


def reset_dynamodb_governance_store() -> None:
    """Reset the DynamoDB governance store singleton (for testing)."""
    global _ddb_store
    _ddb_store = None


__all__ = [
    "PK_ORG",
    "PK_WORKSPACE",
    "PK_KEY",
    "PK_POLICY",
    "PK_GUARDRAIL",
    "PK_PROMPT",
    "PK_SPEND",
    "PK_SPEND_MODEL",
    "governance_backend",
    "dynamodb_backend_enabled",
    "DynamoDBGovernanceStore",
    "get_dynamodb_governance_store",
    "reset_dynamodb_governance_store",
]
