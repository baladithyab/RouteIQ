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
    "governance_backend",
    "dynamodb_backend_enabled",
    "DynamoDBGovernanceStore",
    "get_dynamodb_governance_store",
    "reset_dynamodb_governance_store",
]
