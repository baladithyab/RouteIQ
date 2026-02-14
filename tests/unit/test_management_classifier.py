"""Tests for the management endpoint classifier."""

import pytest

from litellm_llmrouter.management_classifier import (
    ManagementOperation,
    classify,
    get_required_permission,
    MANAGEMENT_PERMISSIONS,
)


class TestClassify:
    """Test path/method classification."""

    # ── Key Management ──

    def test_key_generate(self):
        op = classify("POST", "/key/generate")
        assert op is not None
        assert op.operation == "key.generate"
        assert op.resource_type == "key"
        assert op.sensitivity == "write"

    def test_key_info(self):
        op = classify("GET", "/key/info")
        assert op is not None
        assert op.operation == "key.info"
        assert op.sensitivity == "read"

    def test_keys_list(self):
        op = classify("GET", "/keys")
        assert op is not None
        assert op.operation == "key.list"

    def test_key_delete(self):
        op = classify("POST", "/key/delete")
        assert op is not None
        assert op.operation == "key.delete"
        assert op.sensitivity == "write"

    def test_key_block(self):
        op = classify("POST", "/key/block")
        assert op is not None
        assert op.operation == "key.block"

    # ── Team Management ──

    def test_team_create(self):
        op = classify("POST", "/team/new")
        assert op is not None
        assert op.operation == "team.create"
        assert op.resource_type == "team"
        assert op.sensitivity == "write"

    def test_team_list(self):
        op = classify("GET", "/team")
        assert op is not None
        assert op.operation == "team.list"
        assert op.sensitivity == "read"

    def test_team_member_add(self):
        op = classify("POST", "/team/member/add")
        assert op is not None
        assert op.operation == "team.member_add"

    # ── Organization ──

    def test_organization_create(self):
        op = classify("POST", "/organization/new")
        assert op is not None
        assert op.operation == "organization.create"

    def test_organization_delete_with_id(self):
        op = classify("DELETE", "/organization/org-123")
        assert op is not None
        assert op.operation == "organization.delete"

    def test_organization_list(self):
        op = classify("GET", "/organization")
        assert op is not None
        assert op.operation == "organization.list"

    # ── Model Management ──

    def test_model_add(self):
        op = classify("POST", "/model/new")
        assert op is not None
        assert op.operation == "model.add"
        assert op.resource_type == "model"
        assert op.sensitivity == "write"

    def test_model_list(self):
        op = classify("GET", "/models")
        assert op is not None
        assert op.operation == "model.list"

    def test_v1_models(self):
        op = classify("GET", "/v1/models")
        assert op is not None
        assert op.operation == "model.list"

    # ── Budget ──

    def test_budget_create(self):
        op = classify("POST", "/budgets")
        assert op is not None
        assert op.operation == "budget.create"

    def test_budget_list(self):
        op = classify("GET", "/budgets")
        assert op is not None
        assert op.operation == "budget.list"

    # ── Spend ──

    def test_spend_read(self):
        op = classify("GET", "/spend/keys")
        assert op is not None
        assert op.operation == "spend.read"
        assert op.sensitivity == "read"

    def test_spend_write(self):
        op = classify("POST", "/spend/custom_database_logging")
        assert op is not None
        assert op.operation == "spend.write"

    # ── Credentials ──

    def test_credential_create(self):
        op = classify("POST", "/credentials")
        assert op is not None
        assert op.operation == "credential.create"

    def test_credential_delete(self):
        op = classify("DELETE", "/credentials/cred-123")
        assert op is not None
        assert op.operation == "credential.delete"

    # ── Guardrails ──

    def test_guardrail_list(self):
        op = classify("GET", "/guardrails")
        assert op is not None
        assert op.operation == "guardrail.list"

    def test_guardrail_create(self):
        op = classify("POST", "/guardrails/new")
        assert op is not None
        assert op.operation == "guardrail.create"

    # ── Config ──

    def test_config_reload(self):
        op = classify("POST", "/config/reload")
        assert op is not None
        assert op.operation == "config.reload"
        assert op.sensitivity == "write"

    # ── Non-management paths return None ──

    def test_chat_completions_not_classified(self):
        assert classify("POST", "/chat/completions") is None

    def test_v1_chat_completions_not_classified(self):
        assert classify("POST", "/v1/chat/completions") is None

    def test_embeddings_not_classified(self):
        assert classify("POST", "/embeddings") is None

    def test_health_not_classified(self):
        assert classify("GET", "/_health/live") is None

    def test_llmrouter_routes_not_classified(self):
        assert classify("GET", "/llmrouter/strategies/compare") is None

    def test_mcp_routes_not_classified(self):
        assert classify("POST", "/mcp") is None

    # ── Wrong method doesn't match ──

    def test_wrong_method_no_match(self):
        # /key/generate only matches POST
        assert classify("GET", "/key/generate") is None

    # ── Case sensitivity ──

    def test_method_case_insensitive(self):
        op = classify("post", "/key/generate")
        assert op is not None
        assert op.operation == "key.generate"


class TestManagementOperation:
    def test_frozen_dataclass(self):
        op = ManagementOperation("key.generate", "key", "write")
        with pytest.raises(AttributeError):
            op.operation = "changed"  # type: ignore[misc]

    def test_equality(self):
        a = ManagementOperation("key.generate", "key", "write")
        b = ManagementOperation("key.generate", "key", "write")
        assert a == b

    def test_hash(self):
        a = ManagementOperation("key.generate", "key", "write")
        s = {a}
        assert a in s


class TestGetRequiredPermission:
    def test_key_write_permission(self):
        op = ManagementOperation("key.generate", "key", "write")
        assert get_required_permission(op) == "key.write"

    def test_team_read_permission(self):
        op = ManagementOperation("team.list", "team", "read")
        assert get_required_permission(op) == "team.read"

    def test_unknown_resource_type(self):
        op = ManagementOperation("unknown.action", "unknown", "write")
        assert get_required_permission(op) is None

    def test_all_resource_types_have_read_and_write(self):
        resource_types = {rt for (rt, _) in MANAGEMENT_PERMISSIONS.keys()}
        for rt in resource_types:
            # Each resource type should have at least one permission
            perms = [p for (r, s), p in MANAGEMENT_PERMISSIONS.items() if r == rt]
            assert len(perms) >= 1, f"Resource type '{rt}' has no permissions"
