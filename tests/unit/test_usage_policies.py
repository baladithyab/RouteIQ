"""Unit tests for the Usage Policy Engine (usage_policies.py).

Tests cover:
- Policy CRUD operations
- Condition matching (fnmatch wildcards, dotted paths, exclusions)
- Group-by key construction
- Redis key generation with time-bucket rollover
- Pre-request evaluation (requests-type policies)
- Post-response usage recording (cost/token-type policies)
- Usage queries and resets
- Fail-open / fail-closed behaviour when Redis is unavailable
- Singleton management
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter.usage_policies import (
    LimitPeriod,
    LimitType,
    PolicyAction,
    PolicyEvaluation,
    UsagePolicy,
    UsagePolicyEngine,
    get_usage_policy_engine,
    is_usage_policies_enabled,
    reset_usage_policy_engine,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_policy(**overrides) -> UsagePolicy:
    """Create a UsagePolicy with sensible defaults."""
    defaults = {
        "policy_id": "test-policy",
        "name": "Test Policy",
        "enabled": True,
        "conditions": {},
        "exclusions": {},
        "group_by": [],
        "limit_type": LimitType.REQUESTS,
        "limit_value": 100,
        "limit_period": LimitPeriod.MINUTE,
        "action": PolicyAction.DENY,
        "alert_threshold": 0.8,
        "priority": 100,
    }
    defaults.update(overrides)
    return UsagePolicy(**defaults)


def _make_context(**overrides) -> dict:
    """Create a minimal request context dict."""
    defaults = {
        "api_key": "sk-test-123",
        "model": "gpt-4o",
        "provider": "openai",
        "workspace_id": None,
        "metadata": {"_user": "alice", "_team": "engineering"},
    }
    defaults.update(overrides)
    return defaults


# =============================================================================
# Policy CRUD
# =============================================================================


class TestPolicyCRUD:
    def test_add_and_get(self):
        engine = UsagePolicyEngine()
        p = _make_policy()
        engine.add_policy(p)

        got = engine.get_policy("test-policy")
        assert got is not None
        assert got.name == "Test Policy"
        assert got.created_at is not None
        assert got.updated_at is not None

    def test_add_overwrites(self):
        engine = UsagePolicyEngine()
        p1 = _make_policy(name="v1")
        engine.add_policy(p1)

        p2 = _make_policy(name="v2")
        engine.add_policy(p2)

        got = engine.get_policy("test-policy")
        assert got.name == "v2"

    def test_remove(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy())
        assert engine.remove_policy("test-policy") is True
        assert engine.get_policy("test-policy") is None

    def test_remove_missing(self):
        engine = UsagePolicyEngine()
        assert engine.remove_policy("nope") is False

    def test_list_empty(self):
        engine = UsagePolicyEngine()
        assert engine.list_policies() == []

    def test_list_sorted_by_priority(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy(policy_id="b", priority=200))
        engine.add_policy(_make_policy(policy_id="a", priority=50))
        engine.add_policy(_make_policy(policy_id="c", priority=100))

        ids = [p.policy_id for p in engine.list_policies()]
        assert ids == ["a", "c", "b"]

    def test_list_filter_workspace(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy(policy_id="global", workspace_id=None))
        engine.add_policy(_make_policy(policy_id="ws1", workspace_id="ws-1"))
        engine.add_policy(_make_policy(policy_id="ws2", workspace_id="ws-2"))

        # No filter — all
        assert len(engine.list_policies()) == 3

        # Filter by ws-1 → includes global + ws-1
        ws1 = engine.list_policies(workspace_id="ws-1")
        assert {p.policy_id for p in ws1} == {"global", "ws1"}


# =============================================================================
# Condition Matching
# =============================================================================


class TestConditionMatching:
    def test_empty_conditions_match_all(self):
        engine = UsagePolicyEngine()
        policy = _make_policy(conditions={})
        assert engine._matches_conditions(policy, _make_context()) is True

    def test_exact_match(self):
        engine = UsagePolicyEngine()
        policy = _make_policy(conditions={"model": "gpt-4o"})
        assert engine._matches_conditions(policy, _make_context(model="gpt-4o")) is True
        assert (
            engine._matches_conditions(policy, _make_context(model="claude-3")) is False
        )

    def test_wildcard_match(self):
        engine = UsagePolicyEngine()
        policy = _make_policy(conditions={"model": "gpt-*"})
        assert engine._matches_conditions(policy, _make_context(model="gpt-4o")) is True
        assert (
            engine._matches_conditions(policy, _make_context(model="gpt-3.5-turbo"))
            is True
        )
        assert (
            engine._matches_conditions(policy, _make_context(model="claude-3")) is False
        )

    def test_dotted_path(self):
        engine = UsagePolicyEngine()
        policy = _make_policy(conditions={"metadata._team": "engineering"})
        ctx = _make_context(metadata={"_team": "engineering", "_user": "bob"})
        assert engine._matches_conditions(policy, ctx) is True

    def test_dotted_path_missing(self):
        engine = UsagePolicyEngine()
        policy = _make_policy(conditions={"metadata._role": "admin"})
        ctx = _make_context(metadata={"_team": "engineering"})
        assert engine._matches_conditions(policy, ctx) is False

    def test_case_insensitive(self):
        engine = UsagePolicyEngine()
        policy = _make_policy(conditions={"provider": "OPENAI"})
        assert (
            engine._matches_conditions(policy, _make_context(provider="openai")) is True
        )

    def test_multiple_conditions_and(self):
        engine = UsagePolicyEngine()
        policy = _make_policy(conditions={"model": "gpt-*", "provider": "openai"})
        assert (
            engine._matches_conditions(
                policy, _make_context(model="gpt-4o", provider="openai")
            )
            is True
        )
        assert (
            engine._matches_conditions(
                policy, _make_context(model="gpt-4o", provider="azure")
            )
            is False
        )

    def test_exclusions(self):
        engine = UsagePolicyEngine()
        policy = _make_policy(
            conditions={},
            exclusions={"metadata._role": "admin"},
        )
        # No role → not excluded
        assert (
            engine._matches_conditions(
                policy, _make_context(metadata={"_user": "alice"})
            )
            is True
        )
        # Admin role → excluded
        assert (
            engine._matches_conditions(
                policy, _make_context(metadata={"_role": "admin"})
            )
            is False
        )

    def test_exclusion_wildcard(self):
        engine = UsagePolicyEngine()
        policy = _make_policy(exclusions={"api_key": "sk-admin-*"})
        assert (
            engine._matches_conditions(policy, _make_context(api_key="sk-admin-123"))
            is False
        )
        assert (
            engine._matches_conditions(policy, _make_context(api_key="sk-user-456"))
            is True
        )


# =============================================================================
# Group Key Construction
# =============================================================================


class TestGroupKey:
    def test_no_group_by(self):
        engine = UsagePolicyEngine()
        policy = _make_policy(group_by=[])
        assert engine._build_group_key(policy, _make_context()) == "__global__"

    def test_single_dimension(self):
        engine = UsagePolicyEngine()
        policy = _make_policy(group_by=["metadata._user"])
        ctx = _make_context(metadata={"_user": "alice"})
        assert engine._build_group_key(policy, ctx) == "metadata._user=alice"

    def test_multi_dimension(self):
        engine = UsagePolicyEngine()
        policy = _make_policy(group_by=["metadata._user", "model"])
        ctx = _make_context(model="gpt-4o", metadata={"_user": "alice"})
        key = engine._build_group_key(policy, ctx)
        assert key == "metadata._user=alice|model=gpt-4o"

    def test_missing_dimension(self):
        engine = UsagePolicyEngine()
        policy = _make_policy(group_by=["metadata._role"])
        ctx = _make_context(metadata={"_user": "alice"})
        key = engine._build_group_key(policy, ctx)
        assert key == "metadata._role=__none__"


# =============================================================================
# Redis Key Generation
# =============================================================================


class TestRedisKey:
    def test_key_format(self):
        key = UsagePolicyEngine._get_redis_key(
            "pol-1", "__global__", LimitPeriod.MINUTE
        )
        assert key.startswith("routeiq:usage:pol-1:__global__:")

    def test_lifetime_key_always_zero_bucket(self):
        key = UsagePolicyEngine._get_redis_key("pol-1", "g", LimitPeriod.NONE)
        assert key.endswith(":0")

    def test_minute_key_has_bucket(self):
        key = UsagePolicyEngine._get_redis_key("pol-1", "g", LimitPeriod.MINUTE)
        parts = key.split(":")
        bucket = int(parts[-1])
        expected = int(time.time() // 60)
        assert abs(bucket - expected) <= 1  # Allow 1 second drift


# =============================================================================
# Evaluate (with mocked Redis)
# =============================================================================


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_skip_disabled_policy(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy(enabled=False))
        results = await engine.evaluate(_make_context())
        assert results == []

    @pytest.mark.asyncio
    async def test_skip_non_matching_policy(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy(conditions={"model": "claude-*"}))
        results = await engine.evaluate(_make_context(model="gpt-4o"))
        assert results == []

    @pytest.mark.asyncio
    async def test_request_limit_allowed(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy(limit_type=LimitType.REQUESTS, limit_value=100))

        # Mock Redis Lua script
        mock_script = AsyncMock(return_value=[b"1", 1, 59])
        engine._redis = MagicMock()
        engine._check_incr_script = mock_script
        engine._read_script = AsyncMock()

        results = await engine.evaluate(_make_context())
        assert len(results) == 1
        assert results[0].matched is True
        assert results[0].exceeded is False
        assert results[0].current_usage == 1.0

    @pytest.mark.asyncio
    async def test_request_limit_exceeded(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy(limit_type=LimitType.REQUESTS, limit_value=10))

        mock_script = AsyncMock(return_value=[b"10", 0, 30])
        engine._redis = MagicMock()
        engine._check_incr_script = mock_script

        results = await engine.evaluate(_make_context())
        assert len(results) == 1
        assert results[0].exceeded is True

    @pytest.mark.asyncio
    async def test_cost_limit_read_only(self):
        engine = UsagePolicyEngine()
        engine.add_policy(
            _make_policy(
                limit_type=LimitType.COST,
                limit_value=50.0,
                limit_period=LimitPeriod.MONTH,
            )
        )

        mock_read = AsyncMock(return_value=[b"30.5", 2500000])
        engine._redis = MagicMock()
        engine._check_incr_script = AsyncMock()
        engine._read_script = mock_read

        results = await engine.evaluate(_make_context())
        assert len(results) == 1
        assert results[0].current_usage == 30.5
        assert results[0].exceeded is False
        assert results[0].usage_pct == pytest.approx(0.61, abs=0.01)

    @pytest.mark.asyncio
    async def test_alert_threshold(self):
        engine = UsagePolicyEngine()
        engine.add_policy(
            _make_policy(
                limit_type=LimitType.COST,
                limit_value=100.0,
                alert_threshold=0.8,
            )
        )

        mock_read = AsyncMock(return_value=[b"85.0", 2500000])
        engine._redis = MagicMock()
        engine._read_script = mock_read

        results = await engine.evaluate(_make_context())
        assert results[0].alert_triggered is True

    @pytest.mark.asyncio
    async def test_no_alert_below_threshold(self):
        engine = UsagePolicyEngine()
        engine.add_policy(
            _make_policy(
                limit_type=LimitType.COST,
                limit_value=100.0,
                alert_threshold=0.8,
            )
        )

        mock_read = AsyncMock(return_value=[b"50.0", 2500000])
        engine._redis = MagicMock()
        engine._read_script = mock_read

        results = await engine.evaluate(_make_context())
        assert results[0].alert_triggered is False

    @pytest.mark.asyncio
    async def test_fail_open_no_redis(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy())
        # _redis stays None

        with patch.dict("os.environ", {"ROUTEIQ_USAGE_POLICIES_FAIL_MODE": "open"}):
            results = await engine.evaluate(_make_context())
        assert len(results) == 1
        assert results[0].exceeded is False
        assert results[0].error is not None
        assert "fail-open" in results[0].error

    @pytest.mark.asyncio
    async def test_fail_closed_no_redis(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy())

        with patch.dict("os.environ", {"ROUTEIQ_USAGE_POLICIES_FAIL_MODE": "closed"}):
            results = await engine.evaluate(_make_context())
        assert len(results) == 1
        assert results[0].exceeded is True
        assert "fail-closed" in results[0].error

    @pytest.mark.asyncio
    async def test_multiple_policies_evaluated(self):
        engine = UsagePolicyEngine()
        engine.add_policy(
            _make_policy(policy_id="p1", priority=1, limit_type=LimitType.REQUESTS)
        )
        engine.add_policy(
            _make_policy(policy_id="p2", priority=2, limit_type=LimitType.COST)
        )

        mock_check = AsyncMock(return_value=[b"5", 1, 55])
        mock_read = AsyncMock(return_value=[b"10.0", 2500000])
        engine._redis = MagicMock()
        engine._check_incr_script = mock_check
        engine._read_script = mock_read

        results = await engine.evaluate(_make_context())
        assert len(results) == 2
        assert results[0].policy_id == "p1"
        assert results[1].policy_id == "p2"

    @pytest.mark.asyncio
    async def test_dry_run_does_not_increment(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy(limit_type=LimitType.REQUESTS, limit_value=100))

        mock_script = AsyncMock(return_value=[b"0", 1, 59])
        engine._redis = MagicMock()
        engine._check_incr_script = mock_script

        await engine.evaluate(_make_context(), dry_run=True)
        # Verify dry_run=1 was passed as the 4th arg
        call_args = mock_script.call_args
        assert call_args.kwargs["args"][3] == 1  # dry_run flag


# =============================================================================
# Record Usage
# =============================================================================


class TestRecordUsage:
    @pytest.mark.asyncio
    async def test_record_cost(self):
        engine = UsagePolicyEngine()
        engine.add_policy(
            _make_policy(
                limit_type=LimitType.COST,
                limit_value=100.0,
                limit_period=LimitPeriod.MONTH,
            )
        )

        mock_incr = AsyncMock(return_value=b"0.05")
        engine._redis = MagicMock()
        engine._incr_script = mock_incr

        updated = await engine.record_usage(_make_context(), cost=0.05)
        assert updated == 1
        mock_incr.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_tokens(self):
        engine = UsagePolicyEngine()
        engine.add_policy(
            _make_policy(
                limit_type=LimitType.TOKENS,
                limit_value=1_000_000,
                limit_period=LimitPeriod.DAY,
            )
        )

        mock_incr = AsyncMock(return_value=b"500")
        engine._redis = MagicMock()
        engine._incr_script = mock_incr

        updated = await engine.record_usage(
            _make_context(), input_tokens=300, output_tokens=200
        )
        assert updated == 1

    @pytest.mark.asyncio
    async def test_record_skips_request_type(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy(limit_type=LimitType.REQUESTS))

        mock_incr = AsyncMock()
        engine._redis = MagicMock()
        engine._incr_script = mock_incr

        updated = await engine.record_usage(_make_context(), cost=0.01)
        assert updated == 0
        mock_incr.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_no_redis(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy(limit_type=LimitType.COST, limit_value=100))

        updated = await engine.record_usage(_make_context(), cost=0.01)
        assert updated == 0

    @pytest.mark.asyncio
    async def test_record_skips_non_matching(self):
        engine = UsagePolicyEngine()
        engine.add_policy(
            _make_policy(
                limit_type=LimitType.COST,
                conditions={"model": "claude-*"},
            )
        )

        mock_incr = AsyncMock()
        engine._redis = MagicMock()
        engine._incr_script = mock_incr

        updated = await engine.record_usage(_make_context(model="gpt-4o"), cost=0.01)
        assert updated == 0


# =============================================================================
# Usage Queries & Reset
# =============================================================================


class TestUsageQueries:
    @pytest.mark.asyncio
    async def test_get_usage(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy())

        mock_read = AsyncMock(return_value=[b"42.0", 30])
        engine._redis = MagicMock()
        engine._read_script = mock_read

        usage = await engine.get_usage("test-policy", "__global__")
        assert usage == 42.0

    @pytest.mark.asyncio
    async def test_get_usage_no_redis(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy())

        usage = await engine.get_usage("test-policy")
        assert usage == 0.0

    @pytest.mark.asyncio
    async def test_get_usage_missing_policy(self):
        engine = UsagePolicyEngine()
        usage = await engine.get_usage("nonexistent")
        assert usage == 0.0

    @pytest.mark.asyncio
    async def test_reset_usage(self):
        engine = UsagePolicyEngine()
        engine.add_policy(_make_policy())

        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock(return_value=1)
        engine._redis = mock_redis
        engine._read_script = AsyncMock()

        success = await engine.reset_usage("test-policy")
        assert success is True
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_usage_missing_policy(self):
        engine = UsagePolicyEngine()
        success = await engine.reset_usage("nonexistent")
        assert success is False

    @pytest.mark.asyncio
    async def test_get_policy_usage_summary(self):
        engine = UsagePolicyEngine()
        engine.add_policy(
            _make_policy(
                limit_type=LimitType.COST,
                limit_value=100.0,
                limit_period=LimitPeriod.MONTH,
            )
        )

        mock_read = AsyncMock(return_value=[b"45.0", 2500000])
        engine._redis = MagicMock()
        engine._read_script = mock_read

        summary = await engine.get_policy_usage_summary("test-policy")
        assert summary["policy_id"] == "test-policy"
        assert summary["global_usage"] == 45.0
        assert summary["usage_pct"] == pytest.approx(0.45, abs=0.01)


# =============================================================================
# PolicyEvaluation dataclass
# =============================================================================


class TestPolicyEvaluation:
    def test_to_dict(self):
        ev = PolicyEvaluation(
            policy_id="p1",
            policy_name="Test",
            matched=True,
            group_key="user=alice",
            current_usage=50.0,
            limit_value=100.0,
            usage_pct=0.5,
            action=PolicyAction.DENY,
            exceeded=False,
            alert_triggered=False,
        )
        d = ev.to_dict()
        assert d["policy_id"] == "p1"
        assert d["action"] == "deny"
        assert d["usage_pct"] == 0.5
        assert d["exceeded"] is False


# =============================================================================
# Singleton Management
# =============================================================================


class TestSingleton:
    def test_get_returns_same_instance(self):
        e1 = get_usage_policy_engine()
        e2 = get_usage_policy_engine()
        assert e1 is e2

    def test_reset_clears(self):
        e1 = get_usage_policy_engine()
        reset_usage_policy_engine()
        e2 = get_usage_policy_engine()
        assert e1 is not e2


class TestFeatureFlag:
    def test_disabled_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert is_usage_policies_enabled() is False

    def test_enabled(self):
        with patch.dict("os.environ", {"ROUTEIQ_USAGE_POLICIES_ENABLED": "true"}):
            assert is_usage_policies_enabled() is True

    def test_enabled_1(self):
        with patch.dict("os.environ", {"ROUTEIQ_USAGE_POLICIES_ENABLED": "1"}):
            assert is_usage_policies_enabled() is True


# =============================================================================
# LimitPeriod properties
# =============================================================================


class TestLimitPeriod:
    def test_seconds(self):
        assert LimitPeriod.MINUTE.seconds == 60
        assert LimitPeriod.HOUR.seconds == 3600
        assert LimitPeriod.DAY.seconds == 86400
        assert LimitPeriod.WEEK.seconds == 604800
        assert LimitPeriod.MONTH.seconds == 2592000
        assert LimitPeriod.NONE.seconds == 0

    def test_bucket_divisor(self):
        assert LimitPeriod.MINUTE.bucket_divisor == 60
        assert LimitPeriod.NONE.bucket_divisor == 1
