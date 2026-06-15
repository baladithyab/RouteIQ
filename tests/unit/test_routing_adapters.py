"""Unit tests for the strategy-agnostic routing-adapter / MLOps framework.

Proves the framework is NOT bandit-coupled: the same feedback fan-out and the
same manifest contract drive a STUB classifier adapter as well as the bandit.
Covers: ``AdapterManifest`` defaults + train_mode validation, SemVer ABI
negotiation, the ``route = select_deployment`` alias (every existing
``RoutingStrategy`` satisfies ``RoutingAdapter``), the entry-point loader staging
(with monkeypatched entry_points incl. ABI/capability gates), the SECOND-strategy
attach (acceptance delta), and continuous-vs-one_time subscription behavior.
"""

from __future__ import annotations

from typing import Optional

from litellm_llmrouter.adapters.contract import (
    ADAPTER_API_VERSION,
    TRAIN_MODE_CONTINUOUS,
    TRAIN_MODE_ONE_TIME,
    AdapterManifest,
    ArtifactRef,
    RoutingAdapter,
    RoutingFeedback,
    _abi_compatible,
    attach_route_alias,
)
from litellm_llmrouter.adapters.loader import AdapterLoaderPlugin
from litellm_llmrouter.adapters.mlops import (
    MLOpsCoordinator,
    get_mlops_coordinator,
    wire_mlops_feedback_loop,
)
from litellm_llmrouter.gateway.plugin_manager import PluginCapability
from litellm_llmrouter.strategy_registry import (
    DefaultStrategy,
    RoutingContext,
    get_routing_registry,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _StubClassifierAdapter:
    """A SECOND, NON-bandit learning adapter (the acceptance-delta proof).

    Trivial ``update()`` that records calls. Declares ``learns=True`` so the
    generalized MLOps loop fans feedback to it identically to the bandit.
    """

    def __init__(self, train_mode: str = TRAIN_MODE_CONTINUOUS, name: str = "stub-clf"):
        self._name = name
        self._train_mode = train_mode
        self.update_calls: list[tuple[str, float]] = []
        self.reloaded: list[ArtifactRef] = []

    def declare_capabilities(self) -> AdapterManifest:
        return AdapterManifest(
            name=self._name,
            version="1.0",
            adapter_api_version=ADAPTER_API_VERSION,
            capabilities={PluginCapability.ROUTING_STRATEGY},
            learns=True,
            uses_artifact=True,
            train_mode=self._train_mode,
            artifact_kinds={"json"},
        )

    def route(self, ctx: RoutingContext) -> Optional[dict]:
        router = ctx.router
        deps = getattr(router, "model_list", [])
        return deps[0] if deps else None

    select_deployment = route  # also satisfy the ABC shape

    def update(self, model: str, score: float, **kwargs) -> None:
        self.update_calls.append((model, score))

    def reload(self, ref: ArtifactRef) -> bool:
        self.reloaded.append(ref)
        return True

    def validate(self):
        # stage_strategy() calls validate() unconditionally before promotion.
        return True, None


class _FakeEntryPoint:
    def __init__(self, name, factory):
        self.name = name
        self._factory = factory

    def load(self):
        return self._factory


# ===========================================================================
# 1. AdapterManifest
# ===========================================================================


def test_manifest_defaults():
    m = AdapterManifest(name="x")
    assert m.train_mode == TRAIN_MODE_ONE_TIME
    assert PluginCapability.ROUTING_STRATEGY in m.capabilities
    assert m.family == "x"  # defaults to name
    assert m.learns is False
    assert m.uses_artifact is False


def test_manifest_invalid_train_mode_falls_back():
    m = AdapterManifest(name="x", train_mode="garbage")
    assert m.train_mode == TRAIN_MODE_ONE_TIME


# ===========================================================================
# 2. ABI negotiation
# ===========================================================================


def test_abi_same_major_ok():
    assert _abi_compatible("1.0", "1.0") is True


def test_abi_major_newer_refused():
    assert _abi_compatible("2.0", "1.0") is False


def test_abi_minor_newer_accepted():
    # MINOR-newer is additive-only => accepted (use only known methods).
    assert _abi_compatible("1.5", "1.0") is True


def test_abi_older_accepted():
    assert _abi_compatible("0.9", "1.0") is True


# ===========================================================================
# 3. route = select_deployment alias
# ===========================================================================


def test_existing_strategy_gets_route_alias():
    strat = DefaultStrategy()
    assert not hasattr(strat, "route")
    attach_route_alias(strat)
    assert hasattr(strat, "route")
    assert strat.route == strat.select_deployment


def test_stub_satisfies_routing_adapter_protocol():
    adapter = _StubClassifierAdapter()
    assert isinstance(adapter, RoutingAdapter)


# ===========================================================================
# 4. Loader staging (monkeypatched entry_points)
# ===========================================================================


def test_loader_stages_compatible_adapter():
    plugin = AdapterLoaderPlugin()

    ep = _FakeEntryPoint("stub", lambda: _StubClassifierAdapter(name="loaded-stub"))
    assert plugin._stage_one(ep, None) is True

    registry = get_routing_registry()
    assert registry.get("loaded-stub") is not None


def test_loader_skips_abi_incompatible():
    class _Future:
        def declare_capabilities(self):
            return AdapterManifest(name="future-adapter", adapter_api_version="2.0")

        def route(self, ctx):
            return None

    plugin = AdapterLoaderPlugin()
    ep = _FakeEntryPoint("future", lambda: _Future())
    assert plugin._stage_one(ep, None) is False
    assert get_routing_registry().get("future-adapter") is None


def test_loader_skips_disallowed_capability():
    class _GuardrailAdapter:
        def declare_capabilities(self):
            return AdapterManifest(
                name="guardrail-adapter",
                capabilities={PluginCapability.GUARDRAIL},
            )

        def route(self, ctx):
            return None

    plugin = AdapterLoaderPlugin(
        allowed_capabilities={PluginCapability.ROUTING_STRATEGY}
    )
    ep = _FakeEntryPoint("g", lambda: _GuardrailAdapter())
    assert plugin._stage_one(ep, None) is False
    assert get_routing_registry().get("guardrail-adapter") is None


def test_loader_never_raises_on_bad_adapter():
    def _factory():
        raise RuntimeError("broken adapter")

    plugin = AdapterLoaderPlugin()
    ep = _FakeEntryPoint("broken", _factory)
    # must not raise
    assert plugin._stage_one(ep, None) is False


# ===========================================================================
# 5. SECOND-STRATEGY ATTACH (acceptance delta — not bandit-coupled)
# ===========================================================================


def test_mlops_fans_out_to_stub_classifier():
    coord = MLOpsCoordinator()
    stub = _StubClassifierAdapter()
    assert coord.register_learning_adapter("stub", stub) is True
    coord.on_aggregate_feedback({"bedrock/m": 0.8})
    # quality 0.8 -> score 0.6
    assert stub.update_calls == [("bedrock/m", 0.8 * 2.0 - 1.0)]


def test_mlops_fans_out_to_both_bandit_and_stub():
    from litellm_llmrouter.kumaraswamy_thompson import KumaraswamyThompsonStrategy

    coord = MLOpsCoordinator()
    bandit = KumaraswamyThompsonStrategy(seed=1)
    stub = _StubClassifierAdapter()
    coord.register_learning_adapter("bandit", bandit)
    coord.register_learning_adapter("stub", stub)

    coord.on_aggregate_feedback({"bedrock/m": 1.0})  # quality 1.0 -> score 1.0
    # stub recorded the call
    assert stub.update_calls == [("bedrock/m", 1.0)]
    # bandit posterior moved (success update)
    post = bandit._backend.get("default", "bedrock/m")
    assert post.alpha > 1.0


def test_mlops_skips_non_learning_adapter():
    coord = MLOpsCoordinator()

    class _NonLearner:
        def declare_capabilities(self):
            return AdapterManifest(name="nl", learns=False)

        def route(self, ctx):
            return None

    assert coord.register_learning_adapter("nl", _NonLearner()) is False


def test_one_time_adapter_does_not_subscribe():
    coord = MLOpsCoordinator()
    one_time = _StubClassifierAdapter(train_mode=TRAIN_MODE_ONE_TIME, name="ot")
    coord.register_learning_adapter("ot", one_time)
    coord.on_aggregate_feedback({"m": 0.9})
    # one_time strategies do NOT receive the feedback fan-out.
    assert one_time.update_calls == []


def test_continuous_adapter_subscribes():
    coord = MLOpsCoordinator()
    cont = _StubClassifierAdapter(train_mode=TRAIN_MODE_CONTINUOUS, name="ct")
    coord.register_learning_adapter("ct", cont)
    coord.on_aggregate_feedback({"m": 0.9})
    assert cont.update_calls == [("m", 0.9 * 2.0 - 1.0)]


# ===========================================================================
# 6. Artifact reload via the coordinator
# ===========================================================================


def test_apply_artifact_reloads_adapter():
    coord = MLOpsCoordinator()
    stub = _StubClassifierAdapter()
    coord.register_learning_adapter("stub", stub)
    ref = ArtifactRef(path="", payload={"state": {"posteriors": []}})
    # verify=False since no path/manifest signing here.
    assert coord.apply_artifact("stub", ref, verify=False) is True
    assert stub.reloaded == [ref]


def test_apply_artifact_unknown_adapter():
    coord = MLOpsCoordinator()
    assert coord.apply_artifact("nope", ArtifactRef(), verify=False) is False


# ===========================================================================
# 7. Singleton + RoutingFeedback dataclass
# ===========================================================================


def test_singleton_identity():
    assert get_mlops_coordinator() is get_mlops_coordinator()


def test_routing_feedback_fields():
    fb = RoutingFeedback(model="m", score=0.5, request_id="r", bucket="b")
    assert fb.model == "m"
    assert fb.score == 0.5
    assert fb.request_id == "r"
    assert fb.bucket == "b"
    assert fb.metadata == {}


# ===========================================================================
# 8. FEEDBACK arm: eval_pipeline -> coordinator -> learning adapters
# ===========================================================================


def test_discover_learning_adapters_from_registry():
    # STRATEGY-AGNOSTIC discovery: register a learning stub + a non-learning
    # DefaultStrategy in the registry; only the learner is picked up.
    registry = get_routing_registry()
    registry.register("learner", _StubClassifierAdapter(name="learner"))
    registry.register("plain", DefaultStrategy())

    coord = MLOpsCoordinator()
    discovered = coord.discover_learning_adapters_from_registry()
    assert "learner" in discovered
    assert "plain" not in discovered  # no declare_capabilities -> skipped


async def test_wire_feedback_loop_subscribes_into_eval_pipeline():
    # Build a real EvalPipeline (no judge calls — we drive push_feedback directly
    # by recording into the tracker), register a stub learner in the registry,
    # wire the loop, then push aggregate feedback and assert it reached the stub.
    from litellm_llmrouter.eval_pipeline import EvalPipeline

    registry = get_routing_registry()
    stub = _StubClassifierAdapter(name="reg-learner")
    registry.register("reg-learner", stub)

    coord = MLOpsCoordinator()
    pipeline = EvalPipeline(sample_rate=0.0)

    # force=True bypasses the settings flag for the unit test.
    wired = wire_mlops_feedback_loop(
        coordinator=coord, eval_pipeline=pipeline, force=True
    )
    assert wired is True
    # The stub learner was discovered + registered for fan-out.
    assert "reg-learner" in coord.list_adapters()

    # Seed an aggregate quality and trigger the FEEDBACK push.
    pipeline.tracker.record("bedrock/m", 0.9)
    await pipeline.push_feedback()

    # quality 0.9 -> score 0.8 reached the stub via the wired callback.
    assert stub.update_calls == [("bedrock/m", 0.9 * 2.0 - 1.0)]


def test_wire_feedback_loop_idempotent():
    from litellm_llmrouter.eval_pipeline import EvalPipeline

    coord = MLOpsCoordinator()
    pipeline = EvalPipeline(sample_rate=0.0)
    assert (
        wire_mlops_feedback_loop(coordinator=coord, eval_pipeline=pipeline, force=True)
        is True
    )
    # Second wiring is a no-op (callback already subscribed).
    assert (
        wire_mlops_feedback_loop(coordinator=coord, eval_pipeline=pipeline, force=True)
        is False
    )


def test_wire_feedback_loop_gated_by_flag(monkeypatch):
    from litellm_llmrouter.eval_pipeline import EvalPipeline
    from litellm_llmrouter.settings import reset_settings

    monkeypatch.setenv("ROUTEIQ_ADAPTER_FRAMEWORK__MLOPS_FEEDBACK_LOOP", "false")
    reset_settings()
    coord = MLOpsCoordinator()
    pipeline = EvalPipeline(sample_rate=0.0)
    # No force => flag off => wiring refused.
    assert wire_mlops_feedback_loop(coordinator=coord, eval_pipeline=pipeline) is False


def test_wire_feedback_loop_enabled_by_flag(monkeypatch):
    from litellm_llmrouter.eval_pipeline import EvalPipeline
    from litellm_llmrouter.settings import reset_settings

    monkeypatch.setenv("ROUTEIQ_ADAPTER_FRAMEWORK__MLOPS_FEEDBACK_LOOP", "true")
    reset_settings()
    coord = MLOpsCoordinator()
    pipeline = EvalPipeline(sample_rate=0.0)
    assert wire_mlops_feedback_loop(coordinator=coord, eval_pipeline=pipeline) is True


async def test_feedback_loop_reaches_bandit_posterior():
    # End-to-end FEEDBACK arm for the bandit specifically: registry -> discover
    # -> eval push -> bandit.update -> posterior moves. Proves the bandit is a
    # consumer of the SAME generalized loop a stub classifier rides.
    from litellm_llmrouter.eval_pipeline import EvalPipeline
    from litellm_llmrouter.kumaraswamy_thompson import (
        STRATEGY_NAME,
        KumaraswamyThompsonStrategy,
    )

    registry = get_routing_registry()
    bandit = KumaraswamyThompsonStrategy(seed=5)
    registry.register(STRATEGY_NAME, bandit, version="v1")

    coord = MLOpsCoordinator()
    pipeline = EvalPipeline(sample_rate=0.0)
    assert (
        wire_mlops_feedback_loop(coordinator=coord, eval_pipeline=pipeline, force=True)
        is True
    )
    assert STRATEGY_NAME in coord.list_adapters()

    pipeline.tracker.record("bedrock/claude", 1.0)  # quality 1.0 -> score 1.0
    await pipeline.push_feedback()

    # The bandit's posterior for the (default bucket, arm) moved on success.
    post = bandit._backend.get("default", "bedrock/claude")
    assert post.alpha > 1.0


def test_wire_feedback_loop_no_pipeline_still_registers_adapters():
    # When the eval pipeline is disabled (None), discovery still runs so adapters
    # are ready for direct online updates, but no callback is subscribed.
    registry = get_routing_registry()
    registry.register("only-learner", _StubClassifierAdapter(name="only-learner"))
    coord = MLOpsCoordinator()
    result = wire_mlops_feedback_loop(coordinator=coord, eval_pipeline=None, force=True)
    # get_eval_pipeline() returns None when disabled -> returns False, but the
    # adapter was still discovered.
    assert result is False
    assert "only-learner" in coord.list_adapters()
