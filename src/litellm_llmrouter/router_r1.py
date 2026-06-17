"""Router-R1: Iterative Reasoning Router for RouteIQ.

Implements the Router-R1 concept (NeurIPS 2025) natively using RouteIQ's
own LLM proxy as both the reasoning engine and routing pool. No vLLM
dependency — the router LLM is called through LiteLLM like any other model.

Algorithm:
  1. A configured "router model" receives the user's query
  2. The router enters an iterative loop (up to max_iterations):
     a. THINK: Generate reasoning inside <think>...</think> tags
     b. ROUTE: Generate <route model="...">sub_query</route> to query a model
     c. RESULT: The routed model's response is injected as <result>...</result>
     d. DECIDE: Either continue reasoning or output <answer>final</answer>
  3. The cost-aware aspect: the router's prompt includes model pricing data,
     naturally preferring cheaper models when they suffice

Key differences from upstream Router-R1:
  - No vLLM: uses LiteLLM acompletion (works with any provider)
  - No RL training: uses prompt engineering with model metadata instead
  - Routing pool = RouteIQ's model_list (not an external API)
  - Integrated with centroid routing for fast sub-query routing

Configuration:
  ROUTEIQ_ROUTER_R1_ENABLED=false
  ROUTEIQ_ROUTER_R1_MODEL=gpt-4o-mini (the reasoning/router model)
  ROUTEIQ_ROUTER_R1_MAX_ITERATIONS=3
  ROUTEIQ_ROUTER_R1_TIMEOUT=30 (seconds per iteration)
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "R1Result",
    "RouterR1",
    "RoutingStep",
    "get_router_r1",
    "reset_router_r1",
]

logger = logging.getLogger("litellm_llmrouter.router_r1")


@dataclass
class RoutingStep:
    """A single step in the R1 reasoning loop."""

    iteration: int
    think: str = ""
    routed_model: Optional[str] = None
    routed_query: Optional[str] = None
    result: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: int = 0


@dataclass
class R1Result:
    """Result of a Router-R1 execution."""

    answer: str
    steps: List[RoutingStep] = field(default_factory=list)
    total_iterations: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    router_model: str = ""
    models_used: List[str] = field(default_factory=list)
    #: Why the iterative loop stopped (RouteIQ-81bc): ``answer`` (produced a final
    #: answer), ``max_iterations``, ``token_budget`` / ``latency_budget`` (a
    #: cost/latency gate tripped), ``size_limit``, ``timeout``, or ``error``.
    stop_reason: str = ""


SYSTEM_PROMPT = """You are an AI routing assistant. Your job is to answer the user's question by routing sub-queries to the most appropriate AI models.

Available models (with pricing per 1M tokens):
{model_info}

Instructions:
1. Think about which model(s) would best handle this query
2. Route sub-queries using: <route model="model_name">specific question for that model</route>
3. You'll receive results in <result>...</result> tags
4. When you have enough information, provide your final answer in: <answer>your complete answer</answer>

Guidelines:
- Prefer cheaper models for simple factual queries
- Use expensive models only for complex reasoning or creative tasks
- You can route multiple sub-queries across iterations
- Keep total iterations under {max_iterations}
- Always end with an <answer> tag"""


class RouterR1:
    """Iterative reasoning router using LLM-based routing decisions."""

    def __init__(
        self,
        router_model: str = "gpt-4o-mini",
        max_iterations: int = 3,
        timeout_per_iteration: float = 30.0,
        max_total_tokens: int = 0,
        max_total_latency_ms: float = 0.0,
    ):
        self._router_model = router_model
        self._max_iterations = max_iterations
        self._timeout = timeout_per_iteration
        # Cost/latency gates (RouteIQ-81bc). 0 disables a gate. When the
        # cumulative tokens (cost proxy) or wall-clock latency would exceed the
        # cap, the iterative loop stops BEFORE issuing another round — an
        # iterative reasoning router trades cost/latency for quality, so an
        # operator caps that tradeoff. Default 0/0 preserves prior behavior.
        self._max_total_tokens = max_total_tokens
        self._max_total_latency_ms = max_total_latency_ms
        self._route_pattern = re.compile(
            r'<route\s+model="([^"]+)">(.*?)</route>', re.DOTALL
        )
        self._answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        self._think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def _budget_tripped(self, result: "R1Result", start: float) -> Optional[str]:
        """Return the name of a tripped cost/latency gate, or None (RouteIQ-81bc).

        Checked at the TOP of each iteration so the loop never issues a round
        that would blow the budget. ``max_total_tokens`` is the cost proxy (a
        token cap); ``max_total_latency_ms`` is the wall-clock latency cap. 0
        disables the respective gate.
        """
        if self._max_total_tokens and result.total_tokens >= self._max_total_tokens:
            return "token_budget"
        if self._max_total_latency_ms:
            elapsed_ms = (time.monotonic() - start) * 1000
            if elapsed_ms >= self._max_total_latency_ms:
                return "latency_budget"
        return None

    def _build_model_info(self, deployments: List[Dict]) -> str:
        """Build model info string for the system prompt."""
        from litellm_llmrouter.centroid_routing import (
            MODEL_CAPABILITIES,
            MODEL_COSTS,
        )

        lines = []
        seen: set[str] = set()
        for dep in deployments:
            name = dep.get("model_name", "")
            if name in seen:
                continue
            seen.add(name)
            costs = MODEL_COSTS.get(name, {})
            caps = MODEL_CAPABILITIES.get(name, set())
            cost_str = f"${costs.get('input', '?')}/{costs.get('output', '?')}"
            cap_str = ", ".join(sorted(caps)) if caps else "text"
            lines.append(f"- {name}: {cost_str} input/output, capabilities: {cap_str}")
        return "\n".join(lines[:20])  # Limit to top 20 models

    async def route(
        self,
        query: str,
        deployments: List[Dict],
        system_message: Optional[str] = None,
    ) -> R1Result:
        """Execute the R1 iterative reasoning loop.

        Args:
            query: The user's query
            deployments: Available model deployments from the router
            system_message: Optional system message from the original request

        Returns:
            R1Result with the final answer and execution trace
        """
        import litellm

        model_info = self._build_model_info(deployments)
        system = SYSTEM_PROMPT.format(
            model_info=model_info,
            max_iterations=self._max_iterations,
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        if system_message:
            messages.append(
                {
                    "role": "user",
                    "content": f"[Original system context]: {system_message}",
                }
            )
        messages.append({"role": "user", "content": query})

        result = R1Result(answer="", router_model=self._router_model)
        start = time.monotonic()

        for iteration in range(self._max_iterations):
            step = RoutingStep(iteration=iteration)
            iter_start = time.monotonic()

            # Cost/latency gate (RouteIQ-81bc): stop BEFORE another round when a
            # budget cap is tripped. An iterative router trades cost/latency for
            # quality; the operator caps that tradeoff.
            tripped = self._budget_tripped(result, start)
            if tripped is not None:
                result.stop_reason = tripped
                logger.info(
                    "R1 stopping at iteration %d: %s gate tripped "
                    "(tokens=%d, elapsed_ms=%.0f)",
                    iteration,
                    tripped,
                    result.total_tokens,
                    (time.monotonic() - start) * 1000,
                )
                break

            # Safety check: stop if messages have grown too large
            total_chars = sum(len(str(m.get("content", ""))) for m in messages)
            if total_chars > 100_000:  # ~25K tokens, safety limit
                logger.warning("R1 messages exceed 100K chars, stopping iterations")
                result.stop_reason = "size_limit"
                break

            try:
                # Get router's reasoning
                response = await asyncio.wait_for(
                    litellm.acompletion(
                        model=self._router_model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=2000,
                    ),
                    timeout=self._timeout,
                )

                content = response.choices[0].message.content or ""
                step.tokens_used = getattr(response.usage, "total_tokens", 0)
                result.total_tokens += step.tokens_used

                # Extract thinking
                think_match = self._think_pattern.search(content)
                if think_match:
                    step.think = think_match.group(1).strip()

                # Check for final answer
                answer_match = self._answer_pattern.search(content)
                if answer_match:
                    result.answer = answer_match.group(1).strip()
                    step.latency_ms = (time.monotonic() - iter_start) * 1000
                    result.steps.append(step)
                    result.stop_reason = "answer"
                    break

                # Check for routing request
                route_match = self._route_pattern.search(content)
                if route_match:
                    target_model = route_match.group(1).strip()
                    sub_query = route_match.group(2).strip()
                    step.routed_model = target_model
                    step.routed_query = sub_query

                    # Execute the routed sub-query
                    try:
                        sub_response = await asyncio.wait_for(
                            litellm.acompletion(
                                model=target_model,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": sub_query,
                                    }
                                ],
                                temperature=0.3,
                                max_tokens=1500,
                            ),
                            timeout=self._timeout,
                        )
                        sub_content = sub_response.choices[0].message.content or ""
                        step.result = sub_content
                        result.models_used.append(target_model)
                        sub_tokens = getattr(sub_response.usage, "total_tokens", 0)
                        step.tokens_used += sub_tokens
                        result.total_tokens += sub_tokens
                    except Exception as e:
                        step.result = f"Error from {target_model}: {str(e)}"
                        logger.warning(
                            "R1 sub-query to %s failed: %s",
                            target_model,
                            e,
                        )

                    # Inject result back into conversation
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f'<result model="{target_model}">{step.result}</result>'
                            ),
                        }
                    )
                else:
                    # No route and no answer — treat as partial reasoning
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Please continue your reasoning and either "
                                "route to a model or provide your final "
                                "answer in <answer> tags."
                            ),
                        }
                    )

                step.latency_ms = (time.monotonic() - iter_start) * 1000
                result.steps.append(step)

            except asyncio.TimeoutError:
                step.latency_ms = self._timeout * 1000
                result.steps.append(step)
                logger.warning("R1 iteration %d timed out", iteration)
                result.stop_reason = "timeout"
                break
            except Exception as e:
                logger.error("R1 iteration %d error: %s", iteration, e)
                step.latency_ms = (time.monotonic() - iter_start) * 1000
                result.steps.append(step)
                result.stop_reason = "error"
                break

        # If no answer was produced, use the last content as answer
        if not result.answer and result.steps:
            last_step = result.steps[-1]
            if last_step.result:
                result.answer = last_step.result
            elif last_step.think:
                result.answer = last_step.think

        # Reached the iteration cap without any earlier stop signal.
        if not result.stop_reason:
            result.stop_reason = "max_iterations"

        result.total_iterations = len(result.steps)
        result.total_latency_ms = (time.monotonic() - start) * 1000

        # Eval-loop feedback (RouteIQ-81bc): hand the completed run to the eval
        # pipeline so the iterative router's answer is graded and its observed
        # cost/latency feeds the COLLECT/EVALUATE/AGGREGATE/FEEDBACK loop. A
        # best-effort no-op when the eval pipeline is disabled.
        self._emit_eval_sample(query, result)

        return result

    def _emit_eval_sample(self, query: str, result: "R1Result") -> None:
        """Emit an :class:`EvalSample` for a completed R1 run (RouteIQ-81bc).

        Records the router's final answer plus the run's observed cost
        (``total_tokens``) and latency so the eval feedback loop can grade the
        iterative router and feed quality back into routing. Best-effort: a
        disabled/absent eval pipeline, or any error, is a silent no-op so a
        feedback hiccup never breaks routing.
        """
        try:
            from litellm_llmrouter.eval_pipeline import get_eval_pipeline

            pipeline = get_eval_pipeline()
            if pipeline is None or not pipeline.should_sample():
                return
            from litellm_llmrouter.eval_pipeline import EvalSample

            sample = EvalSample(
                sample_id=f"r1-{int(time.monotonic() * 1000)}",
                timestamp=time.time(),
                model=self._router_model,
                strategy="router-r1",
                tier="router-r1",
                messages=[{"role": "user", "content": query}],
                response_content=result.answer,
                response_tokens=result.total_tokens,
                latency_ms=result.total_latency_ms,
            )
            pipeline.collect(sample)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("R1 eval-sample emit skipped: %s", exc)


# Singleton
_router: Optional[RouterR1] = None


def get_router_r1() -> Optional[RouterR1]:
    """Get the Router-R1 instance if enabled."""
    global _router
    if _router is not None:
        return _router
    import os

    # ROUTEIQ_ROUTER_R1_* env vars don't match pydantic-settings paths,
    # so check env vars first with settings as defaults.
    try:
        from litellm_llmrouter.settings import get_settings

        r1_cfg = get_settings().router_r1
    except Exception:
        r1_cfg = None

    env_enabled = os.environ.get("ROUTEIQ_ROUTER_R1_ENABLED")
    if env_enabled is not None:
        enabled = env_enabled.lower() == "true"
    elif r1_cfg is not None:
        enabled = r1_cfg.enabled
    else:
        enabled = False

    if not enabled:
        return None

    _router = RouterR1(
        router_model=os.environ.get(
            "ROUTEIQ_ROUTER_R1_MODEL",
            r1_cfg.model if r1_cfg else "gpt-4o-mini",
        ),
        max_iterations=int(
            os.environ.get(
                "ROUTEIQ_ROUTER_R1_MAX_ITERATIONS",
                str(r1_cfg.max_iterations if r1_cfg else 3),
            )
        ),
        timeout_per_iteration=float(
            os.environ.get(
                "ROUTEIQ_ROUTER_R1_TIMEOUT",
                str(r1_cfg.timeout if r1_cfg else 30),
            )
        ),
        # Cost/latency gates (RouteIQ-81bc). 0 == gate disabled (default).
        max_total_tokens=int(os.environ.get("ROUTEIQ_ROUTER_R1_MAX_TOTAL_TOKENS", "0")),
        max_total_latency_ms=float(
            os.environ.get("ROUTEIQ_ROUTER_R1_MAX_TOTAL_LATENCY_MS", "0")
        ),
    )
    return _router


def reset_router_r1() -> None:
    """Reset the Router-R1 singleton. Must be called in test fixtures."""
    global _router
    _router = None
