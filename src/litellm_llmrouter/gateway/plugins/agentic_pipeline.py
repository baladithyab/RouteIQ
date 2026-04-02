"""
Agentic Multi-Round Routing Pipeline Plugin
=============================================

For complex multi-part queries, this plugin implements the
decompose → route → execute → aggregate pattern inspired by LLMRouter's
KNNMultiRoundRouter, but as a gateway plugin wrapping the chat completion
lifecycle.

Pipeline stages:

1. **DETECT**: Heuristic complexity scorer evaluates the user message.
   Simple queries (below threshold) pass through unchanged.
2. **DECOMPOSE**: An orchestrator LLM breaks the query into 2–4
   independent sub-queries, each annotated with a suggested capability
   tier (fast / balanced / powerful).
3. **ROUTE**: Each sub-query is dispatched through LiteLLM's standard
   routing (which may use RouteIQ's ML strategies under the hood).
4. **EXECUTE**: Sub-queries are fanned out in parallel (or serially)
   via ``litellm.acompletion``.
5. **AGGREGATE**: The orchestrator LLM synthesises sub-responses into a
   single, coherent final answer.

This is **not** a routing strategy — it is a request lifecycle plugin that
intercepts ``on_llm_pre_call`` and replaces the original message payload
with the aggregated result so the downstream handler sends the correct
response back to the caller.

Configuration (environment variables):

- ``ROUTEIQ_AGENTIC_PIPELINE=false``          — master switch (disabled by default)
- ``ROUTEIQ_AGENTIC_ORCHESTRATOR_MODEL=gpt-4o-mini`` — model for decompose/aggregate
- ``ROUTEIQ_AGENTIC_COMPLEXITY_THRESHOLD=0.6``       — 0–1, queries above this get decomposed
- ``ROUTEIQ_AGENTIC_MAX_SUBQUERIES=4``               — maximum decomposition fan-out
- ``ROUTEIQ_AGENTIC_PARALLEL=true``                   — execute sub-queries in parallel
- ``ROUTEIQ_AGENTIC_TIMEOUT=30``                      — per sub-query timeout (seconds)

OTel span attributes emitted:

- ``agentic.pipeline.active``          — bool, whether the pipeline was triggered
- ``agentic.pipeline.complexity_score``— float, raw complexity score
- ``agentic.pipeline.sub_query_count`` — int, number of decomposed sub-queries
- ``agentic.pipeline.total_tokens``    — int, aggregate token usage
- ``agentic.pipeline.orchestrator_tokens`` — int, tokens consumed by orchestrator
- ``agentic.pipeline.latency_ms``      — float, total pipeline wall time
- ``agentic.pipeline.errors``          — int, number of failed sub-queries

Usage::

    export ROUTEIQ_AGENTIC_PIPELINE=true
    export ROUTEIQ_AGENTIC_ORCHESTRATOR_MODEL=gpt-4o-mini
    export LLMROUTER_PLUGINS=litellm_llmrouter.gateway.plugins.agentic_pipeline.AgenticPipelinePlugin
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from litellm_llmrouter.gateway.plugin_manager import (
    GatewayPlugin,
    PluginCapability,
    PluginContext,
    PluginMetadata,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger("litellm_llmrouter.plugins.agentic_pipeline")

# ---------------------------------------------------------------------------
# OTel — optional at runtime
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None  # type: ignore[assignment]

# OTel span attribute keys
ATTR_PIPELINE_ACTIVE = "agentic.pipeline.active"
ATTR_COMPLEXITY_SCORE = "agentic.pipeline.complexity_score"
ATTR_SUB_QUERY_COUNT = "agentic.pipeline.sub_query_count"
ATTR_TOTAL_TOKENS = "agentic.pipeline.total_tokens"
ATTR_ORCHESTRATOR_TOKENS = "agentic.pipeline.orchestrator_tokens"
ATTR_PIPELINE_LATENCY_MS = "agentic.pipeline.latency_ms"
ATTR_PIPELINE_ERRORS = "agentic.pipeline.errors"

# Internal metadata key injected into kwargs["litellm_params"]["metadata"]
_META_PIPELINE_RESULT = "_agentic_pipeline_result"


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class SubQuery:
    """A decomposed sub-query with routing assignment and execution result."""

    index: int
    query: str
    suggested_capability: str = "balanced"
    """Capability tier suggested by decomposer: fast / balanced / powerful."""

    assigned_model: str | None = None
    """Model actually selected by the router (populated after execution)."""

    response: str | None = None
    """Response text from the assigned model."""

    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class PipelineResult:
    """Aggregated result of a full pipeline execution."""

    original_query: str
    was_decomposed: bool = False
    complexity_score: float = 0.0
    sub_queries: list[SubQuery] = field(default_factory=list)
    aggregation_hint: str = ""
    final_response: str | None = None
    total_tokens: int = 0
    orchestrator_tokens: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0


# ============================================================================
# Stage 1 — Complexity Detection
# ============================================================================


class ComplexityDetector:
    """Heuristic complexity scorer for incoming user messages.

    Scoring factors (additive, capped at 1.0):

    - **Length factor** (0–0.3): word count / 500
    - **Question count** (0–0.2): number of ``?`` marks × 0.1
    - **Multi-step markers** (0–0.25): presence of sequencing language
    - **Complexity markers** (0–0.15): analytical/technical verbs
    - **Code presence** (0–0.1): fenced code or function definitions

    The detector is intentionally conservative — only truly complex,
    multi-part queries should clear the default threshold of 0.6.
    """

    MULTI_STEP_MARKERS: list[str] = [
        "first",
        "then",
        "after that",
        "finally",
        "next",
        "step 1",
        "step 2",
        "1.",
        "2.",
        "3.",
        "and also",
        "additionally",
        "furthermore",
        "moreover",
    ]

    COMPLEXITY_MARKERS: list[str] = [
        "analyze",
        "compare",
        "evaluate",
        "synthesize",
        "design",
        "implement",
        "architecture",
        "optimize",
        "debug",
        "refactor",
    ]

    def score(self, messages: list[dict[str, Any]]) -> float:
        """Score query complexity from 0.0 (trivial) to 1.0 (very complex).

        Extracts the last user message from the conversation and applies
        heuristic scoring.  Returns 0.0 if no user message is found.
        """
        user_msg = self._extract_last_user_text(messages)
        if not user_msg:
            return 0.0

        lower = user_msg.lower()
        score = 0.0

        # Length factor (0–0.3)
        word_count = len(user_msg.split())
        score += min(word_count / 500, 0.3)

        # Question count (0–0.2)
        question_marks = user_msg.count("?")
        score += min(question_marks * 0.1, 0.2)

        # Multi-step markers (0–0.25)
        marker_count = sum(1 for m in self.MULTI_STEP_MARKERS if m in lower)
        score += min(marker_count * 0.05, 0.25)

        # Complexity markers (0–0.15)
        complexity_count = sum(1 for m in self.COMPLEXITY_MARKERS if m in lower)
        score += min(complexity_count * 0.05, 0.15)

        # Code presence (0–0.1)
        if "```" in user_msg or "def " in user_msg or "function " in user_msg:
            score += 0.1

        return min(score, 1.0)

    @staticmethod
    def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
        """Extract the text content of the last user message."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Handle multi-part content (OpenAI vision format)
            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                return " ".join(parts)
        return ""


# ============================================================================
# Stage 2 — Query Decomposition
# ============================================================================


class QueryDecomposer:
    """Decomposes complex queries into sub-queries using an orchestrator LLM.

    The decomposer calls ``litellm.acompletion`` with a structured prompt
    requesting JSON output containing independent sub-queries and an
    aggregation hint.
    """

    _DECOMPOSE_PROMPT = """\
You are a query decomposition assistant. Your job is to break a complex \
query into 2-{max_sub} independent sub-queries that can be answered \
separately and then combined into a final answer.

Rules:
- Each sub-query must be self-contained (no references to other sub-queries).
- If the query is already simple enough, return it as a single sub-query.
- For each sub-query, suggest a capability tier:
  - "fast" — simple factual lookup or formatting
  - "balanced" — moderate reasoning or generation
  - "powerful" — complex analysis, multi-step reasoning, or creative tasks

Respond with valid JSON only (no markdown fences):
{{
  "sub_queries": [
    {{"query": "...", "capability": "fast|balanced|powerful"}},
    ...
  ],
  "aggregation_hint": "Brief instruction on how to combine the sub-responses"
}}

User query:
{query}"""

    def __init__(self, orchestrator_model: str, max_subqueries: int) -> None:
        self._model = orchestrator_model
        self._max = max_subqueries

    async def decompose(self, query: str) -> tuple[list[SubQuery], str, int]:
        """Decompose *query* into sub-queries.

        Returns:
            A 3-tuple of ``(sub_queries, aggregation_hint, tokens_used)``.

        Raises:
            Exception: Propagated from ``litellm.acompletion`` or JSON parsing.
        """
        import litellm

        prompt = self._DECOMPOSE_PROMPT.format(
            query=query[:4000],
            max_sub=self._max,
        )

        response = await litellm.acompletion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1000,
        )

        content: str = response.choices[0].message.content or "{}"
        tokens_used = _extract_total_tokens(response)

        data = json.loads(content)
        sub_queries: list[SubQuery] = []
        for i, sq in enumerate(data.get("sub_queries", [])[: self._max]):
            sub_queries.append(
                SubQuery(
                    index=i,
                    query=sq.get("query", "").strip(),
                    suggested_capability=sq.get("capability", "balanced"),
                )
            )

        # Fallback: if decomposer returned nothing useful, wrap the original
        if not sub_queries:
            sub_queries = [
                SubQuery(index=0, query=query, suggested_capability="balanced")
            ]

        hint = data.get(
            "aggregation_hint", "Combine the sub-responses into a comprehensive answer."
        )
        return sub_queries, hint, tokens_used


# ============================================================================
# Stage 3+4 — Routing + Execution
# ============================================================================


class SubQueryExecutor:
    """Executes individual sub-queries via ``litellm.acompletion``.

    Each sub-query inherits the original request's system message context
    and is dispatched through LiteLLM's normal routing path (which may use
    RouteIQ's ML strategies).
    """

    def __init__(
        self,
        *,
        parallel: bool = True,
        timeout: float = 30.0,
        system_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self._parallel = parallel
        self._timeout = timeout
        self._system_messages = system_messages or []

    async def execute_all(
        self, sub_queries: list[SubQuery], **base_kwargs: Any
    ) -> list[SubQuery]:
        """Execute all sub-queries (parallel or serial) and populate results.

        Returns the same ``SubQuery`` list with ``response``, ``assigned_model``,
        ``input_tokens``, ``output_tokens``, ``latency_ms``, and ``error``
        populated.
        """
        if self._parallel and len(sub_queries) > 1:
            tasks = [self._execute_one(sq, **base_kwargs) for sq in sub_queries]
            await asyncio.gather(*tasks, return_exceptions=False)
        else:
            for sq in sub_queries:
                await self._execute_one(sq, **base_kwargs)
        return sub_queries

    async def _execute_one(self, sq: SubQuery, **base_kwargs: Any) -> None:
        """Execute a single sub-query, catching all errors gracefully."""
        import litellm

        messages: list[dict[str, Any]] = [
            *self._system_messages,
            {"role": "user", "content": sq.query},
        ]

        # Allow model override from base kwargs; strip agentic-specific keys
        call_kwargs: dict[str, Any] = {
            k: v
            for k, v in base_kwargs.items()
            if k not in ("messages", "response_format", "stream")
        }
        # Do NOT force a specific model — let the router decide.
        # The caller's original model parameter is preserved in base_kwargs.

        start = time.monotonic()
        try:
            response = await asyncio.wait_for(
                litellm.acompletion(messages=messages, stream=False, **call_kwargs),
                timeout=self._timeout,
            )
            sq.latency_ms = (time.monotonic() - start) * 1000
            sq.response = response.choices[0].message.content or ""
            sq.assigned_model = getattr(response, "model", None) or call_kwargs.get(
                "model"
            )
            usage = getattr(response, "usage", None)
            if usage:
                sq.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                sq.output_tokens = getattr(usage, "completion_tokens", 0) or 0
        except asyncio.TimeoutError:
            sq.latency_ms = (time.monotonic() - start) * 1000
            sq.error = f"Sub-query {sq.index} timed out after {self._timeout}s"
            logger.warning(sq.error)
        except Exception as exc:
            sq.latency_ms = (time.monotonic() - start) * 1000
            sq.error = f"Sub-query {sq.index} failed: {exc!s}"
            logger.warning(sq.error, exc_info=True)


# ============================================================================
# Stage 5 — Response Aggregation
# ============================================================================


class ResponseAggregator:
    """Synthesises sub-responses into a final answer using the orchestrator LLM."""

    _AGGREGATE_PROMPT = """\
You are a response synthesis assistant. Combine the following sub-responses \
into a single, well-structured answer to the original question. Integrate \
information naturally — do not simply concatenate the sub-responses.

Original question:
{original_query}

Sub-responses:
{sub_responses}

Aggregation guidance: {hint}

Provide a clear, comprehensive response."""

    def __init__(self, orchestrator_model: str) -> None:
        self._model = orchestrator_model

    async def aggregate(
        self,
        original_query: str,
        sub_queries: list[SubQuery],
        hint: str,
    ) -> tuple[str, int]:
        """Aggregate sub-responses into a final answer.

        Returns:
            ``(final_response_text, tokens_used)``

        Raises:
            Exception: Propagated from ``litellm.acompletion``.
        """
        import litellm

        sub_text_parts: list[str] = []
        for sq in sub_queries:
            body = sq.response or sq.error or "(no response)"
            sub_text_parts.append(
                f"--- Sub-query {sq.index + 1} ---\nQ: {sq.query}\nA: {body}"
            )
        sub_responses = "\n\n".join(sub_text_parts)

        prompt = self._AGGREGATE_PROMPT.format(
            original_query=original_query[:2000],
            sub_responses=sub_responses[:8000],
            hint=hint[:500],
        )

        response = await litellm.acompletion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )

        content: str = response.choices[0].message.content or ""
        tokens_used = _extract_total_tokens(response)
        return content, tokens_used


# ============================================================================
# Plugin class
# ============================================================================


class AgenticPipelinePlugin(GatewayPlugin):
    """Gateway plugin implementing the agentic multi-round routing pipeline.

    Uses the ``on_llm_pre_call`` hook to intercept chat completion requests.
    When a query's complexity score exceeds the configured threshold the plugin
    runs the full decompose → execute → aggregate pipeline and replaces the
    original messages with the final aggregated response so that the caller
    receives a single, coherent answer.

    Simple queries pass through unchanged with zero overhead beyond the
    complexity scoring (sub-millisecond).
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="agentic_pipeline",
            version="1.0.0",
            description=(
                "Decomposes complex queries, routes sub-queries independently, "
                "aggregates responses"
            ),
            capabilities={PluginCapability.EVALUATOR},
            priority=30,  # Run before context optimizer (priority 50)
        )

    def __init__(self) -> None:
        self._enabled: bool = False
        self._threshold: float = 0.6
        self._parallel: bool = True
        self._timeout: float = 30.0
        self._orchestrator_model: str = "gpt-4o-mini"
        self._max_subqueries: int = 4

        # Components (initialised in startup)
        self._detector: ComplexityDetector | None = None
        self._decomposer: QueryDecomposer | None = None
        self._aggregator: ResponseAggregator | None = None

        # OTel metrics (created once, recorded at runtime)
        self._counter_pipeline_runs: Any = None
        self._counter_sub_queries: Any = None
        self._histogram_latency: Any = None
        self._counter_errors: Any = None
        self._counter_tokens: Any = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    async def startup(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        enabled = os.environ.get("ROUTEIQ_AGENTIC_PIPELINE", "false").lower() == "true"
        if not enabled:
            logger.info(
                "Agentic pipeline plugin disabled (ROUTEIQ_AGENTIC_PIPELINE!=true)"
            )
            return

        self._orchestrator_model = os.environ.get(
            "ROUTEIQ_AGENTIC_ORCHESTRATOR_MODEL", "gpt-4o-mini"
        )
        self._threshold = float(
            os.environ.get("ROUTEIQ_AGENTIC_COMPLEXITY_THRESHOLD", "0.6")
        )
        self._max_subqueries = int(
            os.environ.get("ROUTEIQ_AGENTIC_MAX_SUBQUERIES", "4")
        )
        self._parallel = (
            os.environ.get("ROUTEIQ_AGENTIC_PARALLEL", "true").lower() == "true"
        )
        self._timeout = float(os.environ.get("ROUTEIQ_AGENTIC_TIMEOUT", "30"))

        self._enabled = True
        self._detector = ComplexityDetector()
        self._decomposer = QueryDecomposer(
            self._orchestrator_model, self._max_subqueries
        )
        self._aggregator = ResponseAggregator(self._orchestrator_model)

        # Initialise OTel metrics if available
        self._init_metrics(context)

        logger.info(
            "Agentic pipeline enabled: "
            f"model={self._orchestrator_model}, "
            f"threshold={self._threshold}, "
            f"max_sub={self._max_subqueries}, "
            f"parallel={self._parallel}, "
            f"timeout={self._timeout}s"
        )

    async def shutdown(
        self, app: "FastAPI", context: PluginContext | None = None
    ) -> None:
        self._enabled = False
        logger.info("Agentic pipeline plugin shut down")

    async def health_check(self) -> dict[str, Any]:
        if not self._enabled:
            return {"status": "ok", "detail": "disabled"}
        return {
            "status": "ok",
            "orchestrator_model": self._orchestrator_model,
            "threshold": self._threshold,
        }

    # ------------------------------------------------------------------
    # LLM lifecycle hook — the core interception point
    # ------------------------------------------------------------------

    async def on_llm_pre_call(
        self,
        model: str,
        messages: list[Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Intercept chat completion requests and run the pipeline if needed.

        Returns ``None`` to pass through (simple queries) or a ``dict`` of
        kwargs overrides that replaces the original messages with the
        aggregated pipeline response.
        """
        if not self._enabled:
            return None

        # Only intercept chat completions (messages-based calls)
        if not messages:
            return None

        # Skip streaming requests — the pipeline returns a complete response
        # and cannot produce a stream of tokens for the aggregated result.
        if kwargs.get("stream"):
            return None

        # Skip if this is already an agentic sub-query (prevent recursion)
        metadata = _get_or_create_metadata(kwargs)
        if metadata.get("_agentic_sub_query"):
            return None

        # Stage 1: Complexity detection
        assert self._detector is not None
        complexity = self._detector.score(messages)
        if complexity < self._threshold:
            logger.debug(
                f"Complexity {complexity:.2f} below threshold {self._threshold} — pass through"
            )
            self._set_span_attributes(active=False, complexity=complexity)
            return None

        logger.info(
            f"Complexity {complexity:.2f} >= {self._threshold} — activating pipeline"
        )

        # Run full pipeline
        pipeline_start = time.monotonic()
        result = await self._run_pipeline(messages, kwargs, complexity)
        result.total_latency_ms = (time.monotonic() - pipeline_start) * 1000

        # Record telemetry
        self._record_telemetry(result)
        self._set_span_attributes(
            active=True,
            complexity=complexity,
            sub_count=len(result.sub_queries),
            total_tokens=result.total_tokens,
            orchestrator_tokens=result.orchestrator_tokens,
            latency_ms=result.total_latency_ms,
            errors=result.error_count,
        )

        if result.final_response is None:
            # Pipeline failed entirely — let the original request proceed
            logger.error(
                "Agentic pipeline failed entirely; falling back to passthrough"
            )
            return None

        # Replace the original messages with the aggregated response.
        # We rewrite messages to contain the aggregated answer as an assistant
        # turn preceded by the original user query, so the downstream handler
        # returns the aggregated content.
        overrides: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": result.original_query},
                {"role": "assistant", "content": result.final_response},
            ],
        }

        # Stash the full pipeline result in metadata for observability
        metadata[_META_PIPELINE_RESULT] = {
            "complexity_score": result.complexity_score,
            "sub_query_count": len(result.sub_queries),
            "total_tokens": result.total_tokens,
            "orchestrator_tokens": result.orchestrator_tokens,
            "latency_ms": round(result.total_latency_ms, 2),
            "errors": result.error_count,
        }

        return overrides

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    async def _run_pipeline(
        self,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
        complexity: float,
    ) -> PipelineResult:
        """Execute the full decompose → execute → aggregate pipeline."""
        user_query = ComplexityDetector._extract_last_user_text(messages)
        result = PipelineResult(
            original_query=user_query,
            was_decomposed=True,
            complexity_score=complexity,
        )

        # Extract system messages for sub-query context
        system_msgs = [m for m in messages if m.get("role") == "system"]

        # Stage 2: Decompose
        assert self._decomposer is not None
        try:
            sub_queries, hint, decompose_tokens = await self._decomposer.decompose(
                user_query,
            )
            result.sub_queries = sub_queries
            result.aggregation_hint = hint
            result.orchestrator_tokens += decompose_tokens
            logger.info(
                f"Decomposed into {len(sub_queries)} sub-queries "
                f"(decompose tokens: {decompose_tokens})"
            )
        except Exception as exc:
            logger.error(f"Decomposition failed: {exc}", exc_info=True)
            result.error_count += 1
            # Fallback: use original query as single sub-query
            result.sub_queries = [
                SubQuery(index=0, query=user_query, suggested_capability="balanced")
            ]
            result.aggregation_hint = "Return the response as-is."

        # Stage 3+4: Route & Execute
        # Build base kwargs for sub-queries, marking them to prevent recursion
        sub_kwargs = self._build_sub_query_kwargs(kwargs)
        executor = SubQueryExecutor(
            parallel=self._parallel,
            timeout=self._timeout,
            system_messages=system_msgs,
        )

        try:
            await executor.execute_all(result.sub_queries, **sub_kwargs)
        except Exception as exc:
            logger.error(f"Sub-query execution failed: {exc}", exc_info=True)

        # Count execution results
        for sq in result.sub_queries:
            result.total_tokens += sq.input_tokens + sq.output_tokens
            if sq.error:
                result.error_count += 1

        successful = [sq for sq in result.sub_queries if sq.response]
        logger.info(
            f"Executed {len(result.sub_queries)} sub-queries: "
            f"{len(successful)} succeeded, {result.error_count} failed"
        )

        # If all sub-queries failed, bail out
        if not successful:
            logger.error("All sub-queries failed — pipeline returning None")
            return result

        # If only one sub-query and it succeeded, skip aggregation
        if len(result.sub_queries) == 1 and successful:
            result.final_response = successful[0].response
            result.total_tokens += result.orchestrator_tokens
            return result

        # Stage 5: Aggregate
        assert self._aggregator is not None
        try:
            final_text, agg_tokens = await self._aggregator.aggregate(
                user_query,
                result.sub_queries,
                result.aggregation_hint,
            )
            result.final_response = final_text
            result.orchestrator_tokens += agg_tokens
            result.total_tokens += result.orchestrator_tokens
            logger.info(f"Aggregation complete (aggregate tokens: {agg_tokens})")
        except Exception as exc:
            logger.error(f"Aggregation failed: {exc}", exc_info=True)
            result.error_count += 1
            # Fallback: concatenate successful sub-responses
            result.final_response = "\n\n".join(
                f"**Part {sq.index + 1}**: {sq.response}" for sq in successful
            )
            result.total_tokens += result.orchestrator_tokens

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sub_query_kwargs(original_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build kwargs for sub-query calls, preventing recursion.

        Strips ``messages`` (sub-queries provide their own) and ``stream``
        (sub-queries are always non-streaming).  Injects a metadata marker
        so the plugin's ``on_llm_pre_call`` skips sub-queries.
        """
        # Shallow copy to avoid mutating the original
        kw: dict[str, Any] = {
            k: v
            for k, v in original_kwargs.items()
            if k not in ("messages", "stream", "response_format")
        }

        # Ensure litellm_params.metadata exists and mark as sub-query
        metadata = _get_or_create_metadata(kw)
        metadata["_agentic_sub_query"] = True

        return kw

    def _init_metrics(self, context: PluginContext | None) -> None:
        """Initialise OTel metric instruments via the metrics accessor."""
        if not OTEL_AVAILABLE:
            return

        try:
            if context and context.metrics:
                meter = context.metrics.get_meter("agentic_pipeline", "1.0.0")
            else:
                from opentelemetry.metrics import get_meter

                meter = get_meter("agentic_pipeline", "1.0.0")

            self._counter_pipeline_runs = meter.create_counter(
                name="agentic.pipeline.runs",
                unit="1",
                description="Number of agentic pipeline activations",
            )
            self._counter_sub_queries = meter.create_counter(
                name="agentic.pipeline.sub_queries",
                unit="1",
                description="Total sub-queries executed across all pipeline runs",
            )
            self._histogram_latency = meter.create_histogram(
                name="agentic.pipeline.latency",
                unit="ms",
                description="End-to-end pipeline latency in milliseconds",
            )
            self._counter_errors = meter.create_counter(
                name="agentic.pipeline.errors",
                unit="1",
                description="Number of sub-query or pipeline-level errors",
            )
            self._counter_tokens = meter.create_counter(
                name="agentic.pipeline.tokens",
                unit="1",
                description="Total tokens consumed by the pipeline (inc. orchestrator)",
            )
        except Exception as exc:
            logger.warning(f"Failed to initialise OTel metrics: {exc}")

    def _record_telemetry(self, result: PipelineResult) -> None:
        """Record OTel metrics for a completed pipeline run."""
        attrs = {"orchestrator_model": self._orchestrator_model}

        if self._counter_pipeline_runs:
            self._counter_pipeline_runs.add(1, attrs)
        if self._counter_sub_queries:
            self._counter_sub_queries.add(len(result.sub_queries), attrs)
        if self._histogram_latency:
            self._histogram_latency.record(result.total_latency_ms, attrs)
        if self._counter_errors and result.error_count > 0:
            self._counter_errors.add(result.error_count, attrs)
        if self._counter_tokens and result.total_tokens > 0:
            self._counter_tokens.add(result.total_tokens, attrs)

    @staticmethod
    def _set_span_attributes(
        *,
        active: bool,
        complexity: float,
        sub_count: int = 0,
        total_tokens: int = 0,
        orchestrator_tokens: int = 0,
        latency_ms: float = 0.0,
        errors: int = 0,
    ) -> None:
        """Set OTel span attributes on the current span (if available)."""
        if not OTEL_AVAILABLE or trace is None:
            return

        span = trace.get_current_span()
        if not span or not span.is_recording():
            return

        span.set_attribute(ATTR_PIPELINE_ACTIVE, active)
        span.set_attribute(ATTR_COMPLEXITY_SCORE, complexity)
        if active:
            span.set_attribute(ATTR_SUB_QUERY_COUNT, sub_count)
            span.set_attribute(ATTR_TOTAL_TOKENS, total_tokens)
            span.set_attribute(ATTR_ORCHESTRATOR_TOKENS, orchestrator_tokens)
            span.set_attribute(ATTR_PIPELINE_LATENCY_MS, latency_ms)
            span.set_attribute(ATTR_PIPELINE_ERRORS, errors)


# ============================================================================
# Module-level helpers
# ============================================================================


def _get_or_create_metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Ensure ``kwargs["litellm_params"]["metadata"]`` exists and return it."""
    litellm_params = kwargs.setdefault("litellm_params", {})
    metadata = litellm_params.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        litellm_params["metadata"] = metadata
    return metadata


def _extract_total_tokens(response: Any) -> int:
    """Extract total token count from a LiteLLM response object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    return getattr(usage, "total_tokens", 0) or 0
