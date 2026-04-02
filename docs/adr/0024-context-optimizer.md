# ADR-0024: Context Optimization via Lossless Transform Pipeline

**Status**: Accepted
**Date**: 2026-04-02

## Context
LLM API costs are dominated by input tokens. Many requests contain redundant
content: repeated JSON schemas, duplicated system prompts, excessive whitespace,
long conversation histories. NadirClaw v0.13.0 demonstrated 30-70% token savings
via deterministic transforms.

## Decision
Implement context optimization as a gateway plugin with two modes:
- Safe (default): 5 lossless, deterministic transforms — zero quality loss
- Aggressive: Safe + semantic deduplication via sentence embeddings

Transform pipeline order (each preserves code blocks):
1. JSON minification — compact-serialize inline JSON
2. Tool schema dedup — replace repeated schemas with references
3. System prompt dedup — remove system text duplicated in user messages
4. Whitespace normalization — collapse excessive whitespace
5. Chat history trimming — keep system + first exchange + last N turns
6. (Aggressive) Semantic dedup — merge near-duplicate messages via difflib

## Implementation
- Plugin hooks into `on_llm_pre_call` (before LLM dispatch)
- Deep-copies messages to avoid mutating caller data
- Code blocks extracted via placeholder pattern before transforms, restored after
- JSON extraction uses brace-balancing (not regex) for nested structures
- Semantic dedup uses difflib.SequenceMatcher for word-level diff preservation
- Telemetry: tokens_saved, reduction_pct, transforms_applied per request

## Consequences
### Positive
- 30-70% token savings with zero quality loss (safe mode)
- Stacks with routing cost savings (eco profile + optimization)
- No additional dependencies for safe mode
- Per-request disable via metadata flag

### Negative
- Aggressive mode requires sentence-transformers (already loaded for centroid)
- JSON extraction has O(n) worst case for deeply nested content
- Deep copy adds ~0.5ms overhead per request
