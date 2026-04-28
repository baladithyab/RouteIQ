# RouteIQ Product Vision

> **Last Updated:** 2026-03-29
> **Status:** Approved
> **Origin:** Office hours design session, formalized into repo guidance

---

## What RouteIQ Is

RouteIQ is **trust infrastructure for model switching**.

Not another API proxy. Not another gateway with a model dropdown. RouteIQ is the system that proves, with your own traffic, that intelligent routing works — and gives you the evidence to act on it.

Every other AI gateway asks you to trust a black box. OpenRouter says "auto" and picks a model. ClawRouters offers three presets. LiteLLM routes by rules you configure manually. None of them show you the proof.

RouteIQ shows you the proof.

## The Problem

Teams using LLMs default to a single model for everything. Claude Opus for classification tasks. GPT-4o for simple Q&A. This works. It's zero effort. And it overpays by 40-70% on tasks where cheaper models produce identical quality.

The cost isn't just price-per-token. It's **cost per successful task completion (CPTC)** — the total cost including retries, reasoning tokens, and tool calls to actually get the job done. A cheap model that needs 3 retries costs more than an expensive model that nails it first try. Nobody measures this. RouteIQ does.

## The Wedge: Evidence-Based Routing

RouteIQ's entry point is the **Routing Evidence Console** — a two-step product loop:

1. **See the waste.** Connect RouteIQ as a drop-in proxy. It auto-classifies your traffic by task type and shows a CPTC breakdown: "You're spending $4,218/month. 73% goes to Simple Q&A and Classification — tasks where cheaper models match quality."

2. **Prove the alternative.** Run transparent A/B experiments. RouteIQ routes a small percentage (1-10%) of traffic to a challenger model, uses LLM-as-a-judge to evaluate quality, and presents the evidence for async human review. You approve or reject based on data, not faith.

This is fundamentally different from black-box routing. The user sees the evidence. The user makes the decision. Trust is earned, not assumed.

## Who It's For

**Primary:** Platform and ML engineers at companies spending $2,000-50,000/month on LLM APIs, currently routing all traffic through a single model or a simple rule-based gateway.

**The moment that matters:** When someone sees the CPTC breakdown for the first time — "$1,890/month on Simple Q&A that Haiku handles at 96.8% quality match" — and realizes they've been overpaying for months without knowing it.

## How RouteIQ Is Different

### vs. OpenRouter
OpenRouter is a managed marketplace with a 5.5% markup. Its "auto" routing is a black box — you don't know why a model was chosen or whether it was the right choice. RouteIQ is self-hosted, BYOK (bring your own key), and shows you exactly why each routing decision was made.

### vs. LiteLLM
RouteIQ is built ON LiteLLM (100% API compatibility). LiteLLM provides the proxy layer — model management, key auth, spend tracking. RouteIQ adds the intelligence layer: 18 ML routing strategies from academic research (UIUC LLMRouter), CPTC measurement, evidence-based A/B testing, and the admin control plane that makes it all visible.

### vs. ClawRouters / Bifrost / Portkey
These gateways offer rule-based routing presets (cheapest/balanced/quality) or focus on performance/compliance. None have ML-based routing intelligence, none measure CPTC, and none provide evidence-based experiment workflows with human approval gates.

### The Actual Differentiator
18 ML routing strategies (KNN, MLP, SVM, ELO, matrix factorization, hybrid) trained on academic benchmarks. This isn't a lookup table. It's learned intelligence that improves with your traffic data. And it's transparent — you can see which strategy was used and why.

## The Flywheel

```
Production traffic
    → OTel traces with CPTC data
        → Feed ML routing strategies
            → Better routing decisions
                → More evidence of savings
                    → More trust from users
                        → Expanded routing scope
                            → More production traffic
```

The gateway gets smarter the more it's used. OTel data from production routing refines the ML strategies, which produce better routing decisions, which generate more evidence, which earns more trust, which means more traffic gets routed intelligently. This is a system that teaches itself.

## Product Principles

1. **Evidence over assertions.** Never tell the user "trust us, this is better." Show them the data. Side-by-side quality comparisons, CPTC deltas, judge confidence scores.

2. **Human-in-the-loop always.** No routing change goes live without explicit approval. The gateway suggests. The human decides.

3. **Zero-disruption experiments.** A/B tests route at most 10% of traffic to challengers. Small enough to be safe. Real enough to produce genuine evidence.

4. **Cost is cost-per-completion, not cost-per-token.** The honest metric. A model's sticker price means nothing if it needs 3 retries to get the answer right.

5. **Self-hosted first.** Enterprise and platform teams want to run the gateway in their own infra. BYOK means RouteIQ never touches billing. A managed version comes later, once the self-hosted product is validated.

6. **LiteLLM compatibility.** RouteIQ is a superset of LiteLLM. One URL change to switch. Every LiteLLM feature works. RouteIQ adds intelligence on top.

## Competitive Landscape (March 2026)

| Gateway | Type | Routing | BYOK | Evidence/Experiments | CPTC |
|---------|------|---------|------|---------------------|------|
| **RouteIQ** | Self-hosted OSS | 18 ML strategies | Yes | Yes (A/B + judge) | Yes |
| OpenRouter | Managed SaaS | Auto (black box) | Yes | No | No |
| LiteLLM | Self-hosted OSS | Rule-based | Yes | No | No |
| ClawRouters | Managed SaaS | 3 presets | Yes (free) | No | No |
| Bifrost | Self-hosted | None (perf focus) | Yes | No | No |
| Portkey | Managed SaaS | Rule-based | Yes | No | No |
| Helicone | Managed SaaS | None (obs focus) | Yes | No | No |

RouteIQ is the only gateway combining ML routing intelligence with evidence-based experiments and CPTC measurement.

## Long-Term Vision

Phase 1 is the Routing Evidence Console — prove the thesis with real users. Phase 2 expands to a full control plane: deep cost analysis, routing decision observatory (OTel traces), strategy manager, model registry. Phase 3, if demand validates it, is a managed service for teams that don't want to self-host.

The end state: **the AI gateway that configures itself** — observes your traffic, identifies optimization opportunities, runs experiments automatically, and presents evidence for human approval. The human stays in the loop, but the system does all the analysis.

---

*See also: [Evidence Console Design](architecture/evidence-console-design.md) | [Routing Strategies](routing-strategies.md) | [Technical Roadmap](../plans/technical-roadmap.md)*
