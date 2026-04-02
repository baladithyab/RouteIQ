# ADR-0011: Pluggable External Services (Bring-Your-Own DB/Redis/OTel)

**Status**: Proposed
**Date**: 2026-04-02
**Decision Makers**: RouteIQ Core Team

## Context

### Problem: All-or-Nothing Infrastructure

RouteIQ's Docker Compose files deploy PostgreSQL, Redis, and OTel Collector
as part of the stack. This is convenient for evaluation but problematic for
production deployments where organizations already have:

- **Managed PostgreSQL** (RDS, Cloud SQL, Azure Database)
- **Managed Redis** (ElastiCache, Memorystore, Azure Cache)
- **Existing OTel Collectors** (Datadog Agent, New Relic, Splunk)

Currently, switching from bundled services to external services requires:
1. Editing Docker Compose to remove bundled services
2. Setting multiple environment variables manually
3. Hoping the startup order works correctly
4. No clear feedback on which services are connected vs missing

### User Stories

- "I want to point RouteIQ at my existing RDS PostgreSQL instance"
- "I want to use my team's existing Redis cluster"
- "I want to send traces to my Datadog Agent, not deploy a new collector"
- "I want to evaluate RouteIQ without any external dependencies"

## Decision

Make all external services optional and configurable, with two deployment
modes: "batteries-included" and "plug-in".

### Batteries-Included Mode (Default)

`docker compose up` deploys everything needed to run RouteIQ with full
functionality, including PostgreSQL, Redis, and OTel Collector. This is
the default for evaluation and development.

### Plug-In Mode

Provide connection strings to existing services via environment variables
or Helm values. No bundled services are deployed for the configured
externals.

| Service | Env Vars | Helm Values |
|---------|----------|-------------|
| PostgreSQL | `DATABASE_URL` | `externalPostgresql.host`, `.port`, `.database`, `.existingSecret` |
| Redis | `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD` | `externalRedis.host`, `.port`, `.existingSecret` |
| OTel Collector | `OTEL_EXPORTER_OTLP_ENDPOINT` | `externalOtel.endpoint`, `.protocol` |

### Startup Probing

At startup, RouteIQ probes each configured service and reports status:

```
[INFO] Service probe results:
  PostgreSQL: connected (DATABASE_URL=postgresql://rds.example.com:5432/routeiq)
  Redis: connected (REDIS_HOST=elasticache.example.com:6379)
  OTel Collector: connected (OTEL_EXPORTER_OTLP_ENDPOINT=http://datadog-agent:4317)
```

When optional services are unavailable, the gateway degrades gracefully:

- **No PostgreSQL**: In-memory registries only (no persistence, no audit log DB)
- **No Redis**: No HA sync, no distributed caching, no quota enforcement
- **No OTel Collector**: Traces/metrics logged locally, not exported

### Helm Chart Integration

The Helm chart uses conditional sub-charts:

```yaml
# values.yaml
postgresql:
  enabled: true  # Set to false when using external
externalPostgresql:
  host: ""
  port: 5432
  database: "routeiq"
  existingSecret: ""  # K8s secret with 'password' key

redis:
  enabled: true
externalRedis:
  host: ""
  port: 6379
  existingSecret: ""

otelCollector:
  enabled: true
externalOtel:
  endpoint: ""
  protocol: "grpc"  # or "http"
```

## Consequences

### Positive

- **Production-ready from day one**: Organizations use their existing managed
  services without deploying duplicate infrastructure.
- **Cost reduction**: No redundant databases or caches running alongside
  existing managed services.
- **Operational simplicity**: Teams manage one PostgreSQL, one Redis, one
  OTel pipeline instead of RouteIQ-specific instances.
- **Graceful degradation**: Missing services don't crash the gateway.

### Negative

- **Configuration complexity**: More environment variables and Helm values
  to document and validate.
- **Testing matrix**: Must test with bundled and external service combinations.
- **SSL/auth variations**: External services may require different SSL modes,
  IAM auth, or mTLS that bundled services don't need.

## Alternatives Considered

### Alternative A: Bundled Services Only

- **Pros**: Simplest deployment; everything works out of the box.
- **Cons**: Wasteful in production; can't leverage existing infrastructure.
- **Rejected**: Not viable for enterprise deployments.

### Alternative B: External Services Only

- **Pros**: Forces proper infrastructure setup; no bundled services to maintain.
- **Cons**: Terrible evaluation experience; new users must set up PostgreSQL
  and Redis before trying RouteIQ.
- **Rejected**: Barrier to entry too high.

## References

- `docker-compose.yml` — Batteries-included deployment
- `deploy/charts/` — Helm chart with conditional sub-charts
- `src/litellm_llmrouter/env_validation.py` — Startup service probing
- [ADR-0004: asyncpg Connection Pooling](0004-asyncpg-connection-pooling.md)
- [ADR-0005: Redis Singleton](0005-redis-singleton.md)
