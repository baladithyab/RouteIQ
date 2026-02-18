# TG7: Cloud-Native Deployment Epic

**Status:** Queued  
**Epic Owner:** DevOps / SRE  
**Last Updated:** 2026-02-04

---

## Goal

Enable production-grade Kubernetes deployments with proper resource management, autoscaling, and infrastructure-as-code patterns for RouteIQ Gateway.

### Non-Goals

- Supporting every cloud provider (focus on AWS EKS, extensible to others)
- Building a custom deployment tool (use Helm + kubectl)
- Implementing multi-cluster federation (single cluster focus)

---

## Sub-TG Breakdown

### TG7.1: Kubernetes Manifests

**Description:** Create production-ready Kubernetes manifests for core deployment.

**Acceptance Criteria:**
- [ ] `Deployment` manifest with resource limits, probes, and anti-affinity
- [ ] `Service` manifest with appropriate type (ClusterIP/LoadBalancer)
- [ ] `ConfigMap` for non-sensitive configuration
- [ ] `Secret` template for sensitive values
- [ ] `PodDisruptionBudget` for availability during updates
- [ ] Kustomize overlays for dev/staging/prod environments

**Key Files:**
- `deploy/k8s/base/` (new directory)
- `deploy/k8s/overlays/dev/`
- `deploy/k8s/overlays/prod/`

---

### TG7.2: Helm Chart Enhancements

**Description:** Enhance the Helm chart for production deployments.

**Acceptance Criteria:**
- [ ] HPA configuration with custom metrics support
- [ ] NetworkPolicy for pod-to-pod communication control
- [ ] ServiceAccount with minimal RBAC permissions
- [ ] Ingress configuration with TLS termination
- [ ] Values schema validation (values.schema.json)
- [ ] Helm chart tests (`helm test`)

**Key Files:**
- `deploy/charts/routeiq-gateway/`
- `deploy/charts/routeiq-gateway/values.schema.json` (new)
- `deploy/charts/routeiq-gateway/templates/tests/` (new)

---

### TG7.3: PostgreSQL HA Setup

**Description:** Configure production-grade PostgreSQL with high availability.

**Acceptance Criteria:**
- [ ] Documentation for RDS setup with Multi-AZ
- [ ] StatefulSet example for self-managed Postgres
- [ ] Connection pooling configuration (PgBouncer)
- [ ] Backup/restore procedures documented
- [ ] Smoke test validates DB failover handling

**Key Files:**
- `deploy/k8s/postgres/` (new)
- `docs/deployment.md`
- `docs/high-availability.md`

---

### TG7.4: Redis Cluster Setup

**Description:** Configure Redis Cluster for distributed caching and locking.

**Acceptance Criteria:**
- [ ] Documentation for ElastiCache setup
- [ ] StatefulSet example for self-managed Redis Cluster
- [ ] Sentinel configuration for failover
- [ ] Application config for Redis Cluster mode
- [ ] Smoke test validates Redis failover handling

**Key Files:**
- `deploy/k8s/redis/` (new)
- `config/config.yaml` (redis cluster config example)
- `docs/high-availability.md`

---

### TG7.5: Hot-Reload Config Sync

**Description:** Implement sidecar for syncing configuration from S3 without restarts.

**Acceptance Criteria:**
- [ ] Sidecar container watches S3 bucket for config changes
- [ ] Config changes trigger graceful reload via signal or API
- [ ] Validation of new config before applying
- [ ] Rollback on invalid config
- [ ] Documentation for S3 bucket setup and IAM permissions

**Key Files:**
- `docker/config-sync-sidecar/` (new)
- `deploy/charts/routeiq-gateway/templates/deployment.yaml`
- `docs/configuration.md`

---

## Branch + Squash Workflow

```bash
# Create feature branch
git checkout -b tg7-cloud-native-deploy

# Work on sub-TGs, commit locally as needed
git add .
git commit -m "feat(tg7.1): add production K8s manifests"
# ... more commits ...

# When TG7 is complete, squash merge to main
git checkout main
git merge --squash tg7-cloud-native-deploy
git commit -m "feat: complete TG7 cloud-native deployment epic"
```

---

## Test/Validation Commands

```bash
# Validate Kubernetes manifests
kubectl apply --dry-run=client -f deploy/k8s/base/

# Validate Helm chart
helm lint deploy/charts/routeiq-gateway/
helm template routeiq deploy/charts/routeiq-gateway/ --debug

# Helm chart tests (requires cluster)
helm install routeiq-test deploy/charts/routeiq-gateway/ --dry-run
helm test routeiq-test

# Kustomize validation
kubectl kustomize deploy/k8s/overlays/prod/ | kubectl apply --dry-run=client -f -

# Docker Compose HA smoke test
docker-compose -f docker-compose.ha.yml up -d
curl -s http://localhost:4000/health/readiness | jq .
docker-compose -f docker-compose.ha.yml down

# Lint and type check (for any Go/Python code)
./scripts/lint.sh check
```

---

## Bulk Publish Workflow

If local `git push` is blocked by Code Defender:

```bash
# Push via Road Runner
rr push

# MANDATORY: Sync local repo after rr push
git pull
```

> ⚠️ **Critical:** Always run `git pull` after `rr push` to sync your local repo with the remote state.

---

## Related Documents

- [Deployment Guide](../docs/deployment.md)
- [High Availability](../docs/high-availability.md)
- [Cloud-Native Architecture](../docs/architecture/cloud-native.md)
- [Helm Chart](../deploy/charts/routeiq-gateway/)
- [Backlog P0-01, P0-02, P0-03, P0-05](backlog.md)
