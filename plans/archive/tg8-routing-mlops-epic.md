# TG8: Routing & MLOps Epic

**Status:** Queued  
**Epic Owner:** ML Engineer / Backend  
**Last Updated:** 2026-02-04

---

## Goal

Complete the MLOps loop for intelligent routing with automated training pipelines, model deployment, and rollback mechanisms for continuous improvement.

### Non-Goals

- Building a custom ML framework (use scikit-learn, PyTorch as needed)
- Implementing real-time training (batch training focus)
- Creating a model registry service (use S3/MLflow integration)

---

## Sub-TG Breakdown

### TG8.1: Automated Training Pipeline

**Description:** End-to-end pipeline: Trace Export → Train → Deploy.

**Acceptance Criteria:**
- [ ] Jaeger trace export script produces training-ready dataset
- [ ] Training script supports multiple strategies (KNN, MLP, SVM)
- [ ] MLflow integration for experiment tracking
- [ ] Model artifacts stored in S3 with versioning
- [ ] Pipeline can be triggered via API or scheduled job
- [ ] Documentation for pipeline configuration and monitoring

**Key Files:**
- `examples/mlops/scripts/train_router.py`
- `examples/mlops/scripts/extract_jaeger_traces.py`
- `examples/mlops/docker-compose.mlops.yml`
- `docs/mlops-training.md`

---

### TG8.2: Model Deployment with Hot-Reload

**Description:** Deploy trained models without gateway restart.

**Acceptance Criteria:**
- [ ] Model deployment script updates S3 and triggers reload
- [ ] Gateway detects new model artifact and loads it
- [ ] Graceful transition from old to new model (no dropped requests)
- [ ] Health endpoint reports active model version
- [ ] Integration test validates hot-reload during traffic

**Key Files:**
- `examples/mlops/scripts/deploy_model.py`
- `src/litellm_llmrouter/hot_reload.py`
- `src/litellm_llmrouter/strategies.py`
- `tests/integration/test_model_hot_reload.py` (new)

---

### TG8.3: Model Rollback Mechanism

**Description:** One-click revert to previous model version.

**Acceptance Criteria:**
- [ ] Admin API endpoint for model rollback (`POST /admin/model/rollback`)
- [ ] Rollback targets specific version or "previous"
- [ ] Rollback logged in audit trail
- [ ] Validation that rollback model is healthy before activation
- [ ] Documentation for emergency rollback procedures

**Key Files:**
- `src/litellm_llmrouter/routes.py`
- `src/litellm_llmrouter/model_manager.py` (new or extended)
- `tests/integration/test_model_rollback.py` (new)
- `docs/mlops-training.md`

---

### TG8.4: Model Sync with Distributed Locking

**Description:** Prevent thundering herd during model downloads across replicas.

**Acceptance Criteria:**
- [ ] Redis-based distributed lock for model artifact download
- [ ] Only one replica downloads, others wait for lock release
- [ ] Lock timeout and retry logic for robustness
- [ ] Metrics for lock contention and wait times
- [ ] Unit tests validate locking behavior

**Key Files:**
- `src/litellm_llmrouter/model_sync.py` (new)
- `src/litellm_llmrouter/strategies.py`
- `tests/unit/test_model_sync.py` (new)

---

### TG8.5: Routing Strategy Registry

**Description:** Dynamic registration and selection of routing strategies.

**Acceptance Criteria:**
- [ ] Strategy registry supports runtime registration
- [ ] Strategies selectable via config or API
- [ ] Strategy metadata (name, version, capabilities) queryable
- [ ] A/B testing support (traffic split between strategies)
- [ ] Documentation for custom strategy development

**Key Files:**
- `src/litellm_llmrouter/strategy_registry.py`
- `src/litellm_llmrouter/strategies.py`
- `docs/routing-strategies.md`

---

## Branch + Squash Workflow

```bash
# Create feature branch
git checkout -b tg8-routing-mlops

# Work on sub-TGs, commit locally as needed
git add .
git commit -m "feat(tg8.1): implement automated training pipeline"
# ... more commits ...

# When TG8 is complete, squash merge to main
git checkout main
git merge --squash tg8-routing-mlops
git commit -m "feat: complete TG8 routing & MLOps epic"
```

---

## Test/Validation Commands

```bash
# Unit tests for routing and strategies
uv run pytest tests/unit/test_strategies.py tests/unit/test_strategy_registry.py -v

# Integration tests for MLOps pipeline
uv run pytest tests/integration/test_model_hot_reload.py -v

# MLOps pipeline smoke test
docker-compose -f examples/mlops/docker-compose.mlops.yml up -d
uv run python examples/mlops/scripts/generate_synthetic_traces.py
uv run python examples/mlops/scripts/train_router.py --config examples/mlops/configs/knn_config.yaml
uv run python examples/mlops/scripts/deploy_model.py
docker-compose -f examples/mlops/docker-compose.mlops.yml down

# Lint and type check
./scripts/lint.sh check
uv run mypy src/litellm_llmrouter/strategies.py src/litellm_llmrouter/strategy_registry.py

# Validate routing decision traces
uv run pytest tests/integration/ -k "routing" -v
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

- [MLOps Training Guide](../docs/mlops-training.md)
- [Routing Strategies](../docs/routing-strategies.md)
- [MLOps Loop Architecture](../docs/architecture/mlops-loop.md)
- [MLOps Examples](../examples/mlops/)
- [Backlog P1-01, P2-01, P2-02](backlog.md)
