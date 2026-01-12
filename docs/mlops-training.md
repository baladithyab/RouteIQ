# MLOps Training Guide

This guide covers training, evaluating, and deploying custom LLMRouter routing models.

## Quick Start

```bash
cd examples/mlops
docker compose -f docker-compose.mlops.yml up -d
```

This starts:
- **MLflow** (port 5000) - Experiment tracking
- **MinIO** (port 9000/9001) - S3-compatible storage
- **Jupyter** (port 8888) - Interactive development
- **Trainer** - Training environment with GPU support

## Training a Router

### 1. Prepare Training Data

Create a dataset with query-model performance pairs:

```json
{
  "queries": [
    {"text": "Explain quantum computing", "best_model": "claude-3-opus"},
    {"text": "Write a hello world in Python", "best_model": "gpt-3.5-turbo"}
  ]
}
```

### 2. Configure Training

Create a YAML config:

```yaml
# configs/knn_config.yaml
data_path:
  train_data: /app/data/train.json
  llm_data: /app/data/llm_candidates.json

hparam:
  k: 5
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
```

### 3. Run Training

```bash
docker compose exec llmrouter-trainer python /app/scripts/train_router.py \
  --router-type knn \
  --config /app/configs/knn_config.yaml
```

### 4. View Results in MLflow

Open http://localhost:5000 to see:
- Training metrics
- Model artifacts
- Experiment comparisons

## Deploying Models

### Register in MLflow

```python
import mlflow

mlflow.register_model(
    f"runs:/{run_id}/model",
    "knn-router-production"
)
```

### Deploy to Production

```bash
docker compose exec model-deployer python /app/scripts/deploy_model.py \
  --model-name knn-router-production \
  --model-stage Production \
  --s3-bucket my-models-bucket
```

### Update Gateway Config

```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_s3_bucket: my-models-bucket
    model_s3_key: models/knn-router-production/
    hot_reload: true
```

## Training Different Router Types

### KNN Router
```bash
python train_router.py --router-type knn --config configs/knn.yaml
```

### MLP Router
```bash
python train_router.py --router-type mlp --config configs/mlp.yaml
```

### BERT Router
```bash
python train_router.py --router-type bert --config configs/bert.yaml
```

## Jupyter Notebooks

Access Jupyter at http://localhost:8888 (token: `llmrouter`)

Example notebooks:
- `01_data_exploration.ipynb` - Analyze routing data
- `02_train_knn_router.ipynb` - Train KNN router
- `03_evaluate_routers.ipynb` - Compare router performance

## CI/CD Integration

Add to your GitHub Actions:

```yaml
- name: Train Router
  run: |
    docker compose -f examples/mlops/docker-compose.mlops.yml run \
      llmrouter-trainer python /app/scripts/train_router.py \
      --router-type ${{ inputs.router_type }} \
      --config ${{ inputs.config }}

- name: Deploy Model
  if: github.ref == 'refs/heads/main'
  run: |
    docker compose run model-deployer python /app/scripts/deploy_model.py \
      --model-name ${{ inputs.model_name }} \
      --model-stage Production
```
