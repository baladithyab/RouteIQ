# MLOps Training Setup for LLMRouter

This directory contains a complete MLOps setup for training, evaluating, and deploying LLMRouter routing models.

## Quick Start

```bash
# Start the MLOps stack
docker compose -f docker-compose.mlops.yml up -d

# Access services:
# - MLflow UI: http://localhost:5000
# - Jupyter Lab: http://localhost:8888 (token: llmrouter)
# - MinIO Console: http://localhost:9001 (admin/minioadmin)
```

## Components

| Service | Port | Description |
|---------|------|-------------|
| MLflow | 5000 | Experiment tracking & model registry |
| MinIO | 9000/9001 | S3-compatible object storage |
| Jupyter | 8888 | Interactive development environment |
| Trainer | - | GPU-enabled training container |
| Deployer | - | Model deployment to production |

## Training Workflow

### 1. Prepare Data

Place your training data in `data/`:

```bash
data/
├── train.json      # Training queries with labels
├── val.json        # Validation data
└── test.json       # Test data
```

### 2. Create Config

Create a config in `configs/`:

```yaml
# configs/knn_config.yaml
data_path:
  train_data: /app/data/train.json
  llm_data: /app/data/llm_candidates.json

hparam:
  k: 5
  embedding_model: all-MiniLM-L6-v2
```

### 3. Train Model

```bash
docker compose exec llmrouter-trainer python /app/scripts/train_router.py \
  --router-type knn \
  --config /app/configs/knn_config.yaml \
  --experiment-name my-experiment
```

### 4. Evaluate

View results in MLflow: http://localhost:5000

### 5. Deploy

```bash
docker compose exec model-deployer python /app/scripts/deploy_model.py \
  --model-name knn-router \
  --model-stage Production \
  --s3-bucket my-production-bucket
```

## Directory Structure

```
examples/mlops/
├── docker-compose.mlops.yml   # MLOps stack definition
├── Dockerfile.trainer         # Training environment
├── Dockerfile.deployer        # Deployment tools
├── scripts/
│   ├── train_router.py       # Training script
│   └── deploy_model.py       # Deployment script
├── configs/                   # Training configurations
├── data/                      # Training data
├── models/                    # Trained models
└── notebooks/                 # Jupyter notebooks
```

## Using with AWS S3

Set AWS credentials:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

Deploy to S3:

```bash
docker compose exec model-deployer python /app/scripts/deploy_model.py \
  --model-name knn-router \
  --s3-bucket your-production-bucket
```

## GPU Training

The trainer container supports NVIDIA GPUs. Ensure you have:
- NVIDIA drivers installed
- nvidia-container-toolkit installed

GPU resources are automatically allocated via Docker.

## Integration with Production Gateway

After deploying a model, update your gateway config:

```yaml
router_settings:
  routing_strategy: llmrouter-knn
  routing_strategy_args:
    model_s3_bucket: your-production-bucket
    model_s3_key: models/knn-router/
    hot_reload: true
```

The gateway will automatically load the new model.

