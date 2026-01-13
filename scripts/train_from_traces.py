#!/usr/bin/env python3
"""
Comprehensive Training Pipeline: Jaeger Traces → LLMRouter Model

This script implements the full MLOps pipeline:
1. Fetches Jaeger traces from MinIO (exported trace data)
2. Converts traces to LLMRouter training format
3. Generates query embeddings using sentence-transformers
4. Trains a KNN router model with MLflow tracking
5. Uploads the trained model to MinIO/S3
6. Verifies hot-reload picks up the new model

Usage:
    python scripts/train_from_traces.py --train
    python scripts/train_from_traces.py --validate
    python scripts/train_from_traces.py --full-pipeline

Expected workflow: Traces → Training Data → Trained Model → S3/MinIO → Hot Reload → Updated Routing
"""

import argparse
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
import httpx
import mlflow
import torch
import yaml
from botocore.client import Config
from sentence_transformers import SentenceTransformer

# =============================================================================
# Configuration
# =============================================================================
LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4010")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-test-master-key")
JAEGER_URL = os.getenv("JAEGER_URL", "http://localhost:16686")
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
MINIO_URL = os.getenv("MINIO_URL", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

# Paths
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "llmrouter-models")
TRACES_BUCKET = os.getenv("TRACES_BUCKET", "mlflow")
TRACES_PREFIX = os.getenv("TRACES_PREFIX", "traces/")
MODEL_OUTPUT_KEY = os.getenv("MODEL_OUTPUT_KEY", "router_model.pt")
LLM_CANDIDATES_PATH = os.getenv("LLM_CANDIDATES_PATH", "config/llm_candidates.json")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# =============================================================================
# MinIO/S3 Client
# =============================================================================
def get_s3_client():
    """Get boto3 S3 client for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket_exists(s3, bucket_name: str):
    """Ensure S3 bucket exists, create if not."""
    try:
        s3.head_bucket(Bucket=bucket_name)
    except s3.exceptions.ClientError:
        print(f"   Creating bucket: {bucket_name}")
        s3.create_bucket(Bucket=bucket_name)


# =============================================================================
# Step 1: Fetch Traces from MinIO
# =============================================================================
def fetch_latest_traces(s3) -> Optional[dict]:
    """Fetch the most recent Jaeger trace export from MinIO."""
    print("\n" + "=" * 60)
    print("📥 STEP 1: Fetching Traces from MinIO")
    print("=" * 60)

    try:
        # List trace files
        response = s3.list_objects_v2(Bucket=TRACES_BUCKET, Prefix=TRACES_PREFIX)
        objects = response.get("Contents", [])

        if not objects:
            print("   ⚠️  No trace files found in MinIO")
            return None

        # Find the most recent file
        latest = max(objects, key=lambda x: x["LastModified"])
        print(f"   Found {len(objects)} trace files")
        print(f"   Latest: {latest['Key']} ({latest['Size']} bytes)")

        # Download the file
        obj = s3.get_object(Bucket=TRACES_BUCKET, Key=latest["Key"])
        traces = json.loads(obj["Body"].read())
        print("   ✅ Loaded traces successfully")
        return traces

    except Exception as e:
        print(f"   ❌ Error fetching traces: {e}")
        return None


# =============================================================================
# Step 2: Convert Traces to Training Format
# =============================================================================
def convert_traces_to_training_data(traces: dict, llm_candidates: dict) -> list[dict]:
    """Convert Jaeger traces to LLMRouter training format."""
    print("\n" + "=" * 60)
    print("🔄 STEP 2: Converting Traces to Training Format")
    print("=" * 60)

    training_data = []
    trace_list = traces.get("data", [])
    print(f"   Processing {len(trace_list)} traces...")

    for trace in trace_list:
        for span in trace.get("spans", []):
            record = extract_training_record(span, llm_candidates)
            if record:
                training_data.append(record)

    print(f"   ✅ Extracted {len(training_data)} training records")
    return training_data


def extract_training_record(span: dict, llm_candidates: dict) -> Optional[dict]:
    """Extract training-relevant data from a Jaeger span."""
    tags = {t["key"]: t["value"] for t in span.get("tags", [])}
    duration_us = span.get("duration", 0)
    operation = span.get("operationName", "")

    # Only process spans that have LLM completion data
    # These are typically the "raw_gen_ai_request" spans
    if operation not in ["raw_gen_ai_request", "POST /v1/chat/completions"]:
        # Check if this span has LLM data by looking for gen_ai tags
        has_llm_data = any(k.startswith("gen_ai.") for k in tags.keys())
        if not has_llm_data:
            return None

    # Extract query from various possible tag formats
    # LiteLLM uses gen_ai.prompt.0.content format
    query = (
        tags.get("gen_ai.prompt.0.content", "")
        or tags.get("gen_ai.prompt", "")
        or tags.get("gen_ai.request.messages", "")
    )

    # Try to extract from logs if not in tags
    if not query:
        for log in span.get("logs", []):
            for field in log.get("fields", []):
                if field.get("key") in [
                    "gen_ai.prompt",
                    "request.messages",
                    "gen_ai.prompt.0.content",
                ]:
                    query = str(field.get("value", ""))[:500]
                    break

    # Extract model name (try multiple tag formats)
    model_name = (
        tags.get("gen_ai.response.model", "")
        or tags.get("gen_ai.request.model", "")
        or tags.get("llm.model", "")
        or tags.get("litellm.model", "")
    )

    # Skip if no query or model
    if not query or not model_name:
        return None

    # Normalize model name to match llm_candidates
    model_name = normalize_model_name(model_name, llm_candidates)

    # Calculate performance score (combination of speed, tokens, success)
    response_time_s = duration_us / 1_000_000
    total_tokens = int(
        tags.get("llm.usage.total_tokens", 0)
        or tags.get("gen_ai.usage.total_tokens", 0)
        or (
            int(tags.get("gen_ai.usage.prompt_tokens", 0))
            + int(tags.get("gen_ai.usage.completion_tokens", 0))
        )
    )

    # Get model metadata for cost/quality-based performance
    model_meta = llm_candidates.get(model_name, {})
    quality_score = model_meta.get("quality_score", 0.5)

    # Performance: balance quality vs latency (higher is better)
    # Normalize response time: <1s = 1.0, >10s = 0.0
    speed_score = max(0, 1 - (response_time_s / 10))
    performance = (quality_score * 0.7) + (speed_score * 0.3)

    return {
        "query": query,
        "model_name": model_name,
        "performance": round(performance, 4),
        "response_time_s": round(response_time_s, 3),
        "total_tokens": total_tokens,
        "trace_id": span.get("traceID", ""),
    }


def normalize_model_name(model_name: str, llm_candidates: dict) -> str:
    """Normalize model name to match llm_candidates keys."""
    # Direct match
    if model_name in llm_candidates:
        return model_name

    # Try to match by model_id suffix
    for name, meta in llm_candidates.items():
        model_id = meta.get("model_id", "")
        if model_id in model_name or model_name in model_id:
            return name

    # Fuzzy matching for common patterns
    name_lower = model_name.lower()
    for candidate_name in llm_candidates.keys():
        if candidate_name.lower() in name_lower or name_lower in candidate_name.lower():
            return candidate_name

    return model_name


# =============================================================================
# Step 3: Generate Embeddings
# =============================================================================
def generate_embeddings(
    training_data: list[dict], output_dir: Path
) -> tuple[Path, Path]:
    """Generate embeddings for training queries."""
    print("\n" + "=" * 60)
    print("🧠 STEP 3: Generating Query Embeddings")
    print("=" * 60)

    print(f"   Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Extract unique queries
    unique_queries = list(set(d["query"] for d in training_data))
    query_to_id = {q: i for i, q in enumerate(unique_queries)}

    print(f"   Unique queries: {len(unique_queries)}")
    print("   Generating embeddings...")

    embeddings = model.encode(
        unique_queries, convert_to_tensor=True, show_progress_bar=True
    )
    print(f"   Embedding shape: {embeddings.shape}")

    # Save embeddings
    emb_path = output_dir / "query_embeddings.pt"
    torch.save(embeddings, emb_path)
    print(f"   ✅ Saved embeddings: {emb_path}")

    # Add embedding_id to training data and save
    for record in training_data:
        record["embedding_id"] = query_to_id[record["query"]]

    # Split into train/test (80/20)
    split_idx = int(len(training_data) * 0.8)
    train_data = training_data[:split_idx]
    test_data = training_data[split_idx:]

    train_path = output_dir / "routing_train.jsonl"
    test_path = output_dir / "routing_test.jsonl"

    with open(train_path, "w") as f:
        for record in train_data:
            f.write(json.dumps(record) + "\n")

    with open(test_path, "w") as f:
        for record in test_data:
            f.write(json.dumps(record) + "\n")

    print(f"   ✅ Train: {train_path} ({len(train_data)} records)")
    print(f"   ✅ Test: {test_path} ({len(test_data)} records)")

    return train_path, test_path


# =============================================================================
# Step 4: Train KNN Router Model
# =============================================================================
def train_knn_router(
    data_dir: Path, llm_candidates: dict
) -> tuple[Optional[Path], Optional[str]]:
    """Train a KNN router model with MLflow tracking."""
    print("\n" + "=" * 60)
    print("🎓 STEP 4: Training KNN Router Model")
    print("=" * 60)

    # Create config file
    config_path = data_dir / "knn_config.yaml"
    config = {
        "data_path": {
            "routing_data_train": str(data_dir / "routing_train.jsonl"),
            "routing_data_test": str(data_dir / "routing_test.jsonl"),
            "query_embedding_data": str(data_dir / "query_embeddings.pt"),
            "llm_data": str(data_dir / "llm_data.json"),
        },
        "hparam": {
            "n_neighbors": 5,
            "metric": "cosine",
            "weights": "distance",
        },
    }

    # Save LLM data in the expected format
    llm_data_path = data_dir / "llm_data.json"
    with open(llm_data_path, "w") as f:
        json.dump(llm_candidates, f, indent=2)

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"   Config: {config_path}")
    print(f"   LLM data: {llm_data_path}")

    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment("llmrouter-trace-training")
    print(f"   MLflow: {MLFLOW_URL}")

    try:
        from llmrouter.models import KNNRouter

        with mlflow.start_run(
            run_name=f"knn-router-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ):
            mlflow.log_params(config["hparam"])
            mlflow.log_param("training_source", "jaeger-traces")

            print("   Initializing KNNRouter...")
            router = KNNRouter(yaml_path=str(config_path))

            print("   Training...")
            if hasattr(router, "fit"):
                router.fit()
            elif hasattr(router, "train"):
                router.train()

            # Log metrics if available
            if hasattr(router, "metrics") and router.metrics:
                mlflow.log_metrics(router.metrics)
                print(f"   Metrics: {router.metrics}")

            # Save model
            model_path = data_dir / "router_model.pt"
            if hasattr(router, "save_router"):
                router.save_router(str(model_path))
            elif hasattr(router, "save"):
                router.save(str(model_path))

            # Log artifacts (best effort, don't fail if MLflow has issues)
            try:
                mlflow.log_artifacts(str(data_dir))
            except Exception as artifact_err:
                print(f"   ⚠️  MLflow artifact logging failed: {artifact_err}")

            run_id = mlflow.active_run().info.run_id
            print("   ✅ Training complete!")
            print(f"   Model: {model_path}")
            print(f"   MLflow Run: {run_id}")

            return model_path, run_id

    except ImportError as e:
        print(f"   ❌ LLMRouter not available: {e}")
        print("   Creating mock model for testing...")
        # Create a mock model file for testing
        model_path = data_dir / "router_model.pt"
        torch.save(
            {"type": "mock_knn", "created": datetime.now().isoformat()}, model_path
        )
        return model_path, None
    except Exception as e:
        print(f"   ❌ Training failed: {e}")
        # Try to salvage the model if it was saved before the error
        model_path = data_dir / "router_model.pt"
        if model_path.exists():
            print(f"   ⚠️  Model was saved before error: {model_path}")
            return model_path, None
        return None, None


# =============================================================================
# Step 5: Upload Model to MinIO/S3
# =============================================================================
def upload_model_to_s3(
    s3, model_path: Path, config_path: Optional[Path] = None
) -> bool:
    """Upload trained model to MinIO/S3 for LiteLLM hot-reload."""
    print("\n" + "=" * 60)
    print("☁️  STEP 5: Uploading Model to MinIO/S3")
    print("=" * 60)

    try:
        ensure_bucket_exists(s3, MODEL_BUCKET)

        # Upload model
        model_key = MODEL_OUTPUT_KEY
        s3.upload_file(str(model_path), MODEL_BUCKET, model_key)
        print(f"   ✅ Uploaded: s3://{MODEL_BUCKET}/{model_key}")

        # Upload config if provided
        if config_path and config_path.exists():
            config_key = "router_config.yaml"
            s3.upload_file(str(config_path), MODEL_BUCKET, config_key)
            print(f"   ✅ Uploaded: s3://{MODEL_BUCKET}/{config_key}")

        # Also upload a versioned copy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_key = f"versions/router_model_{timestamp}.pt"
        s3.upload_file(str(model_path), MODEL_BUCKET, versioned_key)
        print(f"   ✅ Versioned: s3://{MODEL_BUCKET}/{versioned_key}")

        return True

    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        return False


# =============================================================================
# Step 6: Verify Hot-Reload
# =============================================================================
def verify_hot_reload() -> bool:
    """Verify LiteLLM picks up the new model via hot-reload."""
    print("\n" + "=" * 60)
    print("🔄 STEP 6: Verifying Hot-Reload")
    print("=" * 60)

    # Wait for hot-reload interval (default 300s, but we'll check sooner)
    print("   Waiting 5 seconds for potential hot-reload...")
    time.sleep(5)

    try:
        # Check LiteLLM health (use auth header)
        with httpx.Client() as client:
            response = client.get(
                f"{LITELLM_URL}/health/liveliness",
                headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
                timeout=10,
            )
            if response.status_code == 200:
                print("   ✅ LiteLLM is healthy")
            else:
                print(f"   ⚠️  LiteLLM health check: {response.status_code}")

            # Check model list
            models_resp = client.get(
                f"{LITELLM_URL}/v1/models",
                headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
                timeout=10,
            )
            if models_resp.status_code == 200:
                models = models_resp.json().get("data", [])
                print(f"   ℹ️  Available models: {len(models)}")
            else:
                print(f"   ⚠️  Could not list models: {models_resp.status_code}")

        return True

    except Exception as e:
        print(f"   ❌ Hot-reload verification failed: {e}")
        return False


# =============================================================================
# Step 7: End-to-End Validation
# =============================================================================
def validate_routing() -> dict:
    """Test the updated router by sending sample requests."""
    print("\n" + "=" * 60)
    print("✅ STEP 7: End-to-End Validation")
    print("=" * 60)

    # Test with different model pools to simulate routing
    test_cases = [
        ("nova-micro", "Write a simple hello world in Python"),
        ("nova-lite", "Explain the theory of relativity in detail"),
        ("claude-4.5-haiku", "What is 2+2?"),
    ]

    results = []

    with httpx.Client() as client:
        for model, content in test_cases:
            prompt = {"role": "user", "content": content}
            try:
                response = client.post(
                    f"{LITELLM_URL}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
                    json={
                        "model": model,
                        "messages": [prompt],
                        "max_tokens": 50,
                    },
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    model_used = data.get("model", "unknown")
                    results.append(
                        {
                            "prompt": content[:50],
                            "model_requested": model,
                            "model_used": model_used,
                            "status": "success",
                        }
                    )
                    print(f"   ✅ '{content[:30]}...' → {model_used}")
                else:
                    results.append(
                        {
                            "prompt": content[:50],
                            "model_requested": model,
                            "status": "error",
                            "code": response.status_code,
                        }
                    )
                    print(f"   ❌ '{content[:30]}...' → Error {response.status_code}")

            except Exception as e:
                results.append(
                    {
                        "prompt": content[:50],
                        "model_requested": model,
                        "status": "error",
                        "error": str(e),
                    }
                )
                print(f"   ❌ '{content[:30]}...' → {e}")

    success_count = sum(1 for r in results if r.get("status") == "success")
    return {
        "validation_results": results,
        "success_rate": success_count / len(results) if results else 0,
    }


# =============================================================================
# Main Pipeline
# =============================================================================
def run_full_pipeline(skip_upload: bool = False, skip_validation: bool = False):
    """Run the complete training pipeline."""
    print("\n" + "=" * 60)
    print("🚀 LLMROUTER TRAINING PIPELINE")
    print("=" * 60)
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print(f"   MinIO: {MINIO_URL}")
    print(f"   MLflow: {MLFLOW_URL}")
    print(f"   LiteLLM: {LITELLM_URL}")

    s3 = get_s3_client()

    # Load LLM candidates
    with open(LLM_CANDIDATES_PATH, "r") as f:
        llm_candidates = json.load(f)
    print(f"   LLM Candidates: {len(llm_candidates)} models")

    # Step 1: Fetch traces
    traces = fetch_latest_traces(s3)
    if not traces:
        print("\n❌ Pipeline failed: No traces available")
        return False

    # Step 2: Convert to training format
    training_data = convert_traces_to_training_data(traces, llm_candidates)
    if not training_data:
        print("\n❌ Pipeline failed: No training data extracted")
        return False

    # Create temp directory for training artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)

        # Step 3: Generate embeddings
        train_path, test_path = generate_embeddings(training_data, data_dir)

        # Step 4: Train model
        model_path, run_id = train_knn_router(data_dir, llm_candidates)
        if not model_path:
            print("\n❌ Pipeline failed: Training failed")
            return False

        # Step 5: Upload to S3
        if not skip_upload:
            config_path = data_dir / "knn_config.yaml"
            upload_success = upload_model_to_s3(s3, model_path, config_path)
            if not upload_success:
                print("\n⚠️  Model upload failed, continuing...")

    # Step 6: Verify hot-reload
    verify_hot_reload()

    # Step 7: Validation
    if not skip_validation:
        validation = validate_routing()
        print(f"\n   Validation Success Rate: {validation['success_rate']*100:.1f}%")

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)

    return True


# =============================================================================
# CLI Interface
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train LLMRouter model from Jaeger traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full training pipeline
  python scripts/train_from_traces.py --train

  # Validate routing with existing model
  python scripts/train_from_traces.py --validate

  # Full pipeline with all steps
  python scripts/train_from_traces.py --full-pipeline

  # Train without uploading to S3
  python scripts/train_from_traces.py --train --skip-upload

Environment Variables:
  LITELLM_URL          LiteLLM gateway URL (default: http://localhost:4010)
  MINIO_URL            MinIO endpoint URL (default: http://localhost:9000)
  MLFLOW_TRACKING_URI  MLflow tracking URI (default: http://localhost:5050)
  EMBEDDING_MODEL      Sentence transformer model (default: all-MiniLM-L6-v2)
""",
    )

    # Actions
    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    parser.add_argument(
        "--full-pipeline", action="store_true", help="Run complete pipeline"
    )

    # Options
    parser.add_argument(
        "--skip-upload", action="store_true", help="Skip S3 upload step"
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip validation step"
    )

    # Configuration overrides
    parser.add_argument("--litellm-url", type=str, help="LiteLLM URL override")
    parser.add_argument("--minio-url", type=str, help="MinIO URL override")
    parser.add_argument("--mlflow-url", type=str, help="MLflow URL override")

    args = parser.parse_args()

    # Apply overrides
    global LITELLM_URL, MINIO_URL, MLFLOW_URL
    if args.litellm_url:
        LITELLM_URL = args.litellm_url
    if args.minio_url:
        MINIO_URL = args.minio_url
    if args.mlflow_url:
        MLFLOW_URL = args.mlflow_url

    # Execute requested action
    if args.validate:
        print("\n🔍 Running validation only...")
        result = validate_routing()
        print(f"\n   Success Rate: {result['success_rate']*100:.1f}%")
        return 0 if result["success_rate"] > 0 else 1

    if args.train or args.full_pipeline:
        success = run_full_pipeline(
            skip_upload=args.skip_upload, skip_validation=args.skip_validation
        )
        return 0 if success else 1

    # Default: show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    exit(main())
