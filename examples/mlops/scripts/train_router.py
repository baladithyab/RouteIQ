#!/usr/bin/env python3
"""
Train LLMRouter models with MLflow tracking.

Example usage:
    python train_router.py --router-type knn --config configs/knn_config.yaml
"""

import os
import sys
import click
import yaml
import mlflow
from pathlib import Path


@click.command()
@click.option(
    "--router-type",
    required=True,
    type=click.Choice(["knn", "svm", "mlp", "mf", "bert", "causallm", "hybrid"]),
    help="Type of router to train",
)
@click.option(
    "--config", required=True, type=click.Path(exists=True), help="YAML config file"
)
@click.option(
    "--experiment-name", default="llmrouter-training", help="MLflow experiment name"
)
@click.option(
    "--output-dir", default="/app/models", help="Output directory for trained model"
)
def train_router(router_type: str, config: str, experiment_name: str, output_dir: str):
    """Train an LLMRouter routing model."""

    print(f"🚀 Training {router_type} router")

    # Load config
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"   Config: {config}")

    # Set up MLflow
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    print(f"   MLflow URI: {mlflow_uri}")
    print(f"   Experiment: {experiment_name}")

    with mlflow.start_run(run_name=f"{router_type}-router"):
        # Log parameters
        mlflow.log_params(cfg.get("hparam", {}))
        mlflow.log_param("router_type", router_type)

        # Import appropriate trainer based on router type
        if router_type == "knn":
            from llmrouter.models.knn_router import KNNRouter

            router = KNNRouter(yaml_path=config)
        elif router_type == "svm":
            from llmrouter.models.svm_router import SVMRouter

            router = SVMRouter(yaml_path=config)
        elif router_type == "mlp":
            from llmrouter.models.mlp_router import MLPRouter

            router = MLPRouter(yaml_path=config)
        elif router_type == "mf":
            from llmrouter.models.mf_router import MFRouter

            router = MFRouter(yaml_path=config)
        elif router_type == "bert":
            from llmrouter.models.bert_router import BertRouter

            router = BertRouter(yaml_path=config)
        elif router_type == "causallm":
            from llmrouter.models.causallm_router import CausalLMRouter

            router = CausalLMRouter(yaml_path=config)
        elif router_type == "hybrid":
            from llmrouter.models.hybrid_router import HybridRouter

            router = HybridRouter(yaml_path=config)
        else:
            print(f"❌ Unknown router type: {router_type}")
            sys.exit(1)

        print("📊 Starting training...")

        # Train the router (implementation depends on router type)
        # Most routers use .fit() or train() methods
        if hasattr(router, "fit"):
            router.fit()
        elif hasattr(router, "train"):
            router.train()

        # Log metrics if available
        if hasattr(router, "metrics"):
            mlflow.log_metrics(router.metrics)

        # Save model
        output_path = Path(output_dir) / f"{router_type}_router"
        output_path.mkdir(parents=True, exist_ok=True)

        if hasattr(router, "save"):
            router.save(str(output_path / "model.pt"))

        # Save config alongside model
        with open(output_path / "config.yaml", "w") as f:
            yaml.dump(cfg, f)

        # Log artifacts to MLflow
        mlflow.log_artifacts(str(output_path))

        print("✅ Training complete!")
        print(f"   Model saved to: {output_path}")
        print(f"   MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    train_router()
