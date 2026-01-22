#!/usr/bin/env python3
"""
Train LLMRouter models with MLflow tracking.

Example usage:
    python train_router.py --router-type knn --config configs/knn_config.yaml

    # Or via MLflow Projects:
    mlflow run . --entry-point train -P router_type=knn --env-manager local
"""

import os
import click
import yaml
import mlflow
from pathlib import Path


def _create_router_and_trainer(router_type: str, config: str):
    """Create the appropriate router and trainer based on type."""
    # Note: llmrouter uses lowercase module names (e.g., knnrouter, not knn_router)
    if router_type == "knn":
        from llmrouter.models.knnrouter import KNNRouter, KNNRouterTrainer

        router = KNNRouter(yaml_path=config)
        trainer = KNNRouterTrainer(router=router)
        return router, trainer
    elif router_type == "svm":
        from llmrouter.models.svmrouter import SVMRouter, SVMRouterTrainer

        router = SVMRouter(yaml_path=config)
        trainer = SVMRouterTrainer(router=router)
        return router, trainer
    elif router_type == "mlp":
        from llmrouter.models.mlprouter import MLPRouter, MLPRouterTrainer

        router = MLPRouter(yaml_path=config)
        trainer = MLPRouterTrainer(router=router)
        return router, trainer
    elif router_type == "mf":
        from llmrouter.models.mfrouter import MFRouter, MFRouterTrainer

        router = MFRouter(yaml_path=config)
        trainer = MFRouterTrainer(router=router)
        return router, trainer
    elif router_type == "bert":
        from llmrouter.models.routerdc import RouterDC, RouterDCTrainer

        router = RouterDC(yaml_path=config)
        trainer = RouterDCTrainer(router=router)
        return router, trainer
    elif router_type == "causallm":
        from llmrouter.models.causallm_router import CausalLMRouter, CausalLMTrainer

        router = CausalLMRouter(yaml_path=config)
        trainer = CausalLMTrainer(router=router)
        return router, trainer
    elif router_type == "hybrid":
        from llmrouter.models.hybrid_llm import HybridLLMRouter

        router = HybridLLMRouter(yaml_path=config)
        return router, None  # HybridLLM may not have a trainer
    else:
        raise ValueError(f"Unknown router type: {router_type}")


def _run_training(router_type: str, config: str, cfg: dict, output_dir: str):
    """Run the actual training logic."""
    print("📊 Creating router and trainer...")
    router, trainer = _create_router_and_trainer(router_type, config)

    print("📊 Starting training...")
    # Train using the trainer if available
    if trainer is not None:
        trainer.train()
    elif hasattr(router, "fit"):
        router.fit()
    elif hasattr(router, "train"):
        router.train()

    # Log metrics if available
    if trainer is not None and hasattr(trainer, "metrics"):
        mlflow.log_metrics(trainer.metrics)
    elif hasattr(router, "metrics"):
        mlflow.log_metrics(router.metrics)

    # Model is saved by the trainer to the path specified in config (model_path.save_model_path)
    # Get the saved model path from config
    model_path_cfg = cfg.get("model_path", {})
    saved_model_path = model_path_cfg.get("save_model_path")

    output_path = Path(output_dir) / f"{router_type}_router"
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy trained model to output_dir for deployment
    if saved_model_path and Path(saved_model_path).exists():
        import shutil

        dest_model = output_path / Path(saved_model_path).name
        shutil.copy2(saved_model_path, dest_model)
        print(f"   Model copied to: {dest_model}")
        mlflow.log_param("model_output_path", str(dest_model))

    # Save config alongside model
    with open(output_path / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Log artifacts to MLflow (skip if permission issues)
    try:
        mlflow.log_artifacts(str(output_path))
        print("   Artifacts logged to MLflow")
    except PermissionError as e:
        print(f"   ⚠️ Could not log artifacts to MLflow (permission issue): {e}")
        print(f"   Artifacts are available locally at: {output_path}")

    print("✅ Training complete!")
    print(f"   Config saved to: {output_path}")
    if saved_model_path:
        print(f"   Trained model: {saved_model_path}")


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
    "--output-dir", default="/tmp/models", help="Output directory for trained model"
)
def train_router(router_type: str, config: str, experiment_name: str, output_dir: str):
    """Train an LLMRouter routing model."""

    print(f"🚀 Training {router_type} router")

    # Load config
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"   Config: {config}")

    # Set up MLflow
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"   MLflow URI: {mlflow_uri}")

    # Check if we're running inside 'mlflow run' (MLFLOW_RUN_ID is set)
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if run_id:
        # We're inside an mlflow run - log directly without starting a new run
        print(f"   Using existing MLflow run: {run_id}")
        mlflow.log_params(cfg.get("hparam", {}))
        mlflow.log_param("router_type", router_type)
        _run_training(router_type, config, cfg, output_dir)
        return

    # Not in mlflow run, start our own
    print(f"   Experiment: {experiment_name}")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{router_type}-router"):
        mlflow.log_params(cfg.get("hparam", {}))
        mlflow.log_param("router_type", router_type)
        _run_training(router_type, config, cfg, output_dir)
        print(f"   MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    train_router()
