#!/usr/bin/env python3
"""
Deploy trained LLMRouter models to production.

This script fetches the latest model from MLflow model registry and
uploads it to S3 for the production gateway to use.
"""

import os
import sys
import boto3
import mlflow
from pathlib import Path
import click
import yaml


@click.command()
@click.option('--model-name', required=True, help='MLflow model name')
@click.option('--model-stage', default='Production', help='Model stage (Production, Staging)')
@click.option('--s3-bucket', envvar='TARGET_S3_BUCKET', required=True, help='Target S3 bucket')
@click.option('--s3-prefix', default='models/', help='S3 prefix for models')
@click.option('--local-path', default='/app/models', help='Local path to save model')
def deploy_model(model_name: str, model_stage: str, s3_bucket: str, s3_prefix: str, local_path: str):
    """Deploy a model from MLflow registry to S3."""
    
    print(f"🚀 Deploying model: {model_name} ({model_stage})")
    
    # Set up MLflow
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    print(f"   MLflow URI: {mlflow_uri}")
    
    # Download model from registry
    model_uri = f"models:/{model_name}/{model_stage}"
    local_model_path = Path(local_path) / model_name
    
    print(f"📥 Downloading model from: {model_uri}")
    try:
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(local_model_path))
        print(f"   Downloaded to: {local_model_path}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        sys.exit(1)
    
    # Upload to S3
    print(f"📤 Uploading to S3: s3://{s3_bucket}/{s3_prefix}{model_name}/")
    try:
        s3_client = boto3.client('s3')
        
        for file_path in local_model_path.rglob('*'):
            if file_path.is_file():
                s3_key = f"{s3_prefix}{model_name}/{file_path.relative_to(local_model_path)}"
                s3_client.upload_file(str(file_path), s3_bucket, s3_key)
                print(f"   Uploaded: {s3_key}")
        
        print(f"✅ Model deployed successfully!")
        print(f"   S3 URI: s3://{s3_bucket}/{s3_prefix}{model_name}/")
        
    except Exception as e:
        print(f"❌ Failed to upload to S3: {e}")
        sys.exit(1)
    
    # Generate config snippet for LiteLLM
    config_snippet = {
        'router_settings': {
            'routing_strategy': 'llmrouter-custom',
            'routing_strategy_args': {
                'model_s3_bucket': s3_bucket,
                'model_s3_key': f"{s3_prefix}{model_name}/",
                'hot_reload': True
            }
        }
    }
    
    print("\n📝 LiteLLM Configuration Snippet:")
    print(yaml.dump(config_snippet, default_flow_style=False))


if __name__ == '__main__':
    deploy_model()

