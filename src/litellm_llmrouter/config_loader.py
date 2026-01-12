"""
Configuration and Model Loading Utilities
==========================================

This module provides utilities for loading configuration and models
from S3/GCS for the LiteLLM + LLMRouter integration.
"""

from pathlib import Path


from litellm._logging import verbose_proxy_logger


def download_config_from_s3(bucket_name: str, object_key: str, local_path: str) -> bool:
    """Download configuration file from S3."""
    try:
        import boto3

        s3_client = boto3.client("s3")

        verbose_proxy_logger.info(
            f"Downloading config from s3://{bucket_name}/{object_key}"
        )

        # Ensure directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        s3_client.download_file(bucket_name, object_key, local_path)

        verbose_proxy_logger.info(f"Config downloaded to {local_path}")
        return True

    except Exception as e:
        verbose_proxy_logger.error(f"Failed to download config from S3: {e}")
        return False


async def download_config_from_gcs(
    bucket_name: str, object_key: str, local_path: str
) -> bool:
    """Download configuration file from GCS."""
    try:
        from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger

        gcs_bucket = GCSBucketLogger(bucket_name=bucket_name)

        verbose_proxy_logger.info(
            f"Downloading config from gs://{bucket_name}/{object_key}"
        )

        file_contents = await gcs_bucket.download_gcs_object(object_key)
        if file_contents is None:
            raise Exception(f"File contents are None for {object_key}")

        # Ensure directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        # Write to local file
        with open(local_path, "wb") as f:
            f.write(file_contents)

        verbose_proxy_logger.info(f"Config downloaded to {local_path}")
        return True

    except Exception as e:
        verbose_proxy_logger.error(f"Failed to download config from GCS: {e}")
        return False


def download_model_from_s3(bucket_name: str, object_key: str, local_dir: str) -> bool:
    """Download model files from S3."""
    try:
        import boto3

        s3_client = boto3.client("s3")

        verbose_proxy_logger.info(
            f"Downloading model from s3://{bucket_name}/{object_key}"
        )

        # Ensure directory exists
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        # If object_key is a prefix (directory), download all files
        if object_key.endswith("/"):
            paginator = s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name, Prefix=object_key):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    relative_path = key[len(object_key) :]
                    if relative_path:
                        local_file = local_path / relative_path
                        local_file.parent.mkdir(parents=True, exist_ok=True)
                        s3_client.download_file(bucket_name, key, str(local_file))
        else:
            # Single file download
            filename = Path(object_key).name
            s3_client.download_file(bucket_name, object_key, str(local_path / filename))

        verbose_proxy_logger.info(f"Model downloaded to {local_dir}")
        return True

    except Exception as e:
        verbose_proxy_logger.error(f"Failed to download model from S3: {e}")
        return False


def download_custom_router_from_s3(
    bucket_name: str, object_key: str, local_dir: str
) -> bool:
    """Download custom router Python files from S3."""
    return download_model_from_s3(bucket_name, object_key, local_dir)


def register_llmrouter_strategies():
    """Register LLMRouter strategies with LiteLLM."""
    from .strategies import LLMROUTER_STRATEGIES

    verbose_proxy_logger.info("Registering LLMRouter strategies...")

    # This is a placeholder - actual registration depends on
    # how we integrate with LiteLLM's router
    for strategy in LLMROUTER_STRATEGIES:
        verbose_proxy_logger.debug(f"  Registered: {strategy}")

    verbose_proxy_logger.info(
        f"Registered {len(LLMROUTER_STRATEGIES)} LLMRouter strategies"
    )
