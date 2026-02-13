# Multi-Account Bedrock Model Discovery

## Overview

RouteIQ delegates model discovery to the infrastructure layer. This keeps
the gateway focused on routing and avoids embedding AWS SDK logic that
varies across deployment patterns.

## Recommended Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  EventBridge │────▶│   Lambda     │────▶│  S3 Config   │
│  (scheduled) │     │  Discovery   │     │  Bucket      │
└──────────────┘     └──────┬───────┘     └──────┬───────┘
                            │                     │
                     ┌──────▼───────┐     ┌──────▼───────┐
                     │  STS Assume  │     │  RouteIQ     │
                     │  Role (hub)  │     │  Config Sync │
                     └──────────────┘     └──────────────┘
```

### How It Works

1. **EventBridge** triggers a Lambda function on a schedule (e.g., hourly)
2. **Lambda** assumes roles in spoke accounts via STS
3. **Lambda** calls `bedrock:ListFoundationModels` in each account/region
4. **Lambda** generates a RouteIQ-compatible `config.yaml` with discovered models
5. **Lambda** writes the config to S3
6. **RouteIQ** picks up the new config via its existing S3 config sync (`CONFIG_SYNC_ENABLED=true`)

### Lambda Discovery Function (Reference)

```python
import boto3
import yaml

def handler(event, context):
    accounts = event.get("accounts", [])
    models = []

    for account in accounts:
        sts = boto3.client("sts")
        creds = sts.assume_role(
            RoleArn=account["role_arn"],
            RoleSessionName="routeiq-discovery",
        )["Credentials"]

        for region in account["regions"]:
            bedrock = boto3.client(
                "bedrock",
                region_name=region,
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
            )

            response = bedrock.list_foundation_models()
            for model in response.get("modelSummaries", []):
                models.append({
                    "model_name": f"bedrock/{region}/{model['modelId']}",
                    "litellm_params": {
                        "model": f"bedrock/{model['modelId']}",
                        "aws_region_name": region,
                    },
                })

    # Write to S3 as RouteIQ config
    config = {"model_list": models}
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=event["config_bucket"],
        Key=event["config_key"],
        Body=yaml.dump(config),
    )

    return {"discovered": len(models)}
```

### CDK Snippet

```typescript
const discoveryFn = new lambda.Function(this, "BedrockDiscovery", {
  runtime: lambda.Runtime.PYTHON_3_12,
  handler: "discovery.handler",
  code: lambda.Code.fromAsset("lambda/discovery"),
  timeout: Duration.minutes(5),
  environment: {
    CONFIG_BUCKET: configBucket.bucketName,
    CONFIG_KEY: "config/config.yaml",
  },
});

// Schedule hourly
new events.Rule(this, "DiscoverySchedule", {
  schedule: events.Schedule.rate(Duration.hours(1)),
  targets: [new targets.LambdaFunction(discoveryFn)],
});

// Grant S3 write + STS assume role
configBucket.grantWrite(discoveryFn);
```

### Environment Variables (RouteIQ Side)

No new env vars needed. Use the existing config sync:

| Variable | Value | Purpose |
|----------|-------|---------|
| `CONFIG_S3_BUCKET` | your-config-bucket | S3 bucket for config |
| `CONFIG_S3_KEY` | config/config.yaml | S3 key for config |
| `CONFIG_SYNC_ENABLED` | `true` | Enable background sync |
| `CONFIG_HOT_RELOAD` | `true` | Enable hot reload on changes |

### Drift Detection

The Lambda can also compare discovered models against the current config
and emit CloudWatch metrics:

- `routeiq.discovery.models_available` — total models across all accounts
- `routeiq.discovery.models_configured` — models in current RouteIQ config
- `routeiq.discovery.models_unconfigured` — available but not configured

Alarm on `models_unconfigured > 0` to detect new models that need routing config.
