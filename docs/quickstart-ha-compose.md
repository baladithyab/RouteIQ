# Quickstart: High Availability (HA) Setup

This guide demonstrates a production-ready High Availability (HA) setup for RouteIQ Gateway using Docker Compose.

## Architecture

This setup includes:
- **Load Balancer**: Nginx distributing traffic.
- **Gateway Replicas**: Two instances (`litellm-gateway-1`, `litellm-gateway-2`) for redundancy.
- **Shared State**:
    - **PostgreSQL**: Persistent storage for models and configuration.
    - **Redis**: Distributed caching and pub/sub for sync.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/).
- `openssl`.

## 1. Configure Secrets

Start by copying the example environment file:

```bash
cp .env.example .env
```

Set the required secrets in your `.env` file:

**Required Environment Variables:**

| Variable | Description |
|----------|-------------|
| `LITELLM_MASTER_KEY` | **Critical**. Admin master key. |
| `POSTGRES_PASSWORD` | Password for the PostgreSQL database. |
| `DATABASE_URL` | Connection string (e.g., `postgresql://user:pass@db:5432/routeiq`). |
| `REDIS_HOST` | Hostname for Redis (e.g., `redis`). |

```bash
# Generate secure keys
export LITELLM_MASTER_KEY=$(openssl rand -hex 32)
export POSTGRES_PASSWORD=$(openssl rand -hex 16)
# Update your .env file with these keys
```

## 2. Start the HA Cluster

Use the `docker-compose.ha.yml` file:

```bash
docker-compose -f docker-compose.ha.yml up -d
```

## 3. Access Points

- **Load Balancer (Nginx)**: `http://localhost:8080` (Main entry point)
- **Gateway 1**: `http://localhost:4000` (Direct access)
- **Gateway 2**: `http://localhost:4001` (Direct access)

## 4. Verify HA & Sync

1.  **Check Health**:
    ```bash
    curl http://localhost:8080/health
    ```

2.  **Test Load Balancing**:
    Send multiple requests to port `8080`. Nginx will distribute them between the two gateway instances.

3.  **Test Config Sync**:
    The gateways use Redis Pub/Sub to synchronize configuration changes. If you update a model or register an MCP server on one instance, the other will receive the update automatically.

## Configuration Details

### Database & Cache
- **PostgreSQL**: Stores persistent data (users, keys, models).
- **Redis**: Handles response caching and real-time synchronization events.

### Environment Variables

| Variable | Description | Setting in HA |
|----------|-------------|---------------|
| `DATABASE_URL` | Postgres connection string. | Required |
| `REDIS_HOST` | Redis hostname. | `redis` |
| `CONFIG_SYNC_ENABLED` | Enable config synchronization. | `true` |
| `MCP_HA_SYNC_ENABLED` | Sync MCP servers via Redis. | `true` (default) |

## Production Notes

- **Secret Management**: In a real production environment, use a secret manager (e.g., AWS Secrets Manager, Kubernetes Secrets) instead of environment variables.
- **Scaling**: You can add more gateway replicas by duplicating the service definition in `docker-compose.ha.yml` and updating the Nginx upstream config.
- **Security**: Ensure `LITELLM_MASTER_KEY` and `POSTGRES_PASSWORD` are strong and rotated regularly.

## Next Steps

- **Need visibility?** Add the [Observability Stack](quickstart-otel-compose.md).
- **Security:** Review the [Security Guide](security.md).
- **Back to Basics:** [Simple Docker Compose](quickstart-docker-compose.md).
