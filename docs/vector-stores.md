# Vector Stores Integration

RouteIQ Gateway provides vector store capabilities through two primary mechanisms: inherited OpenAI-compatible endpoints and planned deep integrations with external vector databases.

## Inherited OpenAI-Compatible Endpoints

RouteIQ Gateway inherits the standard OpenAI `/v1/vector_stores` endpoints from the upstream LiteLLM proxy. This allows you to use the gateway as a drop-in replacement for OpenAI's file search and vector store APIs.

### Supported Endpoints

- `POST /v1/vector_stores` - Create a vector store.
- `GET /v1/vector_stores` - List vector stores.
- `GET /v1/vector_stores/{vector_store_id}` - Retrieve a vector store.
- `POST /v1/vector_stores/{vector_store_id}` - Modify a vector store.
- `DELETE /v1/vector_stores/{vector_store_id}` - Delete a vector store.
- `POST /v1/vector_stores/{vector_store_id}/file_batches` - Create a file batch.
- `GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}` - Retrieve a file batch.
- `GET /v1/vector_stores/{vector_store_id}/files` - List vector store files.

These endpoints are fully compatible with the OpenAI SDK and can be used immediately with any model that supports file search tools.

## Planned External Integrations

> **Note**: Deep integration with external Vector Databases (Pinecone, Weaviate, etc.) is currently **planned** and not yet fully implemented in RouteIQ Gateway. The following configuration examples represent the target design for future releases.

RouteIQ Gateway aims to provide a unified interface for external vector databases, allowing you to route RAG queries to different stores based on policy or performance.

### OpenSearch

OpenSearch integration for enterprise-grade vector search with hybrid query support.

```yaml
# Planned configuration
vector_stores:
  opensearch:
    enabled: true
    hosts:
      - "https://opensearch-node:9200"
    index_name: "embeddings"
    auth:
      type: "basic"  # or "aws_sigv4"
      username: os.environ/OPENSEARCH_USER
      password: os.environ/OPENSEARCH_PASSWORD
    embedding_model: "text-embedding-3-small"
    dimensions: 1536
```

### LanceDB

LanceDB integration for embedded vector database with multi-modal support.

```yaml
# Planned configuration
vector_stores:
  lancedb:
    enabled: true
    uri: "/app/data/lancedb"  # Local or S3 path
    table_name: "embeddings"
    embedding_model: "text-embedding-3-small"
```

### Qdrant

Qdrant integration for high-performance vector similarity search.

```yaml
# Planned configuration
vector_stores:
  qdrant:
    enabled: true
    url: "http://qdrant:6333"
    api_key: os.environ/QDRANT_API_KEY
    collection_name: "embeddings"
    embedding_model: "text-embedding-3-small"
    vector_size: 1536
```

### Pinecone

Pinecone integration for managed vector database with serverless options.

```yaml
# Planned configuration
vector_stores:
  pinecone:
    enabled: true
    api_key: os.environ/PINECONE_API_KEY
    environment: "us-east-1"
    index_name: "embeddings"
    embedding_model: "text-embedding-3-small"
    namespace: "default"
```

### Milvus

Milvus integration for scalable vector database with GPU acceleration.

```yaml
# Planned configuration
vector_stores:
  milvus:
    enabled: true
    host: "milvus"
    port: 19530
    collection_name: "embeddings"
    embedding_model: "text-embedding-3-small"
    index_type: "IVF_FLAT"
    metric_type: "L2"
```

## Planned Features

### Unified Vector Store API

```bash
# Store embeddings
POST /v1/vector/store
{
  "store": "opensearch",
  "documents": [
    {"id": "doc1", "text": "...", "metadata": {...}}
  ]
}

# Search vectors
POST /v1/vector/search
{
  "store": "opensearch",
  "query": "What is machine learning?",
  "top_k": 5
}
```

### RAG Integration with Routing

The vector store integration will work seamlessly with LLMRouter strategies:

```yaml
router_settings:
  routing_strategy: llmrouter-knn
  rag_enabled: true
  rag_settings:
    vector_store: "opensearch"
    top_k: 5
    rerank: true
    rerank_model: "cohere-rerank"
```

### Multi-Store Support

Query across multiple vector stores:

```yaml
vector_stores:
  primary:
    type: "opensearch"
    # ...
  secondary:
    type: "qdrant"
    # ...

rag_settings:
  stores: ["primary", "secondary"]
  merge_strategy: "reciprocal_rank_fusion"
```

## Environment Variables (Planned)

| Variable | Description |
|----------|-------------|
| `VECTOR_STORE_TYPE` | Default vector store type |
| `OPENSEARCH_HOSTS` | OpenSearch cluster hosts |
| `QDRANT_URL` | Qdrant server URL |
| `PINECONE_API_KEY` | Pinecone API key |
| `MILVUS_HOST` | Milvus server host |
| `LANCEDB_URI` | LanceDB data path |

## Contributing

We welcome contributions for vector store integrations! See our [contribution guide](../CONTRIBUTING.md) for details.

Priority integrations:
1. OpenSearch (enterprise search)
2. LanceDB (embedded/serverless)
3. Qdrant (high-performance)
4. Pinecone (managed service)
5. Milvus (GPU-accelerated)

## See Also

- [MCP Gateway](mcp-gateway.md) - External tool integration
- [Routing Strategies](routing-strategies.md) - ML-powered routing
- [High Availability](high-availability.md) - Production deployment
