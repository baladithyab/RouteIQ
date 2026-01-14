# Implementation Plan: Production AI Gateway

## Overview

This implementation plan focuses on validating the existing LiteLLM + LLMRouter integration, filling gaps in test coverage, and ensuring all production features are properly tested and documented. The system is already implemented, so tasks focus on validation, testing, and documentation rather than new feature development.

## Tasks

- [x] 1. Validate Core LiteLLM Integration
  - Verify all LiteLLM proxy endpoints are accessible and functional
  - Test authentication with master_key and virtual keys
  - Validate provider format support for key providers (OpenAI, Anthropic, Bedrock, Azure)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 1.1 Write property test for request forwarding
  - **Property 1: Request Forwarding Correctness**
  - **Validates: Requirements 1.2**

- [x] 1.2 Write property test for authentication enforcement
  - **Property 2: Authentication Enforcement**
  - **Validates: Requirements 1.3, 7.5, 11.1, 11.4**

- [x] 1.3 Write property test for configuration loading
  - **Property 3: Configuration Loading**
  - **Validates: Requirements 1.5**

- [x] 2. Validate LLMRouter Strategy Integration
  - Verify all 18+ LLMRouter strategies are registered
  - Test routing strategy selection for each strategy type
  - Validate model and LLM candidate data loading
  - Test fallback to LiteLLM built-in strategies
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 2.1 Write property test for routing strategy selection
  - **Property 4: Routing Strategy Selection**
  - **Validates: Requirements 2.2, 2.5, 2.6**

- [x] 3. Validate Hot Reload and Config Sync
  - Test model file hot reload on modification time change
  - Test S3 ETag-based config sync
  - Verify hot reload status endpoint
  - Test config reload trigger via API
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 3.1 Write property test for model hot reload
  - **Property 5: Model and Config Hot Reload**
  - **Validates: Requirements 3.2, 3.4, 3.5**

- [x] 3.2 Write property test for S3 ETag optimization
  - **Property 22: S3 Config Sync with ETag Optimization**
  - **Validates: Requirements 10.3**

- [x] 4. Validate High Availability Infrastructure
  - Test PostgreSQL persistence for virtual keys and request logs
  - Test Redis caching with TTL
  - Test Redis-based rate limiting
  - Verify health check endpoints
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 4.1 Write property test for data persistence
  - **Property 6: Data Persistence**
  - **Validates: Requirements 4.1, 12.2**

- [x] 4.2 Write property test for response caching
  - **Property 7: Response Caching**
  - **Validates: Requirements 4.2, 13.1, 13.3**

- [x] 4.3 Write property test for cache key generation
  - **Property 8: Cache Key Generation**
  - **Validates: Requirements 13.4**

- [x] 4.4 Write property test for rate limiting
  - **Property 9: Rate Limiting Enforcement**
  - **Validates: Requirements 11.6**

- [x] 5. Checkpoint - Ensure all core integration tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Validate MLOps Training Pipeline
  - Test Docker Compose setup in examples/mlops
  - Verify training for key strategy types (KNN, SVM, MLP)
  - Test model artifact compatibility with hot reload
  - Verify evaluation metrics are computed
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [x] 6.1 Write property test for MLOps model training
  - **Property 14: MLOps Model Training**
  - **Validates: Requirements 5.2, 5.3, 5.5**

- [x] 7. Implement OpenTelemetry Observability Integration
  - Add OpenTelemetry SDK dependencies (opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp)
  - Configure OpenTelemetry tracer, logger, and meter providers
  - Implement span emission for routing decisions, LLM calls, and cache operations
  - Implement structured logging with trace correlation (trace_id, span_id in logs)
  - Configure OTLP exporters for traces and logs
  - Add OpenTelemetry semantic conventions for HTTP and LLM spans
  - _Requirements: 6.2, 6.3, 6.5, 6.9, 15.4, 15.5_

- [x] 7.1 Write unit tests for OpenTelemetry integration
  - Test tracer initialization and configuration
  - Test span creation for key operations
  - Test log correlation with trace context
  - Test OTLP exporter configuration

- [ ] 8. Validate Observability and Tracing
  - Test Prometheus metrics endpoint
  - Verify OpenTelemetry span emission with OTLP exporter
  - Verify OpenTelemetry structured logging with trace correlation
  - Test Langfuse integration
  - Verify per-team observability settings
  - Test semantic conventions compliance
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9_

- [ ] 8.1 Write property test for observability span and log emission
  - **Property 15: Observability Span and Log Emission**
  - **Validates: Requirements 6.3, 6.4, 6.5, 6.9, 13.5, 14.6, 15.2, 15.4, 15.5**

- [ ] 8.2 Write property test for per-team observability
  - **Property 16: Per-Team Observability Settings**
  - **Validates: Requirements 6.8**

- [ ] 9. Validate A2A Gateway
  - Test A2A agent registration via API
  - Test agent discovery and filtering by capability
  - Verify agent card endpoint returns A2A-compliant format
  - Test A2A message forwarding
  - Test A2A authentication with virtual keys
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [ ] 9.1 Write property test for A2A agent registration
  - **Property 11: A2A Agent Registration and Discovery**
  - **Validates: Requirements 7.2, 7.6, 7.13**

- [ ] 9.2 Implement A2A Agent Invocation Endpoint
  - Add POST `/a2a/{agent_id}` endpoint for JSON-RPC 2.0 message handling
  - Implement `message/send` method to forward messages to agent backend
  - Return JSON-RPC 2.0 compliant responses
  - _Requirements: 7.8, 7.9_

- [ ] 9.3 Write property test for A2A agent invocation
  - **Property 23: A2A Agent Invocation**
  - **Validates: Requirements 7.8, 7.9**

- [ ] 9.4 Implement A2A Streaming Support
  - Add `message/stream` method support to `/a2a/{agent_id}` endpoint
  - Implement Server-Sent Events (SSE) streaming for agent responses
  - Handle streaming errors and connection management
  - _Requirements: 7.10_

- [ ] 9.5 Write property test for A2A streaming
  - **Property 24: A2A Streaming Response**
  - **Validates: Requirements 7.10**

- [ ] 9.6 Implement A2A Database Persistence
  - Create `a2a_agents` table schema in PostgreSQL
  - Implement DB persistence for agent registration (POST `/v1/agents`)
  - Implement DB retrieval for agent listing (GET `/v1/agents`)
  - Add migration scripts for A2A tables
  - _Requirements: 7.7_

- [ ] 9.7 Write property test for A2A database persistence
  - **Property 25: A2A Database Persistence**
  - **Validates: Requirements 7.7**

- [ ] 9.8 Implement A2A Agent Updates
  - Add PUT `/v1/agents/{agent_id}` endpoint for full agent updates
  - Add PATCH `/v1/agents/{agent_id}` endpoint for partial updates
  - Implement permission checks for update operations
  - _Requirements: 7.11, 7.12_

- [ ] 9.9 Write property test for A2A agent updates
  - **Property 26: A2A Agent Updates**
  - **Validates: Requirements 7.11, 7.12**

- [ ] 9.10 Implement A2A Permission Filtering
  - Add user_id and team_id filtering to agent listing
  - Implement is_public flag support for public agents
  - Add `/v1/agents/{agent_id}/make_public` endpoint
  - _Requirements: 7.13_

- [ ] 9.11 Implement A2A Analytics Endpoint
  - Add `/agent/daily/activity` endpoint for usage analytics
  - Track agent invocation counts and latency
  - Support date range filtering
  - _Requirements: 7.14_

- [ ] 10. Validate MCP Gateway
  - Test MCP server registration via configuration
  - Verify tool loading from MCP servers
  - Test MCP tool invocation in chat completions
  - Test OpenAPI-to-MCP conversion
  - Verify support for different MCP transports
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

- [ ] 10.1 Write property test for MCP tool loading
  - **Property 12: MCP Server Tool Loading**
  - **Validates: Requirements 8.2, 8.3, 8.4**

- [ ] 10.2 Write property test for OpenAPI to MCP conversion
  - **Property 13: OpenAPI to MCP Conversion**
  - **Validates: Requirements 8.6**

- [ ] 10.3 Implement MCP Tool Invocation Endpoint
  - Add POST `/mcp/tools/call` endpoint for direct tool invocation
  - Add GET `/mcp/tools/list` endpoint for listing available tools
  - Implement tool argument validation against input schema
  - Handle tool execution errors gracefully
  - _Requirements: 8.8, 8.9_

- [ ] 10.4 Write property test for MCP tool invocation
  - **Property 27: MCP Tool Invocation**
  - **Validates: Requirements 8.8**

- [ ] 10.5 Implement MCP Database Persistence
  - Create `mcp_servers`, `mcp_tools`, `mcp_resources` table schemas
  - Implement DB persistence for server registration (POST `/v1/mcp/server`)
  - Implement DB retrieval for server listing (GET `/v1/mcp/server`)
  - Add migration scripts for MCP tables
  - _Requirements: 8.7_

- [ ] 10.6 Write property test for MCP database persistence
  - **Property 28: MCP Database Persistence**
  - **Validates: Requirements 8.7**

- [ ] 10.7 Implement MCP OAuth Support
  - Add POST `/v1/mcp/server/oauth/session` for temporary OAuth sessions
  - Add GET `/v1/mcp/server/oauth/{server_id}/authorize` for OAuth authorization
  - Add POST `/v1/mcp/server/oauth/{server_id}/token` for token exchange
  - Add GET `/.well-known/oauth-authorization-server` for OAuth discovery
  - _Requirements: 8.10, 8.11_

- [ ] 10.8 Write property test for MCP OAuth flow
  - **Property 29: MCP OAuth Flow**
  - **Validates: Requirements 8.10, 8.11**

- [ ] 10.9 Implement MCP Server Health Checks
  - Add GET `/v1/mcp/server/health` endpoint
  - Implement connectivity checks for each registered server
  - Return health status with latency metrics
  - _Requirements: 8.13_

- [ ] 10.10 Write property test for MCP health checks
  - **Property 30: MCP Server Health Check**
  - **Validates: Requirements 8.13**

- [ ] 10.11 Implement MCP Registry Endpoint
  - Add GET `/v1/mcp/registry.json` endpoint for MCP discovery
  - Generate registry document listing all servers and capabilities
  - Support filtering by access groups
  - _Requirements: 8.12_

- [ ] 10.12 Write property test for MCP registry
  - **Property 31: MCP Registry Discovery**
  - **Validates: Requirements 8.12**

- [ ] 10.13 Implement MCP Server Updates
  - Add PUT `/v1/mcp/server/{server_id}` endpoint for full server updates
  - Implement tool and resource refresh on server update
  - _Requirements: 8.14_

- [ ] 10.14 Implement MCP Access Groups
  - Add GET `/v1/mcp/access_groups` endpoint for listing access groups
  - Implement access group filtering for server visibility
  - _Requirements: 8.15_

- [ ] 11. Checkpoint - Ensure all gateway extension tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Validate Docker Multi-Architecture Support
  - Verify Docker images exist for amd64 and arm64
  - Test multi-stage build configuration
  - Verify non-root user (UID 1000)
  - Verify tini init process
  - Test health check configuration
  - Verify GHCR image publishing
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

- [ ] 13. Validate AWS Cloud Integration
  - Test AWS Bedrock model support
  - Test IAM role-based authentication
  - Test S3 config and model storage
  - Test AWS Secrets Manager integration
  - Verify CloudWatch X-Ray tracing via OpenTelemetry OTLP
  - Verify deployment example configurations
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [ ] 14. Validate Security and Authentication
  - Test virtual key creation and validation
  - Test per-key budget limits
  - Test team-based access control
  - Test database-backed key validation
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

- [ ] 14.1 Write property test for budget tracking
  - **Property 10: Budget Tracking and Enforcement**
  - **Validates: Requirements 11.5, 12.1, 12.3, 12.6**

- [ ] 15. Validate Cost Tracking and Management
  - Test cost calculation for requests
  - Verify cost persistence to database
  - Test budget enforcement
  - Verify Prometheus cost metrics
  - Test cost data export to external systems
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_

- [ ] 16. Validate Error Handling and Retries
  - Test retry behavior with exponential backoff
  - Test context window fallback
  - Test timeout enforcement
  - Verify error responses include details
  - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6_

- [ ] 16.1 Write property test for retry behavior
  - **Property 17: Retry with Exponential Backoff**
  - **Validates: Requirements 14.1, 14.5**

- [ ] 16.2 Write property test for context window fallback
  - **Property 18: Context Window Fallback**
  - **Validates: Requirements 14.3**

- [ ] 16.3 Write property test for timeout enforcement
  - **Property 19: Timeout Enforcement**
  - **Validates: Requirements 14.4**

- [ ] 17. Validate Logging and Debugging with OpenTelemetry
  - Test verbose logging mode via OpenTelemetry
  - Verify routing decision logging with trace correlation
  - Test privacy-preserving logging (no prompts when disabled)
  - Verify OpenTelemetry structured logging with OTLP exporter
  - Verify trace context (trace_id, span_id) in all log entries
  - Test startup logging
  - Verify error logging with stack traces and trace correlation
  - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7_

- [ ] 17.1 Write property test for routing decision logging
  - **Property 20: Routing Decision Logging with Trace Correlation**
  - **Validates: Requirements 15.2, 15.3, 15.5**

- [ ] 17.2 Write property test for error logging
  - **Property 21: Error Logging with Trace Context**
  - **Validates: Requirements 15.7**

- [ ] 18. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 19. Documentation and Deployment Validation
  - Review and update README with OpenTelemetry observability features
  - Add OpenTelemetry configuration examples to documentation
  - Document trace correlation and semantic conventions
  - Verify all documentation links are valid
  - Test Docker Compose setups (basic, HA, MLOps, with OTEL collector)
  - Validate AWS deployment examples with X-Ray via OTLP
  - Update API documentation with all endpoints
  - _Requirements: All_

## Notes

- All tasks are required for comprehensive validation and testing
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using Hypothesis
- Unit tests validate specific examples and edge cases
- The system is already implemented, so tasks focus on validation, testing, and adding OpenTelemetry observability
- Integration tests should use Docker Compose to test with real Redis, PostgreSQL, S3 (LocalStack), and OpenTelemetry Collector
- OpenTelemetry integration provides unified observability for traces, logs, and metrics
- All logs should include trace correlation IDs (trace_id, span_id) for distributed debugging
- **A2A Full Parity**: Tasks 9.2-9.11 implement full A2A protocol support including agent invocation, streaming, DB persistence, and analytics
- **MCP Full Parity**: Tasks 10.3-10.14 implement full MCP protocol support including tool invocation, OAuth, health checks, and registry
- Database migrations should be created for new A2A and MCP tables
- URL patterns should align with LiteLLM defaults (`/v1/agents`, `/v1/mcp/server`) for ecosystem compatibility
