# Contributing to LiteLLM + LLMRouter

Thank you for your interest in contributing! This document provides guidelines for development.

## Development Setup

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/baladithyab/litellm-llm-router.git
cd litellm-llm-router

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start local development stack
docker compose -f docker-compose.local-test.yml up -d
```

### Building the Container

```bash
# Production build
docker build -f docker/Dockerfile -t litellm-llmrouter:latest .

# Local development build
docker build -f docker/Dockerfile.local -t litellm-llmrouter:local .

# Build with specific versions
docker build -f docker/Dockerfile \
  --build-arg LITELLM_VERSION=1.80.15 \
  --build-arg PYTHON_VERSION=3.12 \
  -t litellm-llmrouter:latest .
```

## Project Structure

```
litellm-llm-router/
├── config/                  # Configuration files
├── custom_routers/          # Custom routing strategies
├── docker/                  # Dockerfiles and entrypoints
├── docs/                    # Documentation
│   └── architecture/        # Architecture docs
├── examples/                # Example configurations
│   └── mlops/              # MLOps training stack
├── models/                  # Trained router models
├── scripts/                 # Utility scripts
├── src/                     # Source code
│   └── litellm_llmrouter/  # Custom integration
└── tests/                   # Test suite
```

## Making Changes

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to functions and classes
- Keep functions focused and small

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/litellm_llmrouter

# Run specific test
pytest tests/test_strategies.py -v
```

### Documentation

- Update relevant docs when making changes
- Add docstrings to new code
- Include examples where helpful

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

### PR Checklist

- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Code follows project style
- [ ] Commit messages are clear
- [ ] No sensitive data committed

## Docker Best Practices

When modifying Dockerfiles:

1. **Multi-stage builds**: Keep production images minimal
2. **Layer caching**: Order commands to maximize cache hits
3. **Security**: Run as non-root user, use tini
4. **Labels**: Include OCI image labels

## Adding New Features

### New Routing Strategy

1. Create strategy in `custom_routers/`
2. Register in configuration
3. Add tests
4. Document in `docs/routing-strategies.md`

### New LLM Provider

1. Add to `config/config.yaml`
2. Test with local development stack
3. Update documentation

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Be clear and provide context in issues
