# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of this project seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT create a public GitHub issue** for security vulnerabilities.

2. **Send a private report** using one of the following methods:
   - **Preferred**: Use [GitHub's private vulnerability reporting](https://github.com/YOUR_ORG/litellm-llm-router/security/advisories/new)
   - **Alternative**: Email security concerns to the maintainers (see repository maintainer contacts)

3. **Include the following information**:
   - Type of vulnerability (e.g., injection, authentication bypass, information disclosure)
   - Step-by-step instructions to reproduce the issue
   - Proof-of-concept or exploit code (if possible)
   - Impact assessment of the vulnerability
   - Any potential mitigations you've identified

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours.
- **Initial Assessment**: We will provide an initial assessment within 7 days.
- **Updates**: We will keep you informed of our progress toward a fix.
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days.
- **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous).

### Scope

The following are in scope for security reports:

- The main application code (`src/`)
- Docker configurations and images
- CI/CD pipelines (`.github/workflows/`)
- Configuration files that may expose secrets
- Dependencies with known vulnerabilities affecting this project

### Out of Scope

- Issues in upstream dependencies (report these to the respective projects)
- Denial of service attacks
- Issues requiring physical access
- Social engineering attacks

## Security Best Practices

When deploying this project:

1. **Never commit secrets** to the repository
2. **Use environment variables** or secret management tools for sensitive configuration
3. **Keep dependencies updated** using Dependabot alerts
4. **Review CodeQL scan results** regularly
5. **Use container scanning** in CI/CD pipelines
6. **Follow the principle of least privilege** for API keys and service accounts

## Security Features

This project implements the following security measures:

- **Automated dependency updates** via Dependabot
- **Static code analysis** via CodeQL
- **Container vulnerability scanning** via Trivy
- **Least-privilege CI permissions** with SHA-pinned actions
- **Non-root container execution**

Thank you for helping keep this project and its users safe!
