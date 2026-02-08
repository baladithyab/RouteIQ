#!/usr/bin/env python3
"""
Secret Detection Script for Lefthook

OS-agnostic script to scan files for common API keys, tokens, and credentials.

Usage:
    python scripts/detect_secrets.py [files...]        # Scan specific files
    python scripts/detect_secrets.py --full-scan       # Scan entire repo (pre-push)
    uv run python scripts/detect_secrets.py [files...] # Via uv
"""

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

# =============================================================================
# Configuration
# =============================================================================

# Secret patterns: (name, regex_pattern)
SECRET_PATTERNS: list[tuple[str, str]] = [
    # AWS
    ("AWS Access Key", r"AKIA[0-9A-Z]{16}"),
    ("AWS Secret Key", r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}"),
    # OpenAI/AI providers
    ("OpenAI API Key", r"sk-[A-Za-z0-9_-]{32,}"),
    ("Anthropic API Key", r"sk-ant-[A-Za-z0-9_-]{32,}"),
    # Google
    ("Google API Key", r"AIza[0-9A-Za-z_-]{35}"),
    # GitHub
    ("GitHub Token", r"gh[pousr]_[A-Za-z0-9_]{36}"),
    ("GitHub Personal Token", r"github_pat_[A-Za-z0-9_]{22,}"),
    # Stripe
    ("Stripe Secret Key", r"sk_live_[A-Za-z0-9]{24,}"),
    ("Stripe Restricted Key", r"rk_live_[A-Za-z0-9]{24,}"),
    # Slack
    ("Slack Token", r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}[a-zA-Z0-9-]*"),
    (
        "Slack Webhook",
        r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[a-zA-Z0-9]+",
    ),
    # Private keys
    ("Private Key Block", r"-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----"),
    # Generic high-entropy patterns (be careful with false positives)
    ("Bearer Token", r"(?i)bearer\s+[A-Za-z0-9_-]{20,}"),
]

# Directories to ignore
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "reference",
    ".tox",
    "build",
    "dist",
    "*.egg-info",
}

# Known fake/test tokens that are safe (used in tests and examples)
# Add any intentionally fake tokens here
ALLOWLISTED_SECRETS = {
    # Test bearer tokens
    "Bearer test_token_12345",
    "Bearer token-please-use-me",
    "Bearer invalid-key-12345",  # Used in e2e tests
    "Bearer invalid-user-key-12345",  # Used in integration tests
    "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",  # Example JWT header (test value)
    "test_token_12345",
    "token-please-use-me",
    "invalid-key-12345",
    "invalid-user-key-12345",
    # RBAC test fixture tokens (test_rbac_enforcement.py)
    "Bearer user-with-mcp-server-write",
    "Bearer user-with-mcp-tool-call",
    "Bearer user-with-mcp-wildcard",
    # Docker Compose health check tokens (local dev/test only)
    "Bearer local-dev-master-key",
    "Bearer quota-test-master-key",
    "Bearer local-dev-streaming-perf-key",
    # Example API keys (obviously fake patterns)
    "AKIAIOSFODNN7EXAMPLE",  # AWS example key
    "test-fake-key-12345",
    "proj-fake-key-for-testing",
}

# File patterns to ignore
IGNORE_FILE_PATTERNS = [
    r"\.env\.example$",
    r"\.lock$",
    r"\.pyc$",
    r"\.pyo$",
    r"\.so$",
    r"\.o$",
    r"\.a$",
    r"\.dll$",
    r"\.exe$",
    r"\.bin$",
    r"\.png$",
    r"\.jpg$",
    r"\.jpeg$",
    r"\.gif$",
    r"\.ico$",
    r"\.woff2?$",
    r"\.ttf$",
    r"\.eot$",
    r"\.md$",  # Markdown files (documentation)
    r"_REPORT\.md$",  # Report files
    r"REPORT\.json$",  # JSON reports
]

# Compile ignore patterns
IGNORE_FILE_REGEXES = [re.compile(p) for p in IGNORE_FILE_PATTERNS]


class Finding(NamedTuple):
    """A secret finding."""

    file: Path
    line_num: int
    pattern_name: str
    masked_line: str


def should_ignore_path(path: Path) -> bool:
    """Check if a path should be ignored."""
    parts = path.parts
    for ignore_dir in IGNORE_DIRS:
        if ignore_dir in parts:
            return True

    filename = path.name
    for regex in IGNORE_FILE_REGEXES:
        if regex.search(filename):
            return True

    return False


def is_binary_file(path: Path) -> bool:
    """Check if a file is binary."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            # Check for null bytes (common in binary files)
            if b"\x00" in chunk:
                return True
            # Try to decode as UTF-8
            try:
                chunk.decode("utf-8")
                return False
            except UnicodeDecodeError:
                return True
    except (IOError, OSError):
        return True


def mask_secret(line: str, pattern: str) -> str:
    """Mask the secret value in a line."""

    def replacer(m: re.Match) -> str:
        matched = m.group(0)
        if len(matched) <= 8:
            return "*" * len(matched)
        return matched[:4] + "***MASKED***" + matched[-4:]

    return re.sub(pattern, replacer, line)


def scan_file(path: Path) -> list[Finding]:
    """Scan a single file for secrets."""
    findings: list[Finding] = []

    if should_ignore_path(path):
        return findings

    if not path.exists() or not path.is_file():
        return findings

    if is_binary_file(path):
        return findings

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return findings

    lines = content.splitlines()

    for line_num, line in enumerate(lines, start=1):
        # Skip lines containing allowlisted tokens
        line_allowlisted = any(token in line for token in ALLOWLISTED_SECRETS)
        if line_allowlisted:
            continue

        for pattern_name, pattern in SECRET_PATTERNS:
            match = re.search(pattern, line)
            if match:
                # Check if the matched value is allowlisted
                matched_value = match.group(0)
                if matched_value in ALLOWLISTED_SECRETS:
                    continue

                masked = mask_secret(line.strip(), pattern)
                findings.append(
                    Finding(
                        file=path,
                        line_num=line_num,
                        pattern_name=pattern_name,
                        masked_line=masked[:100] + "..."
                        if len(masked) > 100
                        else masked,
                    )
                )

    return findings


def scan_directory(root: Path, max_files: int = 1000) -> list[Finding]:
    """Scan a directory recursively for secrets."""
    findings: list[Finding] = []
    file_count = 0

    for path in root.rglob("*"):
        if file_count >= max_files:
            print(
                f"⚠️  Reached max file limit ({max_files}), stopping scan",
                file=sys.stderr,
            )
            break

        if path.is_file() and not should_ignore_path(path):
            file_count += 1
            findings.extend(scan_file(path))

    return findings


def print_findings(findings: list[Finding]) -> None:
    """Print findings in a formatted way."""
    if not findings:
        print("\033[32m✅ No secrets detected\033[0m")
        return

    # Group by file
    by_file: dict[Path, list[Finding]] = {}
    for f in findings:
        by_file.setdefault(f.file, []).append(f)

    for file_path, file_findings in sorted(by_file.items()):
        print(f"\033[31m❌ Potential secrets in: {file_path}\033[0m")
        for finding in file_findings:
            print(
                f"   \033[33m⚠ {finding.pattern_name}\033[0m (line {finding.line_num})"
            )
            print(f"      {finding.masked_line}")
        print()

    print(
        f"\n\033[31m❌ Found {len(findings)} potential secret(s) in {len(by_file)} file(s)\033[0m"
    )
    print()
    print("   If these are false positives, you can:")
    print("   1. Add to .gitignore if the file shouldn't be committed")
    print("   2. Use environment variables instead of hardcoded values")
    print("   3. Add pattern to IGNORE_FILE_PATTERNS in this script")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Scan files for secrets and API keys",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Scan entire repository",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to scan",
    )

    args = parser.parse_args()

    if args.full_scan:
        # Find repo root (look for .git directory)
        root = Path.cwd()
        while root.parent != root:
            if (root / ".git").exists():
                break
            root = root.parent

        print(f"🔒 Scanning repository: {root}")
        findings = scan_directory(root)
    elif args.files:
        # Filter to existing files
        files = [Path(f) for f in args.files if f and Path(f).exists()]
        if not files:
            print("✅ No files to scan")
            return 0

        print(f"🔍 Scanning {len(files)} file(s)...")
        findings = []
        for f in files:
            findings.extend(scan_file(f))
    else:
        parser.print_help()
        return 1

    print_findings(findings)

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
