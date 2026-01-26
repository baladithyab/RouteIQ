#!/usr/bin/env bash
# Lint helper script for Lefthook
# Prefers uv when available, falls back to python -m
#
# Usage:
#   ./scripts/lint.sh format [files...]   # Format Python files
#   ./scripts/lint.sh check [files...]    # Check Python files
#   ./scripts/lint.sh yaml [files...]     # Lint YAML files

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Detect runner: prefer uv, fallback to python -m
run_tool() {
    local tool="$1"
    shift
    if command -v uv &>/dev/null; then
        uv run "$tool" "$@"
    else
        python -m "$tool" "$@"
    fi
}

# Get files, filtering out empty args
get_files() {
    local files=()
    for arg in "$@"; do
        if [[ -n "$arg" && -f "$arg" ]]; then
            files+=("$arg")
        fi
    done
    echo "${files[@]:-}"
}

case "${1:-help}" in
    format)
        shift
        files=$(get_files "$@")
        if [[ -n "$files" ]]; then
            run_tool ruff format $files
        fi
        ;;
    check)
        shift
        files=$(get_files "$@")
        if [[ -n "$files" ]]; then
            run_tool ruff check --fix --exit-non-zero-on-fix $files
        fi
        ;;
    yaml)
        shift
        files=$(get_files "$@")
        if [[ -n "$files" ]]; then
            run_tool yamllint -c "$REPO_ROOT/.yamllint.yaml" $files
        fi
        ;;
    help|*)
        echo "Usage: $0 {format|check|yaml} [files...]"
        echo ""
        echo "Commands:"
        echo "  format   Run ruff format on Python files"
        echo "  check    Run ruff check with auto-fix on Python files"
        echo "  yaml     Run yamllint on YAML files"
        exit 1
        ;;
esac
