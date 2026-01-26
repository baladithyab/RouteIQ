#!/usr/bin/env bash
# Bootstrap script to install Lefthook and configure git hooks
#
# Usage: ./scripts/install_lefthook.sh
#
# This script:
# 1. Checks if Lefthook is already installed globally
# 2. If not, downloads the latest version to .lefthook-local/
# 3. Runs lefthook install to configure git hooks
#
# Environment variables:
#   LEFTHOOK_VERSION - Override the version to install (default: latest)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_BIN="$REPO_ROOT/.lefthook-local"

# Fetch latest version from GitHub releases if not specified
get_lefthook_version() {
    if [[ -n "${LEFTHOOK_VERSION:-}" ]]; then
        echo "$LEFTHOOK_VERSION"
        return
    fi
    
    # Query GitHub API for latest release
    local latest
    latest=$(curl -fsSL "https://api.github.com/repos/evilmartians/lefthook/releases/latest" \
        | grep '"tag_name"' \
        | sed -E 's/.*"v([^"]+)".*/\1/')
    
    if [[ -z "$latest" ]]; then
        echo "Failed to fetch latest version, falling back to 2.0.15" >&2
        echo "2.0.15"
    else
        echo "$latest"
    fi
}

# Detect OS and architecture
detect_platform() {
    local os arch
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    arch="$(uname -m)"

    case "$arch" in
        x86_64|amd64) arch="x86_64" ;;
        aarch64|arm64) arch="arm64" ;;
        *) echo "Unsupported architecture: $arch" >&2; exit 1 ;;
    esac

    case "$os" in
        linux) os="Linux" ;;
        darwin) os="macOS" ;;
        *) echo "Unsupported OS: $os" >&2; exit 1 ;;
    esac

    echo "${os}_${arch}"
}

# Check if lefthook is already available
find_lefthook() {
    if command -v lefthook &>/dev/null; then
        echo "lefthook"
        return 0
    elif [[ -x "$LOCAL_BIN/lefthook" ]]; then
        echo "$LOCAL_BIN/lefthook"
        return 0
    fi
    return 1
}

# Download and install Lefthook locally
install_lefthook_local() {
    local version platform download_url tarball

    version="$(get_lefthook_version)"
    platform="$(detect_platform)"
    download_url="https://github.com/evilmartians/lefthook/releases/download/v${version}/lefthook_${version}_${platform}.gz"

    echo "📦 Installing Lefthook v${version} for ${platform}..."
    mkdir -p "$LOCAL_BIN"

    # Download and extract
    tarball="$LOCAL_BIN/lefthook.gz"
    curl -fsSL "$download_url" -o "$tarball"
    gunzip -f "$tarball"
    chmod +x "$LOCAL_BIN/lefthook"

    echo "✅ Lefthook installed to $LOCAL_BIN/lefthook"
}

main() {
    cd "$REPO_ROOT"

    echo "🔧 Setting up Lefthook for litellm-llm-router..."
    echo ""

    # Find or install lefthook
    if lefthook_bin=$(find_lefthook); then
        echo "✅ Found Lefthook: $lefthook_bin"
    else
        echo "⚠️  Lefthook not found, installing locally..."
        install_lefthook_local
        lefthook_bin="$LOCAL_BIN/lefthook"
    fi

    # Install git hooks
    echo ""
    echo "📌 Installing git hooks..."
    "$lefthook_bin" install

    echo ""
    echo "🎉 Setup complete! Git hooks are now active."
    echo ""
    echo "Commands:"
    echo "  lefthook run pre-commit    # Run pre-commit hooks manually"
    echo "  lefthook run post-commit   # Run post-commit hooks manually"
    echo ""
    echo "To uninstall: lefthook uninstall"
}

main "$@"
