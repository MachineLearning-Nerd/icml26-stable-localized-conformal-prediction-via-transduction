#!/usr/bin/env bash
# The one fixed reproduction command, inherited unchanged by every node.
# What a node runs is decided solely by the committed config/node.json.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v git-lfs >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq git git-lfs >/dev/null
    git lfs install --skip-repo
fi

export UV_PROJECT_ENVIRONMENT=.venv
uv sync --frozen

exec uv run --no-sync python -u src/run_node.py
