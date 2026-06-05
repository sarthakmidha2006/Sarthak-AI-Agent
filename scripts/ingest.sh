#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Ingest the persona corpus (resume + GitHub) into the vector / BM25 indexes.
# Thin wrapper around `python -m app.ingestion.run_ingest`.
# All arguments are forwarded, e.g.:
#   ./scripts/ingest.sh --reset
#   ./scripts/ingest.sh --sources github --username some-user
# -----------------------------------------------------------------------------
set -euo pipefail

# Resolve project root (parent of this scripts/ directory) and run from there so
# relative paths in Settings (./data/...) and the package import resolve.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${PROJECT_ROOT}"

# Allow overriding the interpreter via the PYTHON env var.
PYTHON_BIN="${PYTHON:-python}"

exec "${PYTHON_BIN}" -m app.ingestion.run_ingest "$@"
