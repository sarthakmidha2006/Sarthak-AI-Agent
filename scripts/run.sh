#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Run the FastAPI "brain" app locally with autoreload.
# Thin wrapper around uvicorn; host/port are read from the environment (.env),
# defaulting to the values in app/config.py Settings.
#   ./scripts/run.sh
# Override ad-hoc:  API_HOST=127.0.0.1 API_PORT=9000 ./scripts/run.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# Resolve project root (parent of this scripts/ directory) and run from there.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${PROJECT_ROOT}"

# uvicorn binds to API_HOST/API_PORT; the app itself loads the rest of .env via
# pydantic-settings. We only need those two values here, so we EXTRACT them
# rather than `source` the whole file — sourcing would execute arbitrary content
# and break on unquoted spaces (e.g. PERSONA_NAME=the candidate).
read_env() {
  # read_env KEY -> last non-comment "KEY=value" in .env, surrounding quotes stripped
  local key="$1" line val
  [[ -f ".env" ]] || return 0
  line="$(grep -E "^[[:space:]]*${key}=" ".env" 2>/dev/null | tail -n1 || true)"
  [[ -n "${line}" ]] || return 0
  val="${line#*=}"
  val="${val%\"}"; val="${val#\"}"   # strip paired double quotes
  val="${val%\'}"; val="${val#\'}"   # strip paired single quotes
  printf '%s' "${val}"
}

HOST="${API_HOST:-$(read_env API_HOST)}"; HOST="${HOST:-0.0.0.0}"
PORT="${API_PORT:-$(read_env API_PORT)}"; PORT="${PORT:-8000}"

exec uvicorn app.main:app --reload --host "${HOST}" --port "${PORT}"
