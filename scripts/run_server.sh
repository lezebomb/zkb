#!/usr/bin/env bash
set -euo pipefail

HOST="${APP_HOST:-0.0.0.0}"
PORT="${APP_PORT:-8080}"

uvicorn --app-dir src contest_agent.app:app --host "$HOST" --port "$PORT"
