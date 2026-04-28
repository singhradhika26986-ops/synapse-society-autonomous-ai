#!/bin/sh
set -e
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
exec uvicorn production_app:app --host "$HOST" --port "$PORT"
