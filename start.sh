#!/bin/sh
set -e
PORT="${PORT:-10000}"
exec streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0
