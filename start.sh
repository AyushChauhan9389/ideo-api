#!/usr/bin/env bash
# Start the Ideogram 4 API server (binds 0.0.0.0:8000 by default).
# Usage: ./start.sh [extra api_server.py args, e.g. --port 9000 --quantization fp8 --no...]
set -euo pipefail

cd "$(dirname "$0")"

# Use the project venv if one exists.
if [ -f ".venv/Scripts/activate" ]; then # Windows venv layout
  source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then # POSIX venv layout
  source .venv/bin/activate
fi

exec python api_server.py --preload "$@"
