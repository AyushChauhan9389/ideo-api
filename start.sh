#!/usr/bin/env bash
# Start the Ideogram 4 API server (binds 0.0.0.0:8000 by default).
# Usage: ./start.sh [extra api_server.py args, e.g. --port 9000 --quantization fp8]
#
# The HF token (needed once to download the gated weights) is resolved from:
#   1. HF_TOKEN env var
#   2. .hf_token file in this directory (created on first run)
#   3. the hf CLI's cached login (~/.cache/huggingface/token)
# If none exist, you are prompted once and the token is saved to .hf_token.
set -euo pipefail

cd "$(dirname "$0")"

TOKEN_FILE=".hf_token"

if [ -z "${HF_TOKEN:-}" ]; then
  if [ -f "$TOKEN_FILE" ]; then
    HF_TOKEN="$(<"$TOKEN_FILE")"
  elif [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(<"$HOME/.cache/huggingface/token")"
  else
    echo "No Hugging Face token found (the model weights are gated)."
    echo "Get one at https://huggingface.co/settings/tokens (read access is enough),"
    echo "and make sure you accepted the gate on https://huggingface.co/ideogram-ai/ideogram-4-nf4 (or -fp8)."
    read -r -s -p "Paste your HF token (hf_...): " HF_TOKEN
    echo
    if [ -z "$HF_TOKEN" ]; then
      echo "No token entered, aborting." >&2
      exit 1
    fi
    printf '%s' "$HF_TOKEN" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE" 2>/dev/null || true
    echo "Saved to $TOKEN_FILE for future runs."
  fi
fi
export HF_TOKEN

# Use the project venv if one exists.
if [ -f ".venv/Scripts/activate" ]; then # Windows venv layout
  source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then # POSIX venv layout
  source .venv/bin/activate
fi

PID_FILE=".server.pid"
LOG_FILE="server.log"

# Don't start a second instance if one is already running.
if [ -f "$PID_FILE" ] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
  echo "Server already running (pid $(<"$PID_FILE")). Logs: $LOG_FILE"
  echo "To stop it: kill $(<"$PID_FILE")"
  exit 0
fi

nohup python api_server.py --preload "$@" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

echo "Server starting in background (pid $SERVER_PID) on 0.0.0.0:8000."
echo "First run downloads + loads the model weights, which can take a while."
echo "  logs:   tail -f $LOG_FILE"
echo "  check:  curl http://127.0.0.1:8000/health"
echo "  stop:   kill $SERVER_PID"
