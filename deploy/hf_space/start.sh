#!/usr/bin/env bash
# Start FastAPI in background, wait for it to be healthy, then start
# Streamlit in the foreground so the container stays alive.
set -euo pipefail

echo "[start.sh] launching uvicorn on :8000"
uvicorn api.main:app --host 127.0.0.1 --port 8000 --log-level warning &
API_PID=$!

echo "[start.sh] waiting for API health"
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "[start.sh] API up after ${i}s"
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "[start.sh] uvicorn died before becoming healthy" >&2
    exit 1
  fi
  sleep 1
done

if ! curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "[start.sh] API never became healthy" >&2
  exit 1
fi

echo "[start.sh] launching streamlit on :8501"
exec streamlit run dashboard/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
