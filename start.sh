#!/bin/bash
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"

echo "==> Starting Mental Health EWS"
echo ""

# Backend
echo "[1/2] Starting FastAPI backend (port 8000)…"
cd "$REPO/app/backend"
uvicorn api:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Give backend a moment to bind the port before browser opens
sleep 2

# Frontend
echo "[2/2] Starting React frontend (port 5173)…"
cd "$REPO/app/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "  Note: backend startup takes ~30-60s while the VLE file loads."
echo "  Press Ctrl-C to stop both servers."
echo ""

# On Ctrl-C, kill both child processes
cleanup() {
    echo ""
    echo "Stopping servers…"
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
    wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
    echo "Done."
}
trap cleanup INT TERM

wait
