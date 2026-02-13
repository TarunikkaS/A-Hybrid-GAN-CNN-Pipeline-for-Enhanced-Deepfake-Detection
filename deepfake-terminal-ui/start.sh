#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    DEEPFAKE DETECTOR - TERMINAL UI                           ║
# ║                         LAUNCH SCRIPT v1.0.0                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -e

echo ""
echo "███████╗████████╗ █████╗ ██████╗ ████████╗██╗███╗   ██╗ ██████╗ "
echo "██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝██║████╗  ██║██╔════╝ "
echo "███████╗   ██║   ███████║██████╔╝   ██║   ██║██╔██╗ ██║██║  ███╗"
echo "╚════██║   ██║   ██╔══██║██╔══██╗   ██║   ██║██║╚██╗██║██║   ██║"
echo "███████║   ██║   ██║  ██║██║  ██║   ██║   ██║██║ ╚████║╚██████╔╝"
echo "╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝ "
echo ""
echo "[*] DEEPFAKE DETECTOR - TERMINAL UI"
echo "[*] ================================"
echo ""

# Navigate to project directory
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "[>] Installing Node.js dependencies..."
    npm install
    echo "[OK] Dependencies installed"
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "[*] Shutting down services..."
    kill $API_PID 2>/dev/null || true
    kill $NEXT_PID 2>/dev/null || true
    echo "[OK] Services stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start the Python API server
echo "[>] Starting Python API server..."
cd "$PROJECT_ROOT/.."
python -m uvicorn deepfake-terminal-ui.api.server:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!
cd "$PROJECT_ROOT"

# Wait for API to be ready
echo "[>] Waiting for API to be ready..."
sleep 3

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "[OK] API server is running on http://localhost:8000"
else
    echo "[WARN] API server might still be starting..."
fi

# Start the Next.js development server
echo "[>] Starting Next.js frontend..."
npm run dev &
NEXT_PID=$!

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    SERVICES STARTED                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Frontend:  http://localhost:3000                            ║"
echo "║  API:       http://localhost:8000                            ║"
echo "║  API Docs:  http://localhost:8000/docs                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "[*] Press Ctrl+C to stop all services"
echo ""

# Wait for processes
wait
