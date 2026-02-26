#!/bin/bash

# Configuration
PROJECT_DIR="/Users/mingjian/Documents/sync/GitHub/siliconlm"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG_DIR="$HOME/Library/Logs/SiliconLM"
SERVER_LOG="$LOG_DIR/server.log"
EMBED_LOG="$LOG_DIR/embedding.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Navigate to project directory
cd "$PROJECT_DIR" || exit 1

# Kill existing instances if running (cleanup)
pkill -f "python server.py" 2>/dev/null
pkill -f "python embedding_server.py" 2>/dev/null

echo "Starting SiliconLM..."

# Start Embedding Server (Background)
echo "Starting Embedding Server..." >> "$EMBED_LOG"
nohup "$VENV_PYTHON" embedding_server.py >> "$EMBED_LOG" 2>&1 &
EMBED_PID=$!
echo "Embedding Server PID: $EMBED_PID"

# Wait a moment for embedding server
sleep 2

# Start Main Server (Foreground - kept alive by launchctl)
# We don't use nohup here because launchctl expects the process to stay alive
echo "Starting Main Server..." >> "$SERVER_LOG"
exec "$VENV_PYTHON" server.py >> "$SERVER_LOG" 2>&1
