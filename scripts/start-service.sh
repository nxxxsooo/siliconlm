#!/bin/bash

# Configuration
PROJECT_DIR="/Users/mingjian/Documents/sync/GitHub/siliconlm"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
LOG_DIR="$HOME/Library/Logs/SiliconLM"
SERVER_LOG="$LOG_DIR/server.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Navigate to project directory
cd "$PROJECT_DIR" || exit 1

# Kill existing instances if running (cleanup)
pkill -f "python server.py" 2>/dev/null

echo "Starting SiliconLM..."

# Start Main Server (Foreground - kept alive by launchctl)
# We don't use nohup here because launchctl expects the process to stay alive
echo "Starting Main Server..." >> "$SERVER_LOG"
exec "$VENV_PYTHON" server.py >> "$SERVER_LOG" 2>&1
