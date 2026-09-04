#!/bin/bash
set -e

# Change directory to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure venv python is used
if [ -d ".venv" ]; then
    export PATH="$SCRIPT_DIR/.venv/bin:$PATH"
fi

echo "================================================================="
echo "🦠 啟動 Google TimesFM 3.0 vs. Meta Prophet 傳染病預測系統"
echo "================================================================="
echo "正在啟動 Streamlit Web 介面 (預設連接埠: 8501)..."

streamlit run app.py --server.port 8501 --server.headless true --server.address 0.0.0.0
