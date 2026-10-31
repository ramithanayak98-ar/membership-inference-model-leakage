#!/usr/bin/env bash
set -euo pipefail

echo "================================================================================"
echo "  Reproducible Execution Pipeline: Membership Inference & Model Leakage"
echo "  Team: Ramitha Nayak MV (PES1UG24AM392) & Shreya N Hegde (PES1UG24AM401)"
echo "================================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if command -v /opt/homebrew/bin/python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="/opt/homebrew/bin/python3.12"
else
    PYTHON_BIN="python3"
fi

echo "[INFO] Running experimental pipeline using $PYTHON_BIN..."
"$PYTHON_BIN" run_pipeline.py

echo "[INFO] Verifying generated artifacts..."
ls -lh data/ results/ results/figures/

echo "[SUCCESS] All attacks, defense evaluations, tables, and figures reproduced."
