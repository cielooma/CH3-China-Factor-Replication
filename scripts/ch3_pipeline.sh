#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -x .venv/bin/python ]]; then PYTHON="${PYTHON:-.venv/bin/python}"; else PYTHON="${PYTHON:-python3}"; fi
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
case "${1:-smoke-v2}" in
 data) "$PYTHON" scripts/generate_synthetic_ch3_panel.py ;;
 smoke-v2)
  [[ -f synthetic_data/ch3_monthly_market.csv ]] || "$PYTHON" scripts/generate_synthetic_ch3_panel.py
  "$PYTHON" scripts/run_ch3_reproduction.py --config configs/ch3_synthetic_v2.json --require-alignment --no-registry ;;
 test) "$PYTHON" -m pytest tests/ ;;
 reference) "$PYTHON" scripts/run_ch3_reproduction.py --config configs/ch3_reference_v2.json --no-registry ;;
 *) echo "Usage: bash scripts/ch3_pipeline.sh {data|smoke-v2|test|reference}" >&2; exit 2 ;;
esac
