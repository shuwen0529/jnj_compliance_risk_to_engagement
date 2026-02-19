#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python src/make_data.py
python src/train.py
python src/evaluate.py --capacity_pct 0.05
python src/explain.py --top_n 6
python src/drift_check.py

echo "Done. Open reports/outputs/decision_summary.md and reports/figures/*"
