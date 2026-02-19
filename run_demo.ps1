python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python src/make_data.py
python src/train.py
python src/evaluate.py --capacity_pct 0.05
python src/explain.py --top_n 6
python src/drift_check.py

Write-Host "Done. Open reports/outputs/decision_summary.md and reports/figures/*"
