# J&J Compliance Risk Prioritization (Transferable to Predictive Customer Engagement)

This repo is intentionally framed as **Compliance & Risk Management** (your J&J experience),
but the modeling pattern maps directly to **Next-Best-Action / Customer Engagement**.

What it demonstrates (end-to-end):
1) Generate synthetic, audit-friendly compliance signals (entity-month level)
2) Train a risk scoring model with calibrated thresholds (capacity-based review queue)
3) Evaluate decision performance (PR, top-k lift, calibration, operating point)
4) Explain drivers (permutation importance + partial dependence)
5) Monitor drift (PSI-style checks) and score stability

Quickstart:
    pip install -r requirements.txt
    python src/make_data.py
    python src/train.py
    python src/evaluate.py --capacity_pct 0.05
    python src/explain.py --top_n 6
    python src/drift_check.py

Artifacts:
- data/raw/synthetic_compliance_signals.csv
- models/risk_model.joblib
- reports/figures/*.png
- reports/outputs/decision_summary.md

Transfer mapping (how to explain in interview):
- "Entity" (compliance) == "Customer/HCP" (engagement)
- "Downstream risk event" == "Desired outcome / response" (engagement/adoption)
- "Review capacity" == "Field force / channel capacity" (NBA constraints)
- "Risk tier + rationale" == "Next best action + drivers" (explainability for adoption)

Additional (commercial translation):
    python src/commercial_translation.py --capacity_pct 0.05

After running commercial translation:
- reports/outputs/commercial_summary.md
