import numpy as np
import pandas as pd
import joblib

def psi(a, b, bins=10):
    edges = np.quantile(a, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    a_hist, _ = np.histogram(a, bins=edges)
    b_hist, _ = np.histogram(b, bins=edges)
    a_p = (a_hist + 1e-6) / (a_hist.sum() + 1e-6)
    b_p = (b_hist + 1e-6) / (b_hist.sum() + 1e-6)
    return float(np.sum((b_p - a_p) * np.log(b_p / a_p)))

def main(data_path="data/raw/synthetic_compliance_signals.csv", model_path="models/risk_model.joblib"):
    df = pd.read_csv(data_path)
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feats = artifact["features"]

    base = df[df["month"] <= 9].copy()
    recent = df[df["month"] >= 10].copy()

    print("=== Feature drift (PSI) baseline months1-9 vs recent months10-12 ===")
    for c in ["transactions_volume", "anomaly_score", "behavior_shift_30d", "data_quality_score"]:
        print(f"{c:22s} PSI={psi(base[c].values, recent[c].values):.4f}")

    base_scores = model.predict_proba(base[feats])[:, 1]
    recent_scores = model.predict_proba(recent[feats])[:, 1]
    print("\n=== Score distribution drift (PSI) ===")
    print(f"risk_score               PSI={psi(base_scores, recent_scores):.4f}")

if __name__ == "__main__":
    main()
