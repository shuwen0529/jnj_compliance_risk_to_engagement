import argparse
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/synthetic_compliance_signals.csv")
    ap.add_argument("--model", default="models/risk_model.joblib")
    ap.add_argument("--top_n", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    artifact = joblib.load(args.model)
    model = artifact["model"]
    feats = artifact["features"]

    test_df = df[df["month"] >= 10].copy()
    X = test_df[feats]
    y = test_df["downstream_event"].values

    r = permutation_importance(model, X, y, n_repeats=10, random_state=42, scoring="average_precision")
    imp = pd.DataFrame({"feature": feats, "importance": r.importances_mean}).sort_values("importance", ascending=False)
    imp.to_csv("reports/outputs/permutation_importance.csv", index=False)

    top_feats = imp["feature"].head(args.top_n).tolist()

    plt.figure(figsize=(8, 4))
    plt.barh(imp["feature"].head(args.top_n)[::-1], imp["importance"].head(args.top_n)[::-1])
    plt.xlabel("Permutation importance (ΔAP)")
    plt.title("Top drivers (audit-friendly, model-agnostic)")
    plt.tight_layout()
    plt.savefig("reports/figures/permutation_importance.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(model, X, top_feats, ax=ax)
    plt.tight_layout()
    plt.savefig("reports/figures/partial_dependence.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved permutation importance + partial dependence plots.")

if __name__ == "__main__":
    main()
