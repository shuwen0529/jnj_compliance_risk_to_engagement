import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

FEATURES = [
    "transactions_volume",
    "policy_change_exposure",
    "anomaly_score",
    "prior_case_flag",
    "behavior_shift_30d",
    "data_quality_score",
    "month",
]

def main(data_path="data/raw/synthetic_compliance_signals.csv", model_out="models/risk_model.joblib", seed=42):
    df = pd.read_csv(data_path)

    # Out-of-time split: train months 1-9, test months 10-12
    train_df = df[df["month"] <= 9].copy()
    test_df  = df[df["month"] >= 10].copy()

    X_train, y_train = train_df[FEATURES], train_df["downstream_event"]
    X_test,  y_test  = test_df[FEATURES],  test_df["downstream_event"]

    model = GradientBoostingClassifier(random_state=seed)
    model.fit(X_train, y_train)

    test_scores = model.predict_proba(X_test)[:, 1]
    print("Out-of-time ROC-AUC:", round(roc_auc_score(y_test, test_scores), 4))
    print("Out-of-time Avg Precision (AP):", round(average_precision_score(y_test, test_scores), 4))

    joblib.dump({"model": model, "features": FEATURES}, model_out)
    print(f"Saved model artifact: {model_out}")

if __name__ == "__main__":
    main()
