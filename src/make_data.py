import numpy as np
import pandas as pd

def sigmoid(x):
    return 1/(1+np.exp(-x))

def main(out_path="data/raw/synthetic_compliance_signals.csv", n_entities=1200, months=12, seed=42):
    rng = np.random.default_rng(seed)

    entity_ids = np.arange(1, n_entities+1)
    month_ids = np.arange(1, months+1)

    rows = []
    for e in entity_ids:
        baseline_risk = rng.normal(0, 0.8) + rng.binomial(1, 0.12)*1.5
        prior_case_flag = rng.binomial(1, 0.10 + 0.05*sigmoid(baseline_risk))

        for m in month_ids:
            transactions_volume = rng.lognormal(mean=2.2 + 0.25*baseline_risk, sigma=0.6)
            policy_change_exposure = rng.binomial(1, 0.10 + 0.15*(m in [4,8,12]))
            anomaly_score = np.clip(rng.normal(0.6 + 0.35*baseline_risk + 0.6*policy_change_exposure, 0.7), 0, 4)
            behavior_shift_30d = rng.normal(0.0 + 0.45*baseline_risk + 0.35*policy_change_exposure, 0.9)
            data_quality_score = np.clip(rng.normal(0.85 - 0.08*policy_change_exposure, 0.08), 0.4, 0.98)

            logit = (
                -4.2
                + 0.35*np.log1p(transactions_volume)
                + 0.85*policy_change_exposure
                + 0.70*anomaly_score
                + 1.05*prior_case_flag
                + 0.60*behavior_shift_30d
                + 0.9*baseline_risk
                - 1.25*(1-data_quality_score)
                + rng.normal(0, 0.35)
            )
            p = sigmoid(logit)
            downstream_event = rng.binomial(1, p)

            rows.append({
                "entity_id": int(e),
                "month": int(m),
                "transactions_volume": float(transactions_volume),
                "policy_change_exposure": int(policy_change_exposure),
                "anomaly_score": float(anomaly_score),
                "prior_case_flag": int(prior_case_flag),
                "behavior_shift_30d": float(behavior_shift_30d),
                "data_quality_score": float(data_quality_score),
                "downstream_event": int(downstream_event),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} | rows={len(df):,} | event_rate={df['downstream_event'].mean():.4f}")

if __name__ == "__main__":
    main()
