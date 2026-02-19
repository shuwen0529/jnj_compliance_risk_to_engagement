# Step-by-step walkthrough (Compliance -> Engagement transfer)

## 1) Data (Compliance framing)
Each row is an **entity-month** observation with multi-source signals:
- transactions_volume: activity intensity
- policy_change_exposure: exposure to policy/process shifts
- anomaly_score: automated anomaly signal
- prior_case_flag: historical case involvement
- behavior_shift_30d: change in behavior vs recent baseline
- data_quality_score: quality/coverage proxy

Target:
- downstream_event (1/0): synthetic "downstream compliance event" label (imbalanced)

## 2) Modeling (Risk scoring + operating point)
We train a Gradient Boosting model (tree ensemble) and evaluate with:
- PR curve (better for imbalanced outcomes than ROC)
- Lift / enrichment at top-k (mirrors review queue)
- Calibration (probabilities usable for tiering)

We then pick a threshold based on **capacity_pct** (e.g., top 5% reviewed).

## 3) Explainability (Audit-friendly)
- Permutation importance: model-agnostic global drivers
- Partial dependence: directionality for top drivers

## 4) Drift monitoring (MLOps mindset)
- PSI-style drift per feature between baseline vs recent window
- Score distribution drift check

## 5) Transfer to Customer Engagement
Replace compliance terms with commercialization terms:
- risk score -> propensity / uplift score
- review queue -> NBA prioritized action list under channel capacity
- false positives -> wasted touches
- false negatives -> missed engagement opportunities
- explainability -> rep/brand trust and adoption
