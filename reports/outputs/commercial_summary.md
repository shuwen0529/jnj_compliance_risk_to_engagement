# Commercial Translation Summary

This page translates the compliance risk prioritization demo into commercialization / customer engagement language.

## 1) Mapping
- **Entity-month (compliance)** → **Customer/HCP-period (engagement)**
- **Downstream risk event** → **Engagement outcome / response / adoption**
- **Risk score** → **Propensity / priority score**
- **Review capacity (top 5%)** → **Channel/field capacity constraint (NBA)**

## 2) Decision quality signals (how we know the model is useful)
- **Precision–Recall & Average Precision:** performance for imbalanced outcomes
- **Lift/enrichment in top-ranked groups:** efficiency of prioritization
- **Calibration/Brier score:** probabilities usable for tiering and thresholding

### Example enrichment (from lift_by_decile.csv)
- Base outcome rate (all customers): **0.1619**
- Top decile outcome rate: **0.5472**
- Top decile lift vs base: **3.38x**

## 3) Explainability (why this customer/action)
Top drivers (translated):
- `behavior_shift_30d` → recent behavior change (leading indicator)
- `anomaly_score` → unusual pattern signal needing attention
- `transactions_volume` → activity intensity / opportunity volume
- `prior_case_flag` → historical context / prior signal
- `policy_change_exposure` → exposure to messaging/strategy shifts

## 4) Monitoring & sustainability (MLOps mindset)
- Monitor **feature drift** and **score drift** to detect behavior/channel shifts
- Trigger recalibration/retraining when drift exceeds thresholds

## 5) Interview positioning
In compliance, the system prioritizes who to review next under capacity and governance; in commercialization, the same decision engine prioritizes who to engage next and what action to take under channel constraints.
