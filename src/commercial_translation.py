"""
Commercial Translation: Compliance Risk -> Predictive Customer Engagement (NBA)

Reads compliance-style artifacts produced by:
  - src/evaluate.py (reports/outputs/decision_summary.md, lift_by_decile.csv)
  - src/explain.py (reports/outputs/permutation_importance.csv)

Outputs:
  1) Prints an interview-friendly translation to stdout
  2) Writes a shareable artifact:
       reports/outputs/commercial_summary.md

Run:
    python src/commercial_translation.py --capacity_pct 0.05
"""

import argparse
import os
import pandas as pd

def load_optional_csv(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity_pct", type=float, default=0.05,
                    help="Channel/field capacity fraction (e.g., top 5%)")
    args = ap.parse_args()

    lift_path = "reports/outputs/lift_by_decile.csv"
    imp_path = "reports/outputs/permutation_importance.csv"
    decision_summary_path = "reports/outputs/decision_summary.md"
    out_md_path = "reports/outputs/commercial_summary.md"

    lift = load_optional_csv(lift_path)
    imp = load_optional_csv(imp_path)

    mapping = {
        "transactions_volume": "activity intensity / opportunity volume",
        "policy_change_exposure": "exposure to messaging/strategy shifts",
        "anomaly_score": "unusual pattern signal needing attention",
        "prior_case_flag": "historical context / prior signal",
        "behavior_shift_30d": "recent behavior change (leading indicator)",
        "data_quality_score": "data coverage/quality (governance guardrail)",
        "month": "seasonality / time effects",
    }

    # Compute base + top decile stats if available
    base_rate = None
    top_dec_rate = None
    top_dec_lift = None
    if lift is not None and len(lift) > 0:
        base_rate = float((lift["event_rate"] * lift["n"]).sum() / lift["n"].sum())
        top_dec = lift.sort_values("decile").iloc[0]  # decile 1 = highest score
        top_dec_rate = float(top_dec["event_rate"])
        top_dec_lift = float(top_dec["lift_vs_base"])

    # Top drivers if available
    top_drivers = []
    if imp is not None and len(imp) > 0:
        top = imp.sort_values("importance", ascending=False).head(5)
        for _, r in top.iterrows():
            feat = str(r["feature"])
            top_drivers.append((feat, mapping.get(feat, feat)))

    # ===== Print to stdout =====
    print("\n=== Commercial Translation (what to say in interview) ===\n")
    print("In my prior J&J work, I built a compliance risk prioritization engine.")
    print("The same architecture transfers directly to predictive customer engagement / next-best-action.\n")

    print("1) Units & targets")
    print("   - Compliance 'entity-month'     => Commercial 'customer/HCP-period'")
    print("   - Compliance 'downstream_event' => Commercial 'response / adoption / engagement outcome'\n")

    print("2) Model output & decisioning")
    print("   - Compliance 'risk score'       => Commercial 'propensity/priority score'")
    print("   - Capacity-based review queue   => Channel/field capacity constraint (who to engage next)\n")

    print("3) Operating point (capacity-aware thresholding)")
    print(f"   - Example: prioritize the top {args.capacity_pct:.0%} customers/HCPs based on the score")
    if base_rate is not None:
        print(f"   - Base outcome rate (all):       {base_rate:.4f}")
        print(f"   - Top decile outcome rate:       {top_dec_rate:.4f}")
        print(f"   - Top decile lift vs base:       {top_dec_lift:.2f}x")
    else:
        print("   - (Run: python src/evaluate.py --capacity_pct 0.05)")
    print()

    print("4) What lift means commercially")
    print("   - Higher lift in the top-ranked group means the model concentrates effort on customers most likely to respond.")
    print("   - NBA translation: fewer wasted touches, higher ROI per interaction.\n")

    print("5) Explainability for adoption (drivers in business terms)")
    if top_drivers:
        for feat, desc in top_drivers:
            print(f"   - {feat:22s} -> {desc}")
    else:
        print("   - (Run: python src/explain.py --top_n 6)")
    print()

    print("6) Monitoring & MLOps mindset")
    print("   - Compliance: monitor feature/score drift to keep the prioritization queue stable over time.")
    print("   - Commercial: same monitoring flags behavior/channel shifts (new campaigns, access changes),")
    print("                 triggering recalibration/retraining to protect KPI performance.\n")

    if os.path.exists(decision_summary_path):
        print("7) Stakeholder artifacts")
        print(f"   - Compliance summary:  {decision_summary_path}")
        print(f"   - Commercial summary:  {out_md_path}\n")

    print("Bottom line: Domain changes (risk mitigation -> engagement optimization),")
    print("but the decision-engine pattern (predict -> threshold under constraints -> explain -> monitor) is the same.\n")

    # ===== Write commercial_summary.md =====
    os.makedirs(os.path.dirname(out_md_path), exist_ok=True)
    with open(out_md_path, "w") as f:
        f.write("# Commercial Translation Summary\n\n")
        f.write("This page translates the compliance risk prioritization demo into commercialization / customer engagement language.\n\n")

        f.write("## 1) Mapping\n")
        f.write("- **Entity-month (compliance)** → **Customer/HCP-period (engagement)**\n")
        f.write("- **Downstream risk event** → **Engagement outcome / response / adoption**\n")
        f.write("- **Risk score** → **Propensity / priority score**\n")
        f.write(f"- **Review capacity (top {args.capacity_pct:.0%})** → **Channel/field capacity constraint (NBA)**\n\n")

        f.write("## 2) Decision quality signals (how we know the model is useful)\n")
        f.write("- **Precision–Recall & Average Precision:** performance for imbalanced outcomes\n")
        f.write("- **Lift/enrichment in top-ranked groups:** efficiency of prioritization\n")
        f.write("- **Calibration/Brier score:** probabilities usable for tiering and thresholding\n\n")

        if base_rate is not None:
            f.write("### Example enrichment (from lift_by_decile.csv)\n")
            f.write(f"- Base outcome rate (all customers): **{base_rate:.4f}**\n")
            f.write(f"- Top decile outcome rate: **{top_dec_rate:.4f}**\n")
            f.write(f"- Top decile lift vs base: **{top_dec_lift:.2f}x**\n\n")
        else:
            f.write("> Run `python src/evaluate.py --capacity_pct 0.05` to populate enrichment numbers.\n\n")

        f.write("## 3) Explainability (why this customer/action)\n")
        if top_drivers:
            f.write("Top drivers (translated):\n")
            for feat, desc in top_drivers:
                f.write(f"- `{feat}` → {desc}\n")
            f.write("\n")
        else:
            f.write("> Run `python src/explain.py --top_n 6` to populate top drivers.\n\n")

        f.write("## 4) Monitoring & sustainability (MLOps mindset)\n")
        f.write("- Monitor **feature drift** and **score drift** to detect behavior/channel shifts\n")
        f.write("- Trigger recalibration/retraining when drift exceeds thresholds\n\n")

        f.write("## 5) Interview positioning\n")
        f.write("In compliance, the system prioritizes who to review next under capacity and governance; ")
        f.write("in commercialization, the same decision engine prioritizes who to engage next and what action to take under channel constraints.\n")

    print(f"Wrote {out_md_path}")

if __name__ == "__main__":
    main()
