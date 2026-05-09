"""
Audit whether a limited transaction-based ranking evaluation is usable.

The script is conservative: it writes an audit CSV always, but only writes NDCG
results when enough destination groups have sufficient matched transaction data.
This avoids reporting a weak downstream metric as if it were definitive.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


ROOT = Path(__file__).parents[2]
RESULT = ROOT / "result" / "analysis"
RESULT.mkdir(parents=True, exist_ok=True)

ACCOM = ROOT / "dataset" / "preprocessed" / "accommodations.parquet"
TX = ROOT / "dataset" / "preprocessed" / "transactions.parquet"
QUALITY = ROOT / "result" / "c2_validation" / "transaction_quality_scores.csv"

MIN_GROUP_SIZE = 10
MIN_BOOKED_PER_GROUP = 3
MIN_GROUPS = 20
K_VALUES = [5, 10]


def load_inputs():
    accom_cols = ["id", "provider", "destination_display_name", "star_rating", "guest_rating"]
    accom = pd.read_parquet(ACCOM, columns=accom_cols)
    accom["property_id"] = accom["id"].astype(str)

    tx = pd.read_parquet(TX)
    tx_agg = (
        tx.groupby("property_id", as_index=False)
        .agg(total_transactions=("transaction_count", "sum"), months_active=("year_month", "nunique"))
    )
    tx_agg["property_id"] = tx_agg["property_id"].astype(str)

    quality = pd.read_csv(QUALITY)
    quality["property_id"] = quality["property_id"].astype(str)
    return accom, tx_agg, quality


def quantile_calibrate_star(df):
    out = df.copy()
    pooled = out["star_rating"].dropna().to_numpy()
    out["star_calibrated"] = out["star_rating"]
    if len(pooled) == 0:
        return out
    pooled_sorted = np.sort(pooled)
    for provider, idx in out.groupby("provider").groups.items():
        vals = out.loc[idx, "star_rating"].to_numpy()
        valid = ~np.isnan(vals)
        if valid.sum() < 2:
            continue
        provider_sorted = np.sort(vals[valid])
        ranks = np.searchsorted(provider_sorted, vals[valid], side="right") / len(provider_sorted)
        mapped = np.quantile(pooled_sorted, ranks.clip(0, 1), method="nearest")
        calibrated = vals.copy()
        calibrated[valid] = mapped
        out.loc[idx, "star_calibrated"] = calibrated
    return out


def make_ranking_frame(accom, tx_agg, quality):
    df = accom.merge(tx_agg, on="property_id", how="inner")
    df = df.merge(quality[["property_id", "quality_score"]], on="property_id", how="left")
    df = quantile_calibrate_star(df)
    df["relevance"] = np.log1p(df["total_transactions"].astype(float))
    df["raw_star_score"] = df["star_rating"].fillna(-1)
    df["calibrated_star_score"] = df["star_calibrated"].fillna(-1)
    return df


def audit_groups(df):
    rows = []
    for dest, group in df.groupby("destination_display_name", dropna=False):
        rows.append({
            "destination": dest,
            "n_properties": len(group),
            "n_booked": int((group["total_transactions"] > 0).sum()),
            "total_transactions": int(group["total_transactions"].sum()),
        })
    audit = pd.DataFrame(rows).sort_values(["n_properties", "total_transactions"], ascending=False)
    audit["usable"] = (audit["n_properties"] >= MIN_GROUP_SIZE) & (audit["n_booked"] >= MIN_BOOKED_PER_GROUP)
    return audit


def compute_ndcg(df, usable_destinations):
    score_cols = {
        "raw_star": "raw_star_score",
        "calibrated_star": "calibrated_star_score",
        "embedding_quality": "quality_score",
    }
    rows = []
    for dest in usable_destinations:
        group = df[df["destination_display_name"].eq(dest)].copy()
        y_true = group["relevance"].to_numpy().reshape(1, -1)
        if np.all(y_true == y_true[0, 0]):
            continue
        for method, col in score_cols.items():
            if group[col].isna().all():
                continue
            scores = group[col].fillna(group[col].min() - 1).to_numpy().reshape(1, -1)
            for k in K_VALUES:
                rows.append({
                    "destination": dest,
                    "method": method,
                    "k": k,
                    "ndcg": ndcg_score(y_true, scores, k=min(k, group.shape[0])),
                    "n_properties": group.shape[0],
                    "n_booked": int((group["total_transactions"] > 0).sum()),
                })
    return pd.DataFrame(rows)


def summarize_ndcg(ndcg):
    if ndcg.empty:
        return pd.DataFrame()
    return (
        ndcg.groupby(["method", "k"], as_index=False)
        .agg(mean_ndcg=("ndcg", "mean"), median_ndcg=("ndcg", "median"), n_groups=("destination", "nunique"))
        .sort_values(["k", "mean_ndcg"], ascending=[True, False])
    )


def main():
    print("Loading inputs...", flush=True)
    accom, tx_agg, quality = load_inputs()
    df = make_ranking_frame(accom, tx_agg, quality)
    audit = audit_groups(df)
    audit.to_csv(RESULT / "ranking_audit.csv", index=False)

    usable = audit[audit["usable"]]["destination"].tolist()
    summary_rows = [{
        "matched_properties": len(df),
        "unique_destinations": audit.shape[0],
        "usable_destinations": len(usable),
        "min_group_size": MIN_GROUP_SIZE,
        "min_booked_per_group": MIN_BOOKED_PER_GROUP,
        "report_ndcg": len(usable) >= MIN_GROUPS,
    }]
    pd.DataFrame(summary_rows).to_csv(RESULT / "ranking_audit_summary.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)

    if len(usable) < MIN_GROUPS:
        print(f"Not enough usable groups for NDCG: {len(usable)} < {MIN_GROUPS}. Skipping ranking_ndcg.csv.", flush=True)
        return

    ndcg = compute_ndcg(df, usable)
    ndcg.to_csv(RESULT / "ranking_ndcg.csv", index=False)
    summary = summarize_ndcg(ndcg)
    summary.to_csv(RESULT / "ranking_ndcg_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
