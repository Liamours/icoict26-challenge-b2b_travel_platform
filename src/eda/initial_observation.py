import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

RAW = Path(__file__).parents[2] / "dataset" / "raw"
OUT = Path(__file__).parents[2] / "result" / "eda" / "initial_observation"
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 200_000

# ---------------------------------------------------------------------------
# Expected schemas — (dtype_category, nullable, valid_range or valid_set)
# ---------------------------------------------------------------------------
TRANS_RULES = {
    "payment_date":          ("datetime", False, ("2024-01-01", "2027-01-01")),
    "Property Id":           ("str",      False, None),
    "Product":               ("str",      False, None),
    "Number Transactions":   ("int",      False, (1, 10_000)),
}

ACCOM_RULES = {
    "id":                    ("str",      False, None),
    "popularity":            ("float",    True,  (0.0, 100.0)),
    "popularity_score":      ("float",    False, (0.0, 200.0)),
    "star_rating":           ("float",    True,  (0.0, 5.0)),
    "star_rating_score":     ("float",    False, (1.0, 7.0)),
    "guest_rating":          ("float",    True,  (0.0, 10.0)),
    "guest_rating_count":    ("float",    True,  (0.0, 1_000_000.0)),
    "availability_score":    ("int",      False, (0, 200)),
    "reference_price":       ("float",    False, (0.0, 1_000_000.0)),
    "price_in_aud":          ("float",    False, (0.0, 1_000_000.0)),
    "destination_coordinate": ("coord",   False, None),
}

ACTIVITY_RULES = {
    "rating":                ("float",    False, (0.0, 10.0)),
    "normalized_rating":     ("numeric",  False, (0.0, 10.0)),
    "duration_hours":        ("int",      False, (0, 720)),
    "price":                 ("float",    False, (0.0, 10_000_000.0)),
    "price_in_aud":          ("float",    False, (0.0, 10_000_000.0)),
    "availability_score":    ("int",      False, (0, 200)),
    "destination_coordinate": ("coord",   True,  None),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_coord(series: pd.Series) -> pd.Series:
    """Returns boolean mask: True = invalid coordinate string."""
    def bad(v):
        if pd.isna(v):
            return False
        try:
            parts = str(v).split(",")
            if len(parts) != 2:
                return True
            lat, lon = float(parts[0]), float(parts[1])
            return not (-90 <= lat <= 90 and -180 <= lon <= 180)
        except Exception:
            return True
    return series.apply(bad)


def report_column(name: str, series: pd.Series, rule: tuple, report: list):
    dtype_cat, nullable, valid_range = rule
    issues = {}

    null_count = series.isna().sum()
    issues["null_count"] = int(null_count)
    issues["null_pct"] = round(null_count / len(series) * 100, 2)

    if not nullable and null_count > 0:
        issues["unexpected_nulls"] = True

    if dtype_cat == "datetime":
        try:
            parsed = pd.to_datetime(series, errors="coerce")
            unparseable = parsed.isna().sum() - null_count
            issues["unparseable_dates"] = int(max(unparseable, 0))
            if valid_range:
                lo, hi = pd.Timestamp(valid_range[0]), pd.Timestamp(valid_range[1])
                out_of_range = ((parsed < lo) | (parsed > hi)).sum()
                issues["out_of_range"] = int(out_of_range)
        except Exception as e:
            issues["parse_error"] = str(e)

    elif dtype_cat in ("float", "int", "numeric"):
        numeric = pd.to_numeric(series, errors="coerce")
        non_numeric = numeric.isna().sum() - null_count
        issues["non_numeric"] = int(max(non_numeric, 0))
        if valid_range and not numeric.dropna().empty:
            lo, hi = valid_range
            mask = numeric.dropna()
            below = int((mask < lo).sum())
            above = int((mask > hi).sum())
            issues["below_min"] = below
            issues["above_max"] = above
            issues["out_of_range"] = below + above
            if not numeric.dropna().empty:
                issues["actual_min"] = round(float(numeric.min()), 4)
                issues["actual_max"] = round(float(numeric.max()), 4)

    elif dtype_cat == "coord":
        bad_mask = check_coord(series)
        issues["invalid_coords"] = int(bad_mask.sum())

    report.append({"column": name, **issues})


def summarize_report(report: list, dataset_name: str) -> pd.DataFrame:
    df = pd.DataFrame(report)
    print(f"\n{'='*60}")
    print(f"  {dataset_name}")
    print(f"{'='*60}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Dataset checks
# ---------------------------------------------------------------------------

def check_transactions() -> pd.DataFrame:
    df = pd.read_excel(RAW / "Accommodation - Transactions by HotelCode.xlsx", engine="openpyxl")
    df.columns = df.columns.str.strip()

    report = []
    for col, rule in TRANS_RULES.items():
        if col in df.columns:
            report_column(col, df[col], rule, report)

    # Extra checks
    known_providers = {"HotelBeds", "Expedia", "Agoda", "RateHawk"}
    df["_provider"] = df["Property Id"].str.split("-").str[0]
    unknown_providers = ~df["_provider"].isin(known_providers)
    print(f"\n[Transactions] Unknown providers: {unknown_providers.sum()} rows → {df[unknown_providers]['_provider'].unique()}")

    dup_rows = df.duplicated(subset=["payment_date", "Property Id", "Product"]).sum()
    print(f"[Transactions] Duplicate (date, property, product): {dup_rows}")

    return summarize_report(report, "TRANSACTIONS")


def check_accommodations() -> pd.DataFrame:
    report_acc = []
    first = True
    extra_stats = {col: {"below_min": 0, "above_max": 0, "non_numeric": 0, "null_count": 0} for col in ACCOM_RULES}

    chunks = list(tqdm(
        pd.read_csv(RAW / "accommodations.csv", chunksize=CHUNK, low_memory=False),
        desc="Scanning accommodations"
    ))
    df_full = pd.concat(chunks, ignore_index=True)
    df_full["provider"] = df_full["id"].str.split("-").str[0]

    report = []
    for col, rule in tqdm(ACCOM_RULES.items(), desc="Checking columns"):
        if col in df_full.columns:
            report_column(col, df_full[col], rule, report)

    # Extra checks
    known_providers = {"HotelBeds", "Expedia", "Agoda", "RateHawk"}
    unknown = ~df_full["provider"].isin(known_providers)
    print(f"\n[Accommodations] Unknown providers: {unknown.sum()} rows → {df_full[unknown]['provider'].value_counts().head(5).to_dict()}")

    dup_ids = df_full["id"].duplicated().sum()
    print(f"[Accommodations] Duplicate IDs: {dup_ids}")

    neg_star = (df_full["star_rating"] < 0).sum()
    print(f"[Accommodations] Negative star_rating: {neg_star}")

    neg_avail = (df_full["availability_score"] < 0).sum()
    print(f"[Accommodations] Negative availability_score: {neg_avail}")

    zero_price_pct = (df_full["reference_price"] == 0).sum() / len(df_full) * 100
    print(f"[Accommodations] Zero reference_price: {zero_price_pct:.1f}%")

    return summarize_report(report, "ACCOMMODATIONS")


def check_activities() -> pd.DataFrame:
    chunks = []
    for chunk in tqdm(
        pd.read_csv(RAW / "activity.csv", chunksize=CHUNK, low_memory=False),
        desc="Loading activities"
    ):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    report = []
    for col, rule in tqdm(ACTIVITY_RULES.items(), desc="Checking columns"):
        if col in df.columns:
            report_column(col, df[col], rule, report)

    # Extra checks
    norm = pd.to_numeric(df["normalized_rating"], errors="coerce")
    non_numeric_norm = norm.isna().sum() - df["normalized_rating"].isna().sum()
    print(f"\n[Activities] normalized_rating non-numeric values: {non_numeric_norm}")
    if non_numeric_norm > 0:
        bad_vals = df["normalized_rating"][norm.isna() & df["normalized_rating"].notna()]
        print(f"  Sample bad values: {bad_vals.unique()[:10]}")

    dur_zero = (df["duration_hours"] == 0).sum()
    print(f"[Activities] Zero duration_hours: {dur_zero}")

    rating_gt10 = (df["rating"] > 10).sum()
    print(f"[Activities] rating > 10 (likely invalid): {rating_gt10}")

    zero_price = (df["price_in_aud"] == 0).sum()
    print(f"[Activities] Zero price_in_aud: {zero_price} ({zero_price/len(df)*100:.1f}%)")

    has_bali_dtype = df["has_listing_id_in_bali_db"].dtype
    print(f"[Activities] has_listing_id_in_bali_db dtype: {has_bali_dtype} (should be bool)")
    print(f"  Value counts: {df['has_listing_id_in_bali_db'].value_counts(dropna=False).to_dict()}")

    dup_products = df["product_id"].duplicated().sum()
    print(f"[Activities] Duplicate product_id: {dup_products}")

    return summarize_report(report, "ACTIVITIES")


# ---------------------------------------------------------------------------
# Temporal / time-series validation
# ---------------------------------------------------------------------------

def check_temporal_transactions(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("  TEMPORAL INTEGRITY — TRANSACTIONS")
    print(f"{'='*60}")

    df = df.copy()
    df["payment_date"] = pd.to_datetime(df["payment_date"], utc=True)
    now_utc = pd.Timestamp.now(tz="UTC")

    # 1. Future dates in historical log
    future = (df["payment_date"] > now_utc).sum()
    print(f"[Chrono] Future dates: {future}")

    # 2. Timezone normalization — confirm all normalized to UTC
    tz_set = df["payment_date"].dt.tz
    print(f"[Timezone] All timestamps tz: {tz_set} (UTC={tz_set == 'UTC'})")

    # 3. Out-of-order timestamps (overall sort check)
    sorted_dates = df["payment_date"].sort_values()
    out_of_order = (df["payment_date"].values != sorted_dates.values).sum()
    print(f"[Chrono] Out-of-order rows (global): {out_of_order}")

    # 4. Expected monthly range
    min_date = df["payment_date"].dt.to_period("M").min()
    max_date = df["payment_date"].dt.to_period("M").max()
    expected_months = pd.period_range(min_date, max_date, freq="M")
    actual_months = df["payment_date"].dt.to_period("M").unique()
    missing_months = set(expected_months) - set(actual_months)
    print(f"[Gaps] Expected months: {len(expected_months)}, Actual: {len(actual_months)}, Missing: {sorted(missing_months)}")

    # 5. Per-provider temporal gap detection
    print("\n[Gaps] Provider-level monthly coverage:")
    df["ym"] = df["payment_date"].dt.to_period("M")
    for provider, grp in df.groupby(df["Property Id"].str.split("-").str[0]):
        provider_months = set(grp["ym"].unique())
        missing = set(expected_months) - provider_months
        print(f"  {provider}: {len(provider_months)}/13 months present, missing: {sorted(missing) or 'none'}")

    # 6. Logical shifts — month-over-month transaction count jumps per provider
    print("\n[Logical Shifts] Provider month-over-month transaction delta:")
    provider_monthly = (
        df.assign(provider=df["Property Id"].str.split("-").str[0])
        .groupby(["provider", "ym"])["Number Transactions"]
        .sum()
        .reset_index()
        .sort_values(["provider", "ym"])
    )
    provider_monthly["pct_change"] = provider_monthly.groupby("provider")["Number Transactions"].pct_change() * 100
    extreme_jumps = provider_monthly[provider_monthly["pct_change"].abs() > 100]
    print(extreme_jumps[["provider", "ym", "Number Transactions", "pct_change"]].to_string(index=False))

    # 7. Zero-value skew — months with zero total transactions per provider
    print("\n[Zero Skew] Months with zero transactions per provider:")
    for provider, grp in provider_monthly.groupby("provider"):
        zero_months = grp[grp["Number Transactions"] == 0]["ym"].tolist()
        print(f"  {provider}: {zero_months or 'none'}")

    # 8. Stationarity check — flat-lining (constant value ≥4 consecutive months)
    print("\n[Stationarity] Consecutive months with same transaction count per provider:")
    for provider, grp in provider_monthly.groupby("provider"):
        vals = grp["Number Transactions"].tolist()
        max_run = max_consecutive_equal(vals)
        print(f"  {provider}: max consecutive same-value run = {max_run}")

    # 9. Aggregation drift — verify sum(Number Transactions per row) == sum(monthly aggregate)
    row_sum = df["Number Transactions"].sum()
    monthly_sum = provider_monthly["Number Transactions"].sum()
    print(f"\n[Aggregation Drift] Row-level sum: {row_sum}, Provider-monthly sum: {monthly_sum}, Match: {row_sum == monthly_sum}")


def check_temporal_accommodations(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("  TEMPORAL INTEGRITY — ACCOMMODATIONS")
    print(f"{'='*60}")

    # last_updated_date and indexed_date
    for col in ["last_updated_date", "indexed_date"]:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        now_utc = pd.Timestamp.now(tz="UTC")
        future = (parsed > now_utc).sum()
        null = parsed.isna().sum()
        print(f"[{col}] Future dates: {future}, Unparseable: {null}")

    # availability_score: flag negative or >200 as constraint violation
    avail = df["availability_score"]
    neg = (avail < 0).sum()
    over = (avail > 200).sum()
    print(f"\n[Constraint] availability_score < 0: {neg}, > 200: {over}")
    if neg > 0:
        print(f"  Sample negative values: {avail[avail < 0].head(5).tolist()}")

    # Zero-value skew: availability_score == 0
    zero_avail = (avail == 0).sum()
    print(f"[Zero Skew] availability_score == 0: {zero_avail} ({zero_avail/len(df)*100:.2f}%)")

    # Zero-value skew: reference_price == 0 streaks (sorted by provider + id)
    zero_price_pct = (df["reference_price"] == 0).sum() / len(df) * 100
    print(f"[Zero Skew] reference_price == 0: {zero_price_pct:.1f}%")
    print("[Zero Skew] Zero price by provider:")
    df_tmp = df.copy()
    df_tmp["provider"] = df_tmp["id"].str.split("-").str[0]
    print(df_tmp.groupby("provider").apply(
        lambda g: f"{(g['reference_price'] == 0).sum()/len(g)*100:.1f}%"
    ).to_string())

    # ID consistency: duplicate IDs (same property listed twice)
    dup = df["id"].duplicated()
    print(f"\n[ID Consistency] Duplicate property IDs: {dup.sum()}")
    if dup.sum() > 0:
        dup_ids = df[dup]["id"].head(5).tolist()
        print(f"  Sample: {dup_ids}")


def check_temporal_activities(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("  TEMPORAL INTEGRITY — ACTIVITIES")
    print(f"{'='*60}")

    for col in ["last_updated_date", "indexed_date"]:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        now_utc = pd.Timestamp.now(tz="UTC")
        future = (parsed > now_utc).sum()
        null = parsed.isna().sum()
        oldest = parsed.dropna().min()
        newest = parsed.dropna().max()
        print(f"[{col}] Future: {future}, Unparseable: {null}, Range: {oldest} → {newest}")

    # Logical shift: extreme price outliers as "jump" signal (no true sequence, but flag extremes)
    price = pd.to_numeric(df["price_in_aud"], errors="coerce").dropna()
    price_sorted = price.sort_values().reset_index(drop=True)
    jumps = price_sorted.pct_change().abs()
    extreme = (jumps > 10).sum()
    print(f"\n[Logical Shifts] price_in_aud jumps >1000% between sorted values: {extreme}")

    # Zero-value skew: streaks of 0 in availability_score
    avail = df["availability_score"]
    zero_avail = (avail == 0).sum()
    print(f"[Zero Skew] availability_score == 0: {zero_avail} ({zero_avail/len(df)*100:.2f}%)")

    # Zero price streak
    zero_price = (df["price_in_aud"] == 0).sum()
    print(f"[Zero Skew] price_in_aud == 0: {zero_price} ({zero_price/len(df)*100:.2f}%)")

    # Stationarity: availability_score constant check
    avail_unique = avail.nunique()
    print(f"[Stationarity] availability_score unique values: {avail_unique} (=1 means flat-lined)")
    print(f"  Value distribution: {avail.value_counts().head(5).to_dict()}")

    # ID consistency: duplicate product_ids
    dup = df["product_id"].duplicated().sum()
    print(f"\n[ID Consistency] Duplicate product_id: {dup}")


def max_consecutive_equal(vals: list) -> int:
    if not vals:
        return 0
    max_run, run = 1, 1
    for i in range(1, len(vals)):
        run = run + 1 if vals[i] == vals[i - 1] else 1
        max_run = max(max_run, run)
    return max_run


# ---------------------------------------------------------------------------
# Cross-dataset consistency
# ---------------------------------------------------------------------------

def check_cross_dataset():
    print(f"\n{'='*60}")
    print("  CROSS-DATASET CONSISTENCY")
    print(f"{'='*60}")

    trans = pd.read_excel(RAW / "Accommodation - Transactions by HotelCode.xlsx", engine="openpyxl")
    trans.columns = trans.columns.str.strip()
    trans_ids = set(trans["Property Id"].unique())

    accom_ids = set()
    for chunk in pd.read_csv(RAW / "accommodations.csv", chunksize=CHUNK, low_memory=False,
                              usecols=["id"]):
        accom_ids.update(chunk["id"].unique())

    overlap = trans_ids & accom_ids
    print(f"Transaction property IDs: {len(trans_ids):,}")
    print(f"Accommodation IDs: {len(accom_ids):,}")
    print(f"Joinable (overlap): {len(overlap):,} ({len(overlap)/len(trans_ids)*100:.1f}% of transactions)")
    print(f"Transaction IDs NOT in accommodations: {len(trans_ids - accom_ids):,}")

    no_match = trans[~trans["Property Id"].isin(accom_ids)]
    print(f"\nProvider breakdown of unmatched transactions:")
    no_match["_prov"] = no_match["Property Id"].str.split("-").str[0]
    print(no_match["_prov"].value_counts().to_dict())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    reports = {}

    print("\nChecking transactions...")
    reports["transactions"] = check_transactions()

    print("\nChecking accommodations...")
    reports["accommodations"] = check_accommodations()

    print("\nChecking activities...")
    reports["activities"] = check_activities()

    # --- Temporal checks ---
    print("\nTemporal checks — transactions...")
    trans_df = pd.read_excel(RAW / "Accommodation - Transactions by HotelCode.xlsx", engine="openpyxl")
    trans_df.columns = trans_df.columns.str.strip()
    check_temporal_transactions(trans_df)

    print("\nTemporal checks — accommodations...")
    accom_chunks = []
    for chunk in tqdm(pd.read_csv(RAW / "accommodations.csv", chunksize=CHUNK, low_memory=False), desc="Loading accom"):
        accom_chunks.append(chunk)
    accom_df = pd.concat(accom_chunks, ignore_index=True)
    check_temporal_accommodations(accom_df)
    del accom_df

    print("\nTemporal checks — activities...")
    act_chunks = []
    for chunk in tqdm(pd.read_csv(RAW / "activity.csv", chunksize=CHUNK, low_memory=False), desc="Loading act"):
        act_chunks.append(chunk)
    act_df = pd.concat(act_chunks, ignore_index=True)
    check_temporal_activities(act_df)
    del act_df

    check_cross_dataset()

    for name, df in reports.items():
        df.to_csv(OUT / f"issues_{name}.csv", index=False)

    print(f"\nReports saved to {OUT}")


if __name__ == "__main__":
    main()
