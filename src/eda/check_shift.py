import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm

RAW = Path(__file__).parents[2] / "dataset" / "raw"
OUT = Path(__file__).parents[2] / "result" / "eda" / "initial_observation"
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 200_000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COORD_RE = re.compile(r"^-?\d{1,3}\.\d+,-?\d{1,3}\.\d+$")
ID_RE = re.compile(r"^(HotelBeds|Expedia|Agoda|RateHawk)-\d+$")


def looks_like_coord(s) -> bool:
    return bool(COORD_RE.match(str(s))) if pd.notna(s) else False


def looks_like_id(s) -> bool:
    return bool(ID_RE.match(str(s))) if pd.notna(s) else False


def count_raw_fields(filepath: Path, delimiter: str = ",", sample_rows: int = 50_000) -> dict:
    """Count delimiter occurrences per raw line to detect field-count anomalies."""
    expected = None
    anomalies = {"short": 0, "long": 0, "sample_short": [], "sample_long": []}
    total = 0
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i == 0:
                expected = line.count(delimiter)
                continue
            if i > sample_rows:
                break
            count = line.count(delimiter)
            total += 1
            if count < expected:
                anomalies["short"] += 1
                if len(anomalies["sample_short"]) < 5:
                    anomalies["sample_short"].append((i + 1, count, line[:120]))
            elif count > expected:
                anomalies["long"] += 1
                if len(anomalies["sample_long"]) < 5:
                    anomalies["sample_long"].append((i + 1, count, line[:120]))
    anomalies["expected_delimiters"] = expected
    anomalies["rows_checked"] = total
    return anomalies


def check_numeric_contamination(series: pd.Series, col_name: str, label: str):
    """Flag non-numeric values in a column expected to be numeric."""
    numeric = pd.to_numeric(series, errors="coerce")
    bad = series[numeric.isna() & series.notna()]
    if len(bad) > 0:
        print(f"  [{label}] {col_name}: {len(bad)} non-numeric → samples: {bad.unique()[:8].tolist()}")
    else:
        print(f"  [{label}] {col_name}: clean ✓")


def check_string_contamination(series: pd.Series, col_name: str, label: str, check_fn, check_name: str):
    """Flag values in a string column that look like something else (coord, ID, etc.)."""
    hits = series.dropna().apply(check_fn).sum()
    if hits > 0:
        samples = series.dropna()[series.dropna().apply(check_fn)].head(5).tolist()
        print(f"  [{label}] {col_name}: {hits} values look like {check_name} → {samples}")


def cross_validate_price_columns(df: pd.DataFrame, col_a: str, col_b: str, label: str, tol: float = 0.01):
    """Check if two price columns that should be proportional have consistent ratio."""
    both = df[[col_a, col_b]].dropna()
    both = both[(both[col_a] > 0) & (both[col_b] > 0)]
    if len(both) == 0:
        print(f"  [{label}] {col_a} vs {col_b}: no overlapping non-zero rows")
        return
    ratio = (both[col_a] / both[col_b])
    cv = ratio.std() / ratio.mean() if ratio.mean() != 0 else float("inf")
    print(f"  [{label}] {col_a}/{col_b} ratio — mean: {ratio.mean():.4f}, std: {ratio.std():.4f}, CV: {cv:.4f}")
    extreme = ((ratio < 0.5) | (ratio > 2.0)).sum()
    print(f"    Rows with ratio outside [0.5, 2.0]: {extreme} (potential shift)")


def check_value_in_wrong_column(df: pd.DataFrame, numeric_col: str, label: str,
                                  lo: float, hi: float, note: str):
    """Values wildly outside expected range may be from an adjacent shifted column."""
    col = pd.to_numeric(df[numeric_col], errors="coerce").dropna()
    outliers = col[(col < lo) | (col > hi)]
    if len(outliers) > 0:
        print(f"  [{label}] {numeric_col}: {len(outliers)} values outside [{lo}, {hi}] — may be shifted from {note}")
        print(f"    Range of outliers: {outliers.min():.2f} → {outliers.max():.2f}")
        print(f"    Sample: {outliers.head(5).tolist()}")
    else:
        print(f"  [{label}] {numeric_col}: no out-of-range values ✓")


def check_adjacent_column_bleed(df: pd.DataFrame, cols: list, label: str):
    """
    For each pair of adjacent columns (as they appear in the DataFrame),
    check if values in col[i] look like they belong to col[i+1] or col[i-1].
    Heuristic: numeric col contains coord-like strings, or string col contains numbers.
    """
    print(f"\n  [{label}] Adjacent column bleed check:")
    for i in range(len(cols) - 1):
        c1, c2 = cols[i], cols[i + 1]
        if c1 not in df.columns or c2 not in df.columns:
            continue
        s1_dtype = df[c1].dtype
        s2_dtype = df[c2].dtype
        if s1_dtype == object and s2_dtype != object:
            bleed = df[c1].dropna().apply(lambda x: str(x).replace(".", "").replace("-", "").isdigit()).sum()
            if bleed > 0:
                print(f"    {c1} (str) contains {bleed} purely numeric values → possible shift from {c2}")
        if s2_dtype == object and s1_dtype != object:
            bleed = df[c2].dropna().apply(lambda x: str(x).replace(".", "").replace("-", "").isdigit()).sum()
            if bleed > 0:
                print(f"    {c2} (str) contains {bleed} purely numeric values → possible shift from {c1}")


# ---------------------------------------------------------------------------
# Per-dataset checks
# ---------------------------------------------------------------------------

def shift_check_transactions():
    print(f"\n{'='*60}")
    print("  SHIFT CHECK — TRANSACTIONS (xlsx, no raw delimiter scan)")
    print(f"{'='*60}")
    df = pd.read_excel(RAW / "Accommodation - Transactions by HotelCode.xlsx", engine="openpyxl")
    df.columns = df.columns.str.strip()

    print("\n  Column order:", df.columns.tolist())
    print(f"  Expected columns: 4, Actual: {len(df.columns)}")

    check_numeric_contamination(df["Number Transactions"], "Number Transactions", "TRANS")

    # Property Id should always match provider-ID format
    bad_ids = ~df["Property Id"].str.match(r"^(HotelBeds|Expedia|Agoda|RateHawk)-\d+$", na=False)
    print(f"  [TRANS] Property Id malformed: {bad_ids.sum()} → samples: {df[bad_ids]['Property Id'].head(5).tolist()}")

    # Product should be hotel names — flag if numeric or coord-like
    numeric_product = df["Product"].apply(lambda x: str(x).replace(".", "").replace("-", "").isdigit()).sum()
    coord_product = df["Product"].apply(looks_like_coord).sum()
    print(f"  [TRANS] Product column — purely numeric values: {numeric_product}, coord-like: {coord_product}")

    check_adjacent_column_bleed(df, df.columns.tolist(), "TRANS")


def shift_check_accommodations():
    print(f"\n{'='*60}")
    print("  SHIFT CHECK — ACCOMMODATIONS")
    print(f"{'='*60}")

    print("\n  Scanning raw CSV field counts (first 50K rows)...")
    result = count_raw_fields(RAW / "accommodations.csv", sample_rows=50_000)
    print(f"  Expected delimiters/row: {result['expected_delimiters']}")
    print(f"  Short rows (missing fields): {result['short']}")
    print(f"  Long rows (extra fields): {result['long']}")
    if result["sample_long"]:
        print("  Sample long rows:")
        for row_num, count, line in result["sample_long"]:
            print(f"    Row {row_num} ({count} delimiters): {line}")
    if result["sample_short"]:
        print("  Sample short rows:")
        for row_num, count, line in result["sample_short"]:
            print(f"    Row {row_num} ({count} delimiters): {line}")

    print("\n  Loading sample (200K rows) for column checks...")
    df = pd.read_csv(RAW / "accommodations.csv", nrows=200_000, low_memory=False)
    df["provider"] = df["id"].str.split("-").str[0]

    print(f"\n  Column order: {df.columns.tolist()}")

    print("\n  Numeric contamination:")
    for col in ["popularity", "popularity_score", "star_rating", "guest_rating",
                "availability_score", "reference_price", "price_in_aud"]:
        if col in df.columns:
            check_numeric_contamination(df[col], col, "ACCOM")

    print("\n  String column contamination (coord/ID bleed):")
    for col in ["name", "city", "country", "destination_display_name"]:
        if col in df.columns:
            check_string_contamination(df[col], col, "ACCOM", looks_like_coord, "coordinate")
            check_string_contamination(df[col], col, "ACCOM", looks_like_id, "property ID")

    print("\n  Cross-column price consistency:")
    cross_validate_price_columns(df, "reference_price", "price_in_aud", "ACCOM")

    print("\n  Out-of-range values (possible column shift):")
    check_value_in_wrong_column(df, "star_rating", "ACCOM", 0, 5, "adjacent numeric column")
    check_value_in_wrong_column(df, "guest_rating", "ACCOM", 0, 100, "adjacent column")
    check_value_in_wrong_column(df, "availability_score", "ACCOM", 0, 150, "derived field overflow")

    # Check if destination_coordinate contains non-coordinate strings
    coord_col = df["destination_coordinate"].dropna()
    bad_coords = coord_col[~coord_col.apply(looks_like_coord)]
    print(f"\n  [ACCOM] destination_coordinate non-coord values: {len(bad_coords)}")
    if len(bad_coords) > 0:
        print(f"    Samples: {bad_coords.head(5).tolist()}")

    check_adjacent_column_bleed(df, df.columns.tolist(), "ACCOM")


def shift_check_activities():
    print(f"\n{'='*60}")
    print("  SHIFT CHECK — ACTIVITIES")
    print(f"{'='*60}")

    print("\n  Scanning raw CSV field counts (first 50K rows)...")
    result = count_raw_fields(RAW / "activity.csv", sample_rows=50_000)
    print(f"  Expected delimiters/row: {result['expected_delimiters']}")
    print(f"  Short rows (missing fields): {result['short']}")
    print(f"  Long rows (extra fields): {result['long']}")
    if result["sample_long"]:
        print("  Sample long rows:")
        for row_num, count, line in result["sample_long"]:
            print(f"    Row {row_num} ({count} delimiters): {line}")

    print("\n  Loading sample (100K rows) for column checks...")
    df = pd.read_csv(RAW / "activity.csv", nrows=100_000, low_memory=False)

    print(f"\n  Column order: {df.columns.tolist()}")

    print("\n  Numeric contamination:")
    for col in ["rating", "normalized_rating", "duration_hours", "price", "price_in_aud", "availability_score"]:
        if col in df.columns:
            check_numeric_contamination(df[col], col, "ACT")

    print("\n  normalized_rating artifact deep-dive (apostrophe-prefix pattern):")
    norm = df["normalized_rating"].astype(str)
    apostrophe_vals = norm[norm.str.startswith("'")]
    print(f"  Apostrophe-prefixed values: {len(apostrophe_vals)}")
    print(f"  Unique patterns: {apostrophe_vals.unique()[:15].tolist()}")
    stripped = pd.to_numeric(apostrophe_vals.str.lstrip("'"), errors="coerce")
    print(f"  After stripping apostrophe → numeric: {stripped.notna().sum()}, still invalid: {stripped.isna().sum()}")
    print(f"  Range of stripped values: {stripped.min():.2f} → {stripped.max():.2f}")

    print("\n  Out-of-range values (possible column shift):")
    check_value_in_wrong_column(df, "rating", "ACT", 0, 10, "price or view_count column")
    check_value_in_wrong_column(df, "duration_hours", "ACT", 0, 720, "price column (24000 hrs = price value?)")
    check_value_in_wrong_column(df, "price_in_aud", "ACT", 0, 100_000, "extreme outlier / shifted field")

    print("\n  Cross-column price consistency:")
    cross_validate_price_columns(df, "price", "price_in_aud", "ACT")

    print("\n  String column contamination:")
    for col in ["name", "destination_display_name", "destination_country"]:
        if col in df.columns:
            check_string_contamination(df[col], col, "ACT", looks_like_coord, "coordinate")

    # Check destination_coordinate format
    coord_col = df["destination_coordinate"].dropna()
    bad_coords = coord_col[~coord_col.apply(looks_like_coord)]
    print(f"\n  [ACT] destination_coordinate non-coord values: {len(bad_coords)}")
    if len(bad_coords) > 0:
        print(f"    Unique bad patterns (first 10): {bad_coords.unique()[:10].tolist()}")

    # Check if categories column bled into adjacent columns
    print("\n  Category bleed check (pipe-delimited categories vs adjacent columns):")
    pipe_in_rating = df["rating"].astype(str).str.contains(r"\|").sum()
    pipe_in_duration = df["duration_hours"].astype(str).str.contains(r"\|").sum()
    print(f"  Pipe char in 'rating': {pipe_in_rating}, in 'duration_hours': {pipe_in_duration}")

    check_adjacent_column_bleed(df, df.columns.tolist(), "ACT")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    shift_check_transactions()
    shift_check_accommodations()
    shift_check_activities()
    print("\nShift check complete.")


if __name__ == "__main__":
    main()
