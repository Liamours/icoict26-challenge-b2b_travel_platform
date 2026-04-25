import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

RAW = Path(__file__).parents[2] / "dataset" / "raw"
OUT = Path(__file__).parents[2] / "dataset" / "preprocessed"
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 200_000


# ---------------------------------------------------------------------------
# Shared utils
# ---------------------------------------------------------------------------

def parse_coordinate(series: pd.Series) -> pd.DataFrame:
    """Split 'lat,lon' string (with optional leading apostrophe) into lat/lon floats."""
    cleaned = series.astype(str).str.lstrip("'").str.strip()
    split = cleaned.str.split(",", n=1, expand=True)
    lat = pd.to_numeric(split[0], errors="coerce")
    lon = pd.to_numeric(split[1], errors="coerce")
    null_island = (lat == 0) & (lon == 0)
    lat[null_island] = np.nan
    lon[null_island] = np.nan
    return pd.DataFrame({"lat": lat, "lon": lon})


def log_drop(label: str, before: int, after: int):
    dropped = before - after
    print(f"  [{label}] Dropped {dropped:,} rows ({dropped/before*100:.2f}%) → {after:,} remaining")


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

def preprocess_transactions():
    print("\n=== TRANSACTIONS ===")
    df = pd.read_excel(RAW / "Accommodation - Transactions by HotelCode.xlsx", engine="openpyxl")
    df.columns = df.columns.str.strip()
    n0 = len(df)

    df["payment_date"] = pd.to_datetime(df["payment_date"]).dt.tz_localize("UTC")
    df["year_month"] = df["payment_date"].dt.to_period("M").dt.to_timestamp()
    df["provider"] = df["Property Id"].str.split("-").str[0]

    df = df.rename(columns={"Property Id": "property_id", "Product": "product_name",
                             "Number Transactions": "transaction_count"})

    df = df[["payment_date", "year_month", "property_id", "provider", "product_name", "transaction_count"]]

    log_drop("TRANS", n0, len(df))
    df.to_parquet(OUT / "transactions.parquet", index=False)
    print(f"  Saved: transactions.parquet ({len(df):,} rows)")

    # Monthly aggregate per provider — primary input for M1
    monthly = (
        df.groupby(["provider", "year_month"])
        .agg(
            total_transactions=("transaction_count", "sum"),
            unique_properties=("property_id", "nunique"),
            unique_products=("product_name", "nunique"),
        )
        .reset_index()
        .sort_values(["provider", "year_month"])
    )
    monthly.to_parquet(OUT / "transactions_monthly.parquet", index=False)
    print(f"  Saved: transactions_monthly.parquet ({len(monthly):,} rows — {monthly['provider'].nunique()} providers × {monthly['year_month'].nunique()} months)")

    return df


# ---------------------------------------------------------------------------
# Accommodations
# ---------------------------------------------------------------------------

def preprocess_accommodations():
    print("\n=== ACCOMMODATIONS ===")

    DROP_COLS = ["tags", "reference_price", "star_rating_as_string", "images",
                 "popularity", "destination_neighborhoods"]

    chunks_out = []
    n_raw = 0
    n_kept = 0

    chunks = list(tqdm(
        pd.read_csv(RAW / "accommodations.csv", chunksize=CHUNK, low_memory=False),
        desc="  Processing chunks"
    ))

    for chunk in tqdm(chunks, desc="  Cleaning"):
        n_raw += len(chunk)

        # Drop invalid availability_score rows
        chunk = chunk[(chunk["availability_score"] >= 0) & (chunk["availability_score"] <= 200)]

        # Parse and drop null-island coordinates
        coords = parse_coordinate(chunk["destination_coordinate"])
        chunk = chunk[coords["lat"].notna() & coords["lon"].notna()].copy()
        coords = coords.loc[chunk.index]
        chunk["lat"] = coords["lat"].values
        chunk["lon"] = coords["lon"].values
        chunk = chunk.drop(columns=["destination_coordinate"])

        # Fix star_rating: clip negatives to 0
        chunk["star_rating"] = chunk["star_rating"].clip(lower=0)

        # Normalize guest_rating from 0-100 → 0-10
        chunk["guest_rating"] = chunk["guest_rating"].clip(upper=100) / 10.0

        # Cap extreme price outliers at 99th percentile (computed per chunk — approximate)
        for col in ["price_in_aud"]:
            p99 = chunk[col][chunk[col] > 0].quantile(0.99) if (chunk[col] > 0).any() else np.nan
            if pd.notna(p99):
                chunk[col] = chunk[col].clip(upper=p99)

        # Replace zero prices with NaN (unreliable signal)
        chunk.loc[chunk["price_in_aud"] == 0, "price_in_aud"] = np.nan

        # Extract provider from id
        chunk["provider"] = chunk["id"].str.split("-").str[0]

        # Drop low-signal columns
        chunk = chunk.drop(columns=[c for c in DROP_COLS if c in chunk.columns])

        n_kept += len(chunk)
        chunks_out.append(chunk)

    df = pd.concat(chunks_out, ignore_index=True)
    log_drop("ACCOM", n_raw, n_kept)

    df.to_parquet(OUT / "accommodations.parquet", index=False)
    print(f"  Saved: accommodations.parquet ({len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB)")

    print(f"\n  Provider breakdown after cleaning:")
    print(df["provider"].value_counts().to_string())

    return df


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------

def preprocess_activities():
    print("\n=== ACTIVITIES ===")

    DROP_COLS = ["price", "tags", "images"]

    chunks_out = []
    n_raw = 0

    price_aud_p99 = None

    # First pass: compute global 99th pct for price_in_aud
    print("  Computing price_in_aud 99th percentile...")
    prices = []
    for chunk in pd.read_csv(RAW / "activity.csv", chunksize=CHUNK, low_memory=False,
                              usecols=["price_in_aud"]):
        p = chunk["price_in_aud"]
        prices.append(p[p > 0].values)
    price_aud_p99 = np.nanpercentile(np.concatenate(prices), 99)
    print(f"  price_in_aud 99th pct: {price_aud_p99:.2f} AUD")

    chunks = list(tqdm(
        pd.read_csv(RAW / "activity.csv", chunksize=CHUNK, low_memory=False),
        desc="  Processing chunks"
    ))

    for chunk in tqdm(chunks, desc="  Cleaning"):
        n_raw += len(chunk)

        # Fix normalized_rating: strip apostrophe, convert to numeric
        chunk["normalized_rating"] = pd.to_numeric(
            chunk["normalized_rating"].astype(str).str.lstrip("'"),
            errors="coerce"
        )

        # Clip rating at 10 (max valid scale)
        chunk["rating"] = chunk["rating"].clip(upper=10)

        # duration_hours: zero → NaN, >720 → NaN
        chunk["duration_hours"] = chunk["duration_hours"].replace(0, np.nan)
        chunk.loc[chunk["duration_hours"] > 720, "duration_hours"] = np.nan

        # price_in_aud: cap at p99, zero → NaN
        chunk["price_in_aud"] = chunk["price_in_aud"].clip(upper=price_aud_p99)
        chunk.loc[chunk["price_in_aud"] == 0, "price_in_aud"] = np.nan

        # has_listing_id_in_bali_db: object → bool
        chunk["has_listing_id_in_bali_db"] = (
            chunk["has_listing_id_in_bali_db"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False, "nan": False})
            .fillna(False)
            .astype(bool)
        )

        # Parse coordinates (strip apostrophe artifact)
        coords = parse_coordinate(chunk["destination_coordinate"].fillna("nan,nan"))
        chunk["lat"] = coords["lat"].values
        chunk["lon"] = coords["lon"].values
        chunk = chunk.drop(columns=["destination_coordinate"])

        # Drop low-signal columns
        chunk = chunk.drop(columns=[c for c in DROP_COLS if c in chunk.columns])

        chunks_out.append(chunk)

    df = pd.concat(chunks_out, ignore_index=True)
    log_drop("ACT", n_raw, len(df))

    df.to_parquet(OUT / "activities.parquet", index=False)
    print(f"  Saved: activities.parquet ({len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB)")

    # Bali subset
    bali = df[df["has_listing_id_in_bali_db"]].copy()
    bali.to_parquet(OUT / "activities_bali.parquet", index=False)
    print(f"  Saved: activities_bali.parquet ({len(bali):,} rows)")

    print(f"\n  Category distribution (top 10):")
    print(df["categories"].value_counts().head(10).to_string())

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Starting preprocessing...")
    preprocess_transactions()
    preprocess_accommodations()
    preprocess_activities()
    print(f"\nAll outputs saved to: {OUT}")
    print("\nFiles:")
    for f in sorted(OUT.glob("*.parquet")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:<45} {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
