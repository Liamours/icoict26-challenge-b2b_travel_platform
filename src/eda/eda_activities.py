import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

RAW = Path(__file__).parents[2] / "dataset" / "raw"
OUT = Path(__file__).parents[2] / "result" / "eda" / "activities"
OUT.mkdir(parents=True, exist_ok=True)

ACTIVITY_FILE = RAW / "activity.csv"
CHUNK_SIZE = 100_000


def load_data() -> pd.DataFrame:
    chunks = []
    for chunk in tqdm(
        pd.read_csv(ACTIVITY_FILE, chunksize=CHUNK_SIZE, low_memory=False),
        desc="Loading activities"
    ):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def basic_stats(df: pd.DataFrame):
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nDescribe (numeric):\n{df.describe()}")


def analyze_categories(df: pd.DataFrame):
    # categories column: Outdoor / Cultural / Culinary — main split for M3
    if "categories" not in df.columns:
        print("'categories' column not found")
        return

    cat_counts = df["categories"].value_counts()
    print(f"\nCategory distribution:\n{cat_counts}")

    fig, ax = plt.subplots(figsize=(10, 4))
    cat_counts.head(20).plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Activity Category Distribution (Top 20)")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(OUT / "category_distribution.png", dpi=150)
    plt.close()


def plot_rating_analysis(df: pd.DataFrame):
    rating_cols = [c for c in ["rating", "normalized_rating"] if c in df.columns]

    fig, axes = plt.subplots(1, len(rating_cols), figsize=(6 * len(rating_cols), 4))
    if len(rating_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, rating_cols):
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        series.hist(bins=50, ax=ax, color="steelblue")
        ax.set_title(f"{col} Distribution")
        ax.set_xlabel(col)

    plt.tight_layout()
    plt.savefig(OUT / "rating_distributions.png", dpi=150)
    plt.close()

    if "rating" in df.columns and "normalized_rating" in df.columns:
        # Check if normalized_rating is just a linear rescale of rating
        # If correlation ≈ 1.0 and linear, then it adds no new info — naive rescale confirmed
        tmp = df[["rating", "normalized_rating"]].copy()
        tmp["normalized_rating"] = pd.to_numeric(tmp["normalized_rating"], errors="coerce")
        corr = tmp.dropna().corr().iloc[0, 1]
        print(f"\nrating ↔ normalized_rating Pearson correlation: {corr:.4f}")
        print("(If ≈1.0, normalized_rating is naive rescale — no new signal for M2 debiasing)")

        # Plot scatter to visualize
        sample = tmp.dropna().sample(min(5000, len(tmp.dropna())), random_state=42)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sample["rating"], sample["normalized_rating"], alpha=0.3, s=5)
        ax.set_xlabel("rating")
        ax.set_ylabel("normalized_rating")
        ax.set_title("rating vs normalized_rating")
        plt.tight_layout()
        plt.savefig(OUT / "rating_vs_normalized.png", dpi=150)
        plt.close()


def plot_price_by_category(df: pd.DataFrame):
    if "price_in_aud" not in df.columns or "categories" not in df.columns:
        return
    price = df[["price_in_aud", "categories"]].dropna()
    price = price[price["price_in_aud"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    price.boxplot(column="price_in_aud", by="categories", ax=axes[0])
    axes[0].set_title("Price by Category")
    plt.suptitle("")

    for cat, grp in price.groupby("categories"):
        grp["price_in_aud"].apply(np.log1p).hist(bins=40, alpha=0.5, ax=axes[1], label=cat)
    axes[1].set_title("log(Price) by Category")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT / "price_by_category.png", dpi=150)
    plt.close()

    print(f"\nPrice stats by category:")
    print(price.groupby("categories")["price_in_aud"].describe())


def plot_duration_analysis(df: pd.DataFrame):
    if "duration_hours" not in df.columns:
        return
    dur = df["duration_hours"].dropna()
    dur = dur[dur > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    dur.hist(bins=50, ax=axes[0], color="steelblue")
    axes[0].set_title("Duration (hours) Distribution")
    axes[0].set_xlabel("Hours")

    dur[dur < dur.quantile(0.95)].hist(bins=50, ax=axes[1], color="coral")
    axes[1].set_title("Duration — 95th pct clip")
    axes[1].set_xlabel("Hours")

    plt.tight_layout()
    plt.savefig(OUT / "duration_distribution.png", dpi=150)
    plt.close()

    print(f"\nDuration stats:\n{dur.describe()}")


def analyze_bali_subset(df: pd.DataFrame):
    # has_listing_id_in_bali_db flag — Bali-specific subset used for M3
    if "has_listing_id_in_bali_db" not in df.columns:
        print("has_listing_id_in_bali_db column not found")
        return

    bali = df[df["has_listing_id_in_bali_db"] == True]
    non_bali = df[df["has_listing_id_in_bali_db"] != True]

    print(f"\nBali subset: {len(bali):,} rows ({len(bali)/len(df)*100:.1f}%)")
    print(f"Non-Bali: {len(non_bali):,} rows")

    if "categories" in df.columns:
        print(f"\nBali category distribution:\n{bali['categories'].value_counts()}")

    if "rating" in df.columns:
        print(f"\nBali rating stats:\n{bali['rating'].describe()}")
        print(f"\nNon-Bali rating stats:\n{non_bali['rating'].describe()}")


def analyze_geographic_coverage(df: pd.DataFrame):
    if "destination_display_name" in df.columns:
        top_dest = df["destination_display_name"].value_counts().head(30)
        print(f"\nTop 30 activity destinations:\n{top_dest}")

        fig, ax = plt.subplots(figsize=(10, 6))
        top_dest.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title("Top 30 Activity Destinations")
        ax.set_xlabel("Count")
        plt.tight_layout()
        plt.savefig(OUT / "top_destinations.png", dpi=150)
        plt.close()

    if "destination_country" in df.columns:
        print(f"\nCountry distribution:\n{df['destination_country'].value_counts().head(20)}")

    if "destination_coordinate" in df.columns:
        non_null = df["destination_coordinate"].dropna()
        print(f"\ndestination_coordinate non-null: {len(non_null):,} / {len(df):,}")


def check_accommodation_spatial_overlap(df: pd.DataFrame):
    # M3 join strategy: activity ↔ accommodation via destination_coordinate
    # This function checks coordinate format and assesses overlap feasibility
    accom_file = RAW / "accommodations.csv"
    accom_sample = pd.read_csv(accom_file, nrows=10_000, low_memory=False)

    print(f"\nActivity coordinate sample:\n{df['destination_coordinate'].dropna().head(5).values}")
    print(f"\nAccommodation coordinate sample:\n{accom_sample['destination_coordinate'].dropna().head(5).values}")

    # Check if both use same format (lat,lon string vs separate columns)
    # Spatial join will need haversine distance or geohash bucketing
    act_coords = df["destination_coordinate"].dropna()
    acc_coords = accom_sample["destination_coordinate"].dropna()
    print(f"\nActivity coord dtype: {act_coords.dtype}")
    print(f"Accommodation coord dtype: {acc_coords.dtype}")
    print(f"\nActivity unique coords: {act_coords.nunique():,}")


def analyze_providers(df: pd.DataFrame):
    if "providers" not in df.columns:
        return
    print(f"\nActivity provider distribution:\n{df['providers'].value_counts().head(20)}")


def main():
    print("Loading activities...")
    df = load_data()

    print("\n--- Basic Stats ---")
    basic_stats(df)

    steps = [
        ("Category analysis", analyze_categories),
        ("Rating analysis", plot_rating_analysis),
        ("Price by category", plot_price_by_category),
        ("Duration analysis", plot_duration_analysis),
        ("Bali subset", analyze_bali_subset),
        ("Geographic coverage", analyze_geographic_coverage),
        ("Accommodation spatial overlap", check_accommodation_spatial_overlap),
        ("Provider analysis", analyze_providers),
    ]

    for name, fn in tqdm(steps, desc="EDA steps"):
        print(f"\n--- {name} ---")
        fn(df)

    print(f"\nAll plots saved to {OUT}")


if __name__ == "__main__":
    main()
