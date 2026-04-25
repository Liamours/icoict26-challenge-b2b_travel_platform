import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

RAW = Path(__file__).parents[2] / "dataset" / "raw"
OUT = Path(__file__).parents[2] / "result" / "eda" / "accommodations"
OUT.mkdir(parents=True, exist_ok=True)

ACCOM_FILE = RAW / "accommodations.csv"
CHUNK_SIZE = 200_000


def load_data() -> pd.DataFrame:
    # 3.7M rows — load in chunks, sample for plots, full pass for aggregates
    chunks = []
    for chunk in tqdm(
        pd.read_csv(ACCOM_FILE, chunksize=CHUNK_SIZE, low_memory=False),
        desc="Loading accommodations"
    ):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df["provider"] = df["id"].str.split("-").str[0]
    return df


def basic_stats(df: pd.DataFrame):
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nDescribe (numeric):\n{df.describe()}")


def plot_provider_distribution(df: pd.DataFrame):
    provider_counts = df["provider"].value_counts()
    print(f"\nProvider distribution:\n{provider_counts}")
    print(f"\nProvider share (%):\n{(provider_counts / len(df) * 100).round(2)}")

    fig, ax = plt.subplots(figsize=(8, 4))
    provider_counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Accommodation Count by Provider")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(OUT / "provider_distribution.png", dpi=150)
    plt.close()


def plot_rating_distributions(df: pd.DataFrame):
    # Key insight for Module 2: do providers have different rating scales?
    # If yes, contrastive debiasing is justified
    rating_cols = ["guest_rating", "star_rating", "popularity_score"]
    available = [c for c in rating_cols if c in df.columns]

    fig, axes = plt.subplots(len(available), 2, figsize=(14, 4 * len(available)))
    if len(available) == 1:
        axes = [axes]

    for i, col in enumerate(available):
        # Global distribution
        df[col].dropna().hist(bins=50, ax=axes[i][0], color="steelblue")
        axes[i][0].set_title(f"{col} — Global Distribution")
        axes[i][0].set_xlabel(col)

        # Per-provider distribution — critical for bias detection
        for provider, grp in df.groupby("provider"):
            grp[col].dropna().hist(bins=30, ax=axes[i][1], alpha=0.5, label=provider)
        axes[i][1].set_title(f"{col} — By Provider (overlap)")
        axes[i][1].set_xlabel(col)
        axes[i][1].legend()

    plt.tight_layout()
    plt.savefig(OUT / "rating_distributions.png", dpi=150)
    plt.close()

    # KS test between providers to quantify distribution divergence
    from scipy.stats import ks_2samp
    providers = df["provider"].unique()
    print("\nKS test results (provider rating divergence):")
    for col in available:
        print(f"\n  {col}:")
        provider_data = {p: df[df["provider"] == p][col].dropna().values for p in providers}
        for i, p1 in enumerate(providers):
            for p2 in providers[i + 1:]:
                if len(provider_data[p1]) > 0 and len(provider_data[p2]) > 0:
                    stat, pval = ks_2samp(provider_data[p1], provider_data[p2])
                    print(f"    {p1} vs {p2}: KS={stat:.3f}, p={pval:.2e}")


def plot_provider_rating_boxplots(df: pd.DataFrame):
    rating_cols = [c for c in ["guest_rating", "star_rating", "popularity_score"] if c in df.columns]
    fig, axes = plt.subplots(1, len(rating_cols), figsize=(6 * len(rating_cols), 5))
    if len(rating_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, rating_cols):
        sample = df[[col, "provider"]].dropna().sample(min(50_000, len(df)), random_state=42)
        sample.boxplot(column=col, by="provider", ax=ax)
        ax.set_title(f"{col} by Provider")
        ax.set_xlabel("Provider")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(OUT / "provider_rating_boxplots.png", dpi=150)
    plt.close()


def plot_price_distribution(df: pd.DataFrame):
    if "reference_price" not in df.columns:
        print("reference_price column not found, skipping")
        return
    price = df["reference_price"].dropna()
    price = price[price > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(np.log1p(price), bins=60, color="steelblue")
    axes[0].set_title("log(1 + reference_price) Distribution")
    axes[0].set_xlabel("log price")

    for provider, grp in df.groupby("provider"):
        grp_price = grp["reference_price"].dropna()
        grp_price = grp_price[grp_price > 0]
        axes[1].hist(np.log1p(grp_price), bins=40, alpha=0.5, label=provider)
    axes[1].set_title("log Price by Provider")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT / "price_distribution.png", dpi=150)
    plt.close()

    print(f"\nPrice stats:\n{price.describe()}")
    print(f"\nPrice by provider median:")
    print(df.groupby("provider")["reference_price"].median())


def analyze_geographic_coverage(df: pd.DataFrame):
    # destination_display_name and destination_coordinate → cluster candidates for M1/M3
    if "destination_display_name" in df.columns:
        top_destinations = df["destination_display_name"].value_counts().head(30)
        print(f"\nTop 30 destinations:\n{top_destinations}")

        fig, ax = plt.subplots(figsize=(10, 6))
        top_destinations.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title("Top 30 Destinations by Property Count")
        ax.set_xlabel("Count")
        plt.tight_layout()
        plt.savefig(OUT / "top_destinations.png", dpi=150)
        plt.close()

    if "country" in df.columns:
        print(f"\nTop 20 countries:\n{df['country'].value_counts().head(20)}")

    if "destination_coordinate" in df.columns:
        non_null = df["destination_coordinate"].dropna()
        print(f"\ndestination_coordinate non-null: {len(non_null):,} / {len(df):,}")


def analyze_availability(df: pd.DataFrame):
    # availability columns → important for inventory scoring (M2)
    avail_cols = [c for c in df.columns if "avail" in c.lower()]
    print(f"\nAvailability-related columns: {avail_cols}")
    if avail_cols:
        print(df[avail_cols].describe())
        print(df[avail_cols].isnull().sum())


def check_transaction_join_coverage(df: pd.DataFrame):
    # How many transaction Property Ids exist in accommodations?
    # Critical: validates that M1 and M2 can be linked
    trans_file = RAW / "Accommodation - Transactions by HotelCode.xlsx"
    trans = pd.read_excel(trans_file, engine="openpyxl")
    trans_ids = set(trans["Property Id"].unique())
    accom_ids = set(df["id"].unique())
    overlap = trans_ids & accom_ids
    print(f"\nTransaction Property IDs: {len(trans_ids):,}")
    print(f"Accommodation IDs: {len(accom_ids):,}")
    print(f"Overlap (joinable): {len(overlap):,} ({len(overlap)/len(trans_ids)*100:.1f}% of transactions)")


def main():
    print("Loading accommodations (3.7M rows)...")
    df = load_data()

    print("\n--- Basic Stats ---")
    basic_stats(df)

    steps = [
        ("Provider distribution", plot_provider_distribution),
        ("Rating distributions + KS test", plot_rating_distributions),
        ("Provider rating boxplots", plot_provider_rating_boxplots),
        ("Price distribution", plot_price_distribution),
        ("Geographic coverage", analyze_geographic_coverage),
        ("Availability columns", analyze_availability),
        ("Transaction join coverage", check_transaction_join_coverage),
    ]

    for name, fn in tqdm(steps, desc="EDA steps"):
        print(f"\n--- {name} ---")
        fn(df)

    print(f"\nAll plots saved to {OUT}")


if __name__ == "__main__":
    main()
