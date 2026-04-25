import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

RAW = Path(__file__).parents[2] / "dataset" / "raw"
OUT = Path(__file__).parents[2] / "result" / "eda" / "transactions"
OUT.mkdir(parents=True, exist_ok=True)

TRANS_FILE = RAW / "Accommodation - Transactions by HotelCode.xlsx"


def load_data() -> pd.DataFrame:
    df = pd.read_excel(TRANS_FILE, engine="openpyxl")
    df.columns = df.columns.str.strip()
    df["payment_date"] = pd.to_datetime(df["payment_date"])
    df["year_month"] = df["payment_date"].dt.to_period("M")
    df["provider"] = df["Property Id"].str.split("-").str[0]
    return df


def basic_stats(df: pd.DataFrame):
    print(f"Rows: {len(df):,}")
    print(f"Date range: {df['payment_date'].min()} → {df['payment_date'].max()}")
    print(f"Unique properties: {df['Property Id'].nunique():,}")
    print(f"Unique months: {df['year_month'].nunique()}")
    print(f"Unique products: {df['Product'].nunique()}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nDescribe:\n{df.describe(include='all')}")


def plot_monthly_volume(df: pd.DataFrame):
    monthly = df.groupby("year_month")["Number Transactions"].sum().reset_index()
    monthly["year_month"] = monthly["year_month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly["year_month"], monthly["Number Transactions"], width=20, color="steelblue")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Monthly Transaction Volume")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Transactions")
    plt.tight_layout()
    plt.savefig(OUT / "monthly_volume.png", dpi=150)
    plt.close()


def plot_provider_distribution(df: pd.DataFrame):
    provider_counts = df.groupby("provider")["Number Transactions"].sum().sort_values(ascending=False)
    provider_rows = df["provider"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    provider_counts.plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Transaction Volume by Provider")
    axes[0].set_ylabel("Total Transactions")
    axes[0].tick_params(axis="x", rotation=45)

    provider_rows.plot(kind="bar", ax=axes[1], color="coral")
    axes[1].set_title("Row Count by Provider")
    axes[1].set_ylabel("Rows")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(OUT / "provider_distribution.png", dpi=150)
    plt.close()

    print("\nProvider transaction share:")
    print((provider_counts / provider_counts.sum() * 100).round(1))


def plot_property_transaction_distribution(df: pd.DataFrame):
    prop_total = df.groupby("Property Id")["Number Transactions"].sum()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(prop_total, bins=50, color="steelblue", log=True)
    axes[0].set_title("Transaction Distribution per Property (log scale)")
    axes[0].set_xlabel("Total Transactions")
    axes[0].set_ylabel("Count (log)")

    axes[1].hist(prop_total[prop_total < prop_total.quantile(0.95)], bins=50, color="coral")
    axes[1].set_title("Transaction Distribution per Property (95th pct clip)")
    axes[1].set_xlabel("Total Transactions")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(OUT / "property_transaction_dist.png", dpi=150)
    plt.close()

    print(f"\nProperties with only 1 transaction row: {(prop_total == 1).sum()}")
    print(f"Properties with ≥5 transaction rows: {(prop_total >= 5).sum()}")
    print(f"Top-10 properties by transactions:\n{prop_total.nlargest(10)}")


def analyze_temporal_coverage(df: pd.DataFrame):
    pivot = df.pivot_table(
        index="Property Id", columns="year_month", values="Number Transactions", aggfunc="sum"
    ).fillna(0)
    active_months = (pivot > 0).sum(axis=1)
    print(f"\nTemporal coverage per property:")
    print(active_months.describe())
    print(f"\nProperties active all 13 months: {(active_months == pivot.shape[1]).sum()}")
    print(f"Properties active ≤3 months: {(active_months <= 3).sum()}")

    fig, ax = plt.subplots(figsize=(8, 4))
    active_months.hist(bins=range(1, pivot.shape[1] + 2), ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Active Months per Property")
    ax.set_xlabel("Number of Active Months")
    ax.set_ylabel("Property Count")
    plt.tight_layout()
    plt.savefig(OUT / "property_temporal_coverage.png", dpi=150)
    plt.close()


def propose_destination_clusters(df: pd.DataFrame):
    # Properties need to be grouped into destination clusters for TFT
    # (13 months per property is too sparse for individual TS)
    # Clustering strategy: use Property Id prefix (provider) OR
    # join with accommodations.csv on Property Id → city/destination_display_name
    # This function prints candidate cluster sizes to inform M1 design

    provider_monthly = (
        df.groupby(["provider", "year_month"])["Number Transactions"]
        .sum()
        .reset_index()
    )
    print("\nMonthly transaction stats per provider cluster:")
    print(provider_monthly.groupby("provider")["Number Transactions"].describe())

    # Check how many unique properties per product type
    print(f"\nProduct breakdown:\n{df['Product'].value_counts()}")
    print(f"\nTransactions per product:\n{df.groupby('Product')['Number Transactions'].sum()}")


def plot_product_monthly_trends(df: pd.DataFrame):
    product_monthly = (
        df.groupby(["year_month", "Product"])["Number Transactions"]
        .sum()
        .reset_index()
    )
    product_monthly["year_month"] = product_monthly["year_month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    for product, grp in product_monthly.groupby("Product"):
        ax.plot(grp["year_month"], grp["Number Transactions"], marker="o", label=product)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Monthly Transactions by Product Type")
    ax.set_ylabel("Total Transactions")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "product_monthly_trends.png", dpi=150)
    plt.close()


def main():
    print("Loading transactions...")
    df = load_data()

    print("\n--- Basic Stats ---")
    basic_stats(df)

    steps = [
        ("Monthly volume", plot_monthly_volume),
        ("Provider distribution", plot_provider_distribution),
        ("Property transaction dist", plot_property_transaction_distribution),
        ("Temporal coverage", analyze_temporal_coverage),
        ("Destination cluster candidates", propose_destination_clusters),
        ("Product monthly trends", plot_product_monthly_trends),
    ]

    for name, fn in tqdm(steps, desc="EDA steps"):
        print(f"\n--- {name} ---")
        fn(df)

    print(f"\nAll plots saved to {OUT}")


if __name__ == "__main__":
    main()
