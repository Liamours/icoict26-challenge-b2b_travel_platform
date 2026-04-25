import pandas as pd
import numpy as np

df = pd.read_parquet("dataset/preprocessed/activities.parquet")
print("Shape:", df.shape)
print("Cols:", df.columns.tolist())
print("\ndtypes:\n", df.dtypes)
print("\ncats sample:", df["categories_as_string"].dropna().head(5).tolist())
print("\ntop dests:", df["destination_display_name"].value_counts().head(15).to_dict())
print("\nprice NaN:", df["price_in_aud"].isna().sum())
print("rating NaN:", df["rating"].isna().sum())
print("duration NaN:", df["duration_hours"].isna().sum())
print("lat NaN:", df["lat"].isna().sum())
print("\nprice range:", df["price_in_aud"].min(), df["price_in_aud"].max())
print("rating range:", df["rating"].min(), df["rating"].max())
print("duration range:", df["duration_hours"].min(), df["duration_hours"].max())

# Check destinations with >=100 activities AND appear in accommodations
acc = pd.read_parquet("dataset/preprocessed/accommodations.parquet", columns=["destination_display_name"])
acc_dests = set(acc["destination_display_name"].dropna().unique())
act_dests = df["destination_display_name"].value_counts()
big = act_dests[act_dests >= 100]
overlap = [d for d in big.index if d in acc_dests]
print(f"\nDests with >=100 activities AND in accommodations: {len(overlap)}")
print(overlap[:20])
