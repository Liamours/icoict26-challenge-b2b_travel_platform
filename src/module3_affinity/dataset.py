"""
C3 — Activity feature builder + destination cluster embeddings.

Activity feature vector (25-dim base, 27-dim with OSM):
  continuous (9): log1p_price, rating/10, log1p_duration, lat/90, lon/180,
                  avail/200, price_miss, duration_miss, lat_miss
  osm (2, optional): poi_density_1km_norm, poi_density_5km_norm
  category emb (16): Embedding(n_cats+1, 16), primary category = first token
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

ACTIVITIES = Path(__file__).parents[2] / "dataset" / "preprocessed" / "activities.parquet"
ACCOMMODATIONS = Path(__file__).parents[2] / "dataset" / "preprocessed" / "accommodations.parquet"
EMB_PATH = Path(__file__).parents[2] / "result" / "c1_contrastive" / "property_embeddings.npy"
IDS_PATH = Path(__file__).parents[2] / "result" / "c1_contrastive" / "property_ids.npy"
OSM_PATH = Path(__file__).parents[2] / "dataset" / "external" / "osm_poi_features.parquet"

MIN_ACTIVITIES = 100   # min activities per destination
MIN_PROPERTIES = 10    # min properties with C1 embeddings per destination
CONT_DIM_BASE = 9
CONT_DIM_OSM = 11      # base + 2 OSM features
EMB_DIM = 16
CONT_DIM = CONT_DIM_BASE   # updated at runtime if OSM loaded
FEAT_DIM = CONT_DIM_BASE + EMB_DIM  # 25 base, 27 with OSM


def parse_primary_category(s):
    """Take first comma-separated token as primary category."""
    if pd.isna(s) or s == "":
        return "__unk__"
    return s.split(",")[0].strip()


class ActivityVocab:
    def __init__(self, categories: list):
        unique = sorted(set(categories))
        self.cat_map = {c: i + 1 for i, c in enumerate(unique)}  # 0 = unknown
        self.n_categories = len(self.cat_map)

    def encode(self, categories: list) -> np.ndarray:
        return np.array([self.cat_map.get(c, 0) for c in categories], dtype=np.int16)


def build_activity_features(df: pd.DataFrame) -> np.ndarray:
    """Returns (N, 9) continuous feature matrix."""
    price = df["price_in_aud"].values
    price_miss = np.isnan(price).astype(np.float32)
    price_norm = np.where(np.isnan(price), 0.0,
                          np.log1p(np.where(np.isnan(price), 0, price)) / 10.0).astype(np.float32)

    rating = (df["rating"].fillna(0).values / 10.0).astype(np.float32)

    dur = df["duration_hours"].values
    dur_median = float(np.nanmedian(dur))
    dur_miss = np.isnan(dur).astype(np.float32)
    dur_norm = np.where(np.isnan(dur), dur_median,
                        np.log1p(np.where(np.isnan(dur), 0, dur)) / 7.0).astype(np.float32)

    lat = df["lat"].values
    lat_miss = np.isnan(lat).astype(np.float32)
    lat_norm = (df["lat"].fillna(0).values / 90.0).astype(np.float32)
    lon_norm = (df["lon"].fillna(0).values / 180.0).astype(np.float32)

    avail = (df["availability_score"].fillna(0).values / 200.0).astype(np.float32)

    return np.stack([
        price_norm, price_miss,
        rating,
        dur_norm, dur_miss,
        lat_norm, lon_norm,
        avail,
        lat_miss,
    ], axis=1)  # (N, 9)


def load_cluster_embeddings(min_activities=MIN_ACTIVITIES, min_properties=MIN_PROPERTIES):
    """
    Returns:
      cluster_emb: dict[dest_name → mean C1 embedding (128,)]
      valid_dests: list of destination names passing both thresholds
    """
    print("Loading C1 embeddings...")
    embeddings = np.load(EMB_PATH, mmap_mode="r")   # (N_acc, 128)
    prop_ids = np.load(IDS_PATH, allow_pickle=True)  # (N_acc,)

    print("Loading accommodations (dest column)...")
    acc = pd.read_parquet(ACCOMMODATIONS, columns=["id", "destination_display_name"])
    acc["id_str"] = acc["id"].astype(str)
    id_to_dest = acc.set_index("id_str")["destination_display_name"].to_dict()

    # Map each embedding to its destination
    dest_embs = {}
    for i, pid in enumerate(prop_ids):
        dest = id_to_dest.get(str(pid))
        if dest is None or pd.isna(dest):
            continue
        if dest not in dest_embs:
            dest_embs[dest] = []
        dest_embs[dest].append(i)

    # Filter by min_properties
    print("Loading activities for dest filter...")
    act = pd.read_parquet(ACTIVITIES, columns=["destination_display_name"])
    act_counts = act["destination_display_name"].value_counts()

    cluster_emb = {}
    for dest, idxs in dest_embs.items():
        if len(idxs) < min_properties:
            continue
        if act_counts.get(dest, 0) < min_activities:
            continue
        mean_emb = embeddings[idxs].mean(axis=0)
        cluster_emb[dest] = mean_emb.astype(np.float32)

    print(f"Valid clusters: {len(cluster_emb)}")
    return cluster_emb


class ActivityDataset(Dataset):
    """
    Each item: (activity_cont (9 or 11,), activity_cat_idx (int), cluster_emb (128,))
    Only activities whose destination is in valid_clusters.
    If OSM features exist, cont_dim = 11 (adds poi_density_1km_norm, poi_density_5km_norm).
    """
    def __init__(self, cluster_emb: dict, use_osm: bool = True):
        print("Loading activities...")
        df = pd.read_parquet(ACTIVITIES)
        df = df[df["destination_display_name"].isin(cluster_emb)].reset_index(drop=True)
        print(f"Activities in valid clusters: {len(df):,}")

        self.cont = build_activity_features(df)  # (N, 9)

        # OSM enrichment — append if available and requested
        self.use_osm = False
        if use_osm and OSM_PATH.exists():
            print("Loading OSM POI features...")
            osm = pd.read_parquet(OSM_PATH)
            osm_cols = ["poi_density_1km_norm", "poi_density_5km_norm"]
            if all(c in osm.columns for c in osm_cols):
                df = df.merge(osm[["product_id"] + osm_cols], on="product_id", how="left")
                for c in osm_cols:
                    df[c] = df[c].fillna(0.0).astype(np.float32)
                osm_feats = df[osm_cols].values.astype(np.float32)
                self.cont = np.concatenate([self.cont, osm_feats], axis=1)
                self.use_osm = True
                print(f"OSM enriched: cont_dim = {self.cont.shape[1]}")
            else:
                print("OSM file found but missing expected columns — skipping.")
        else:
            print("No OSM features found — using base 9-dim features.")

        self.cont_dim = self.cont.shape[1]

        primary_cats = df["categories_as_string"].apply(parse_primary_category).tolist()
        self.vocab = ActivityVocab(primary_cats)
        self.cat_idx = self.vocab.encode(primary_cats)
        self.cluster_targets = np.stack(
            [cluster_emb[d] for d in df["destination_display_name"]], axis=0
        )  # (N, 128)
        self.dest_names = df["destination_display_name"].values
        self.activity_names = df["name"].values
        self.primary_cats_arr = np.array(primary_cats)

    def __len__(self):
        return len(self.cont)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.cont[idx]),
            torch.tensor(int(self.cat_idx[idx]), dtype=torch.long),
            torch.from_numpy(self.cluster_targets[idx]),
        )
