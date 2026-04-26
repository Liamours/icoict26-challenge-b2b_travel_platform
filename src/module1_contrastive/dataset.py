import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm


PARQUET = Path(__file__).parents[2] / "dataset" / "preprocessed" / "accommodations.parquet"


class AccommodationVocab:
    def __init__(self, df: pd.DataFrame):
        self.provider_map = {v: i + 1 for i, v in enumerate(sorted(df["provider"].dropna().unique()))}
        self.category_map = {v: i + 1 for i, v in enumerate(sorted(df["category"].dropna().unique()))}
        top_countries = df["country"].value_counts().head(300).index.tolist()
        self.country_map = {v: i + 1 for i, v in enumerate(top_countries)}

        self.n_providers = len(self.provider_map)     # 4
        self.n_categories = len(self.category_map)
        self.n_countries = len(self.country_map)      # capped at 300

    def encode(self, df: pd.DataFrame):
        provider_idx = df["provider"].map(self.provider_map).fillna(0).astype(np.int16).values
        category_idx = df["category"].map(self.category_map).fillna(0).astype(np.int16).values
        country_idx = df["country"].map(self.country_map).fillna(0).astype(np.int16).values
        return provider_idx, category_idx, country_idx


def build_continuous_features(df: pd.DataFrame) -> np.ndarray:
    star = df["star_rating"].values
    star_miss = np.isnan(star).astype(np.float32)
    star_norm = np.where(np.isnan(star), 0.0, star / 5.0).astype(np.float32)   # range [0,5]

    guest = df["guest_rating"].values
    guest_miss = np.isnan(guest).astype(np.float32)
    guest_norm = np.where(np.isnan(guest), 0.0, guest / 10.0).astype(np.float32)  # range [0,10]

    star_score = (df["star_rating_score"].fillna(0).values / 7.0).astype(np.float32)   # observed max=7
    pop_score = (df["popularity_score"].fillna(0).values / 102.0).astype(np.float32)   # observed max=102
    avail = (df["availability_score"].values / 200.0).astype(np.float32)               # range [0,200]

    price = df["price_in_aud"].values
    price_miss = np.isnan(price).astype(np.float32)
    # log1p then /15 maps 99th-pct price (~AUD 3.3M) to ~1.2; keeps outlier impact bounded
    price_norm = np.where(np.isnan(price), 0.0, np.log1p(np.where(np.isnan(price), 0, price)) / 15.0).astype(np.float32)

    lat_norm = (df["lat"].fillna(0).values / 90.0).astype(np.float32)    # range [-90,90]
    lon_norm = (df["lon"].fillna(0).values / 180.0).astype(np.float32)   # range [-180,180]

    return np.stack([
        star_norm, star_miss,
        guest_norm, guest_miss,
        star_score, pop_score, avail,
        price_norm, price_miss,
        lat_norm, lon_norm,
    ], axis=1)  # (N, 11)


class AccommodationPairDataset(Dataset):
    def __init__(
        self,
        parquet_path: Path = PARQUET,
        n_pairs_per_epoch: int = 100_000,
        min_cluster_size: int = 5,
        p_provider_mask: float = 0.5,
        p_feature_drop: float = 0.2,
        noise_std: float = 0.05,
    ):
        self.n_pairs = n_pairs_per_epoch
        self.p_mask = p_provider_mask
        self.p_drop = p_feature_drop
        self.noise = noise_std

        print("Loading accommodations parquet...")
        df = pd.read_parquet(parquet_path)
        df = df.reset_index(drop=True)

        self.vocab = AccommodationVocab(df)
        self.cont = build_continuous_features(df)
        self.provider_idx, self.category_idx, self.country_idx = self.vocab.encode(df)

        self._build_clusters(df, min_cluster_size)

    def _build_clusters(self, df: pd.DataFrame, min_size: int):
        print("Building destination-quality clusters...")
        tiers = np.floor(df["star_rating"].fillna(-1)).clip(-1, 4).astype(int)
        dest = df["destination_display_name"].fillna("__unk__").values

        tmp = pd.DataFrame({
            "pos": np.arange(len(df)),
            "tier": tiers,
            "dest": dest,
        })
        tmp = tmp[tmp["tier"] >= 0]
        tmp["key"] = tmp["dest"] + "||" + tmp["tier"].astype(str)

        sizes = tmp.groupby("key").size()
        valid_keys = sizes[sizes >= min_size].index
        tmp = tmp[tmp["key"].isin(valid_keys)]

        groups = tmp.groupby("key")["pos"].apply(np.array)
        self.valid_clusters = groups.to_dict()
        self.cluster_keys = list(self.valid_clusters.keys())

        n_props = sum(len(v) for v in self.valid_clusters.values())
        print(f"Valid clusters: {len(self.cluster_keys):,} | Properties covered: {n_props:,}")

    def _augment(self, cont: np.ndarray, prov_idx: int):
        c = cont.copy()
        n_cont = 9  # first 9 dims are numeric (not mask flags)
        c[:n_cont] += np.random.randn(n_cont).astype(np.float32) * self.noise
        drop = np.random.rand(n_cont) < self.p_drop
        c[:n_cont] = np.where(drop, 0.0, c[:n_cont])
        p = np.int16(0) if np.random.rand() < self.p_mask else prov_idx
        return c, p

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, _):
        key = self.cluster_keys[np.random.randint(len(self.cluster_keys))]
        pool = self.valid_clusters[key]
        if len(pool) < 2:
            i = j = pool[0]
        else:
            i, j = pool[np.random.choice(len(pool), size=2, replace=False)]

        c1, p1 = self._augment(self.cont[i], self.provider_idx[i])
        c2, p2 = self._augment(self.cont[j], self.provider_idx[j])

        return {
            "cont1": torch.from_numpy(c1),
            "provider1": torch.tensor(int(p1), dtype=torch.long),
            "category1": torch.tensor(int(self.category_idx[i]), dtype=torch.long),
            "country1": torch.tensor(int(self.country_idx[i]), dtype=torch.long),
            "cont2": torch.from_numpy(c2),
            "provider2": torch.tensor(int(p2), dtype=torch.long),
            "category2": torch.tensor(int(self.category_idx[j]), dtype=torch.long),
            "country2": torch.tensor(int(self.country_idx[j]), dtype=torch.long),
            "true_provider1": torch.tensor(int(self.provider_idx[i]), dtype=torch.long),
            "true_provider2": torch.tensor(int(self.provider_idx[j]), dtype=torch.long),
        }

    def get_full_batch(self, indices: np.ndarray, device: torch.device, mask_provider: bool = True):
        """Return feature tensors for a set of indices (used in generate_all_embeddings)."""
        cont = torch.from_numpy(self.cont[indices]).to(device)
        prov = torch.zeros(len(indices), dtype=torch.long, device=device) if mask_provider \
            else torch.from_numpy(self.provider_idx[indices].astype(np.int64)).to(device)
        cat = torch.from_numpy(self.category_idx[indices].astype(np.int64)).to(device)
        cntry = torch.from_numpy(self.country_idx[indices].astype(np.int64)).to(device)
        return cont, prov, cat, cntry
