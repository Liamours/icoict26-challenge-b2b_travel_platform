"""
Additional representation-level validation for reviewer concerns.

This script does not retrain models. It strengthens the evidence around:
  - residual provider signal via proxy provider classifiers,
  - distribution alignment via MMD and Wasserstein distance,
  - CP-NN sensitivity under cosine-distance thresholds,
  - positive-pair cluster coverage under different sparsity thresholds,
  - raw-feature provider signal via permutation importance.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize


ROOT = Path(__file__).parents[2]
RESULT = ROOT / "result" / "analysis" / "validation_suite"
RESULT.mkdir(parents=True, exist_ok=True)

ACCOM = ROOT / "dataset" / "preprocessed" / "accommodations.parquet"
EMB = ROOT / "result" / "c1_contrastive" / "property_embeddings.npy"
IDS = ROOT / "result" / "c1_contrastive" / "property_ids.npy"

SEED = 42
N_PER_PROVIDER = 1_000
N_MMD_PER_PROVIDER = 500
N_FEATURE_IMPORTANCE = 20_000
K = 10
DISTANCE_THRESHOLDS = [0.05, 0.10, 0.20, 0.40]
CLUSTER_THRESHOLDS = [2, 5, 10, 20, 50]

FEATURE_NAMES = [
    "star_norm", "star_missing", "guest_norm", "guest_missing",
    "star_score", "popularity_score", "availability_score",
    "price_norm", "price_missing", "lat_norm", "lon_norm",
]


def log(msg):
    print(msg, flush=True)


def build_continuous_features(df):
    star = df["star_rating"].to_numpy()
    guest = df["guest_rating"].to_numpy()
    price = df["price_in_aud"].to_numpy()

    star_missing = np.isnan(star).astype(np.float32)
    guest_missing = np.isnan(guest).astype(np.float32)
    price_missing = np.isnan(price).astype(np.float32)

    star_norm = np.where(np.isnan(star), 0.0, star / 5.0).astype(np.float32)
    guest_norm = np.where(np.isnan(guest), 0.0, guest / 10.0).astype(np.float32)
    price_norm = np.where(np.isnan(price), 0.0, np.log1p(np.where(np.isnan(price), 0, price)) / 15.0).astype(np.float32)

    return np.stack([
        star_norm,
        star_missing,
        guest_norm,
        guest_missing,
        (df["star_rating_score"].fillna(0).to_numpy() / 7.0).astype(np.float32),
        (df["popularity_score"].fillna(0).to_numpy() / 102.0).astype(np.float32),
        (df["availability_score"].to_numpy() / 200.0).astype(np.float32),
        price_norm,
        price_missing,
        (df["lat"].fillna(0).to_numpy() / 90.0).astype(np.float32),
        (df["lon"].fillna(0).to_numpy() / 180.0).astype(np.float32),
    ], axis=1)


def append_category_country(df, continuous):
    category_map = {v: i + 1 for i, v in enumerate(sorted(df["category"].dropna().unique()))}
    top_countries = df["country"].value_counts().head(300).index.tolist()
    country_map = {v: i + 1 for i, v in enumerate(top_countries)}
    category = df["category"].map(category_map).fillna(0).to_numpy(dtype=np.float32)
    country = df["country"].map(country_map).fillna(0).to_numpy(dtype=np.float32)
    category = (category / max(len(category_map), 1)).reshape(-1, 1)
    country = (country / max(len(country_map), 1)).reshape(-1, 1)
    return np.concatenate([continuous, category, country], axis=1)


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


def load_balanced_sample():
    rng = np.random.default_rng(SEED)
    cols = [
        "id", "provider", "category", "country", "destination_display_name",
        "star_rating", "guest_rating", "star_rating_score", "popularity_score",
        "availability_score", "price_in_aud", "lat", "lon",
    ]
    df = pd.read_parquet(ACCOM, columns=cols).reset_index(drop=True)
    idx = []
    for provider in sorted(df["provider"].dropna().unique()):
        pool = df.index[df["provider"] == provider].to_numpy()
        idx.append(rng.choice(pool, min(N_PER_PROVIDER, len(pool)), replace=False))
    sample = df.iloc[np.concatenate(idx)].reset_index(drop=True)
    return df, sample


def load_embeddings_for_sample(sample):
    all_ids = np.load(IDS, allow_pickle=True)
    id_to_idx = {str(pid): i for i, pid in enumerate(all_ids)}
    all_emb = np.load(EMB, mmap_mode="r")
    rows, keep = [], []
    for pid in sample["id"].astype(str).to_numpy():
        idx = id_to_idx.get(pid)
        keep.append(idx is not None)
        rows.append(all_emb[idx] if idx is not None else np.zeros(128, dtype=np.float32))
    keep = np.array(keep, dtype=bool)
    return sample.loc[keep].reset_index(drop=True), np.stack(rows)[keep]


def make_spaces(sample, emb):
    raw_11 = build_continuous_features(sample)
    raw_13 = append_category_country(sample, raw_11)
    calibrated = quantile_calibrate_star(sample)
    calibrated["star_rating"] = calibrated["star_calibrated"]
    cal_11 = build_continuous_features(calibrated)
    cal_13 = append_category_country(calibrated, cal_11)
    return {
        "raw_11d": raw_11,
        "raw_13d": raw_13,
        "calibrated_11d": cal_11,
        "calibrated_13d": cal_13,
        "embedding_128d": emb,
    }


def proxy_provider_classifier(spaces, providers):
    y = LabelEncoder().fit_transform(providers)
    rows = []
    for name, X in spaces.items():
        Xn = normalize(X, norm="l2") if name != "raw_11d" and name != "calibrated_11d" else X
        Xtr, Xte, ytr, yte = train_test_split(Xn, y, test_size=0.25, random_state=SEED, stratify=y)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)
        rows.append({
            "space": name,
            "provider_accuracy": accuracy_score(yte, pred),
            "provider_macro_auc_ovr": roc_auc_score(yte, proba, multi_class="ovr", average="macro"),
            "n_train": len(ytr),
            "n_test": len(yte),
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULT / "proxy_provider_classifier.csv", index=False)
    return out


def median_gamma(X):
    d = pdist(X, metric="sqeuclidean")
    med = np.median(d[d > 0])
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return 1.0 / med


def mmd2_unbiased(X, Y, gamma):
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    nx, ny = len(X), len(Y)
    return Kxx.sum() / (nx * (nx - 1)) + Kyy.sum() / (ny * (ny - 1)) - 2 * Kxy.mean()


def distribution_alignment(spaces, providers):
    rng = np.random.default_rng(SEED)
    rows = []
    unique = sorted(pd.unique(providers))
    for name, X in spaces.items():
        Xn = normalize(X, norm="l2") if "embedding" in name else X
        for i, p in enumerate(unique):
            xp_all = Xn[providers == p]
            xp = xp_all[rng.choice(len(xp_all), min(N_MMD_PER_PROVIDER, len(xp_all)), replace=False)]
            for q in unique[i + 1:]:
                xq_all = Xn[providers == q]
                xq = xq_all[rng.choice(len(xq_all), min(N_MMD_PER_PROVIDER, len(xq_all)), replace=False)]
                gamma = median_gamma(np.vstack([xp, xq]))
                wd = np.mean([wasserstein_distance(xp[:, d], xq[:, d]) for d in range(Xn.shape[1])])
                rows.append({
                    "space": name,
                    "provider_pair": f"{p} vs {q}",
                    "mmd2_rbf": mmd2_unbiased(xp, xq, gamma),
                    "wasserstein_mean_dim": wd,
                    "gamma": gamma,
                    "n_per_provider": len(xp),
                })
    out = pd.DataFrame(rows)
    out.to_csv(RESULT / "distribution_alignment.csv", index=False)
    summary = (
        out.groupby("space", as_index=False)
        .agg(mean_mmd2=("mmd2_rbf", "mean"), mean_wasserstein=("wasserstein_mean_dim", "mean"))
        .sort_values("mean_mmd2")
    )
    summary.to_csv(RESULT / "distribution_alignment_summary.csv", index=False)
    return summary


def cpnn_threshold_sensitivity(spaces, providers):
    rows = []
    for name in ["raw_13d", "calibrated_13d", "embedding_128d"]:
        X = normalize(spaces[name], norm="l2")
        sim = cosine_similarity(X)
        np.fill_diagonal(sim, -np.inf)
        top = np.argpartition(sim, -K, axis=1)[:, -K:]
        for threshold in DISTANCE_THRESHOLDS:
            kept = cross = 0
            for i in range(len(providers)):
                nbrs = top[i]
                distances = 1.0 - sim[i, nbrs]
                selected = nbrs[distances <= threshold]
                kept += len(selected)
                cross += int(np.sum(providers[selected] != providers[i]))
            rows.append({
                "space": name,
                "k": K,
                "cosine_distance_threshold": threshold,
                "neighbour_pairs_kept": kept,
                "coverage_per_query": kept / len(providers),
                "cp_rate": np.nan if kept == 0 else cross / kept,
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULT / "cpnn_threshold_sensitivity.csv", index=False)
    return out


def cluster_threshold_sensitivity(df):
    tiers = np.floor(df["star_rating"].fillna(-1)).clip(-1, 4).astype(int)
    tmp = pd.DataFrame({
        "tier": tiers,
        "dest": df["destination_display_name"].fillna("__unk__").to_numpy(),
    })
    tmp = tmp[tmp["tier"] >= 0]
    tmp["key"] = tmp["dest"] + "||" + tmp["tier"].astype(str)
    sizes = tmp.groupby("key").size()
    rows = []
    for threshold in CLUSTER_THRESHOLDS:
        valid = sizes[sizes >= threshold]
        rows.append({
            "min_cluster_size": threshold,
            "valid_clusters": len(valid),
            "properties_covered": int(valid.sum()),
            "coverage_pct": valid.sum() / len(df) * 100,
            "median_cluster_size": float(valid.median()) if len(valid) else np.nan,
            "p90_cluster_size": float(valid.quantile(0.90)) if len(valid) else np.nan,
            "max_cluster_size": int(valid.max()) if len(valid) else 0,
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULT / "cluster_threshold_sensitivity.csv", index=False)
    return out


def feature_provider_importance(df):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), min(N_FEATURE_IMPORTANCE, len(df)), replace=False)
    sub = df.iloc[idx].reset_index(drop=True)
    X = build_continuous_features(sub)
    y = LabelEncoder().fit_transform(sub["provider"].to_numpy())
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    result = permutation_importance(clf, Xte, yte, n_repeats=10, random_state=SEED, scoring="accuracy")
    out = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    out.to_csv(RESULT / "raw_feature_provider_importance.csv", index=False)
    return out


def main():
    log("Loading balanced sample and embeddings...")
    full_df, sample = load_balanced_sample()
    sample, emb = load_embeddings_for_sample(sample)
    providers = sample["provider"].to_numpy()
    spaces = make_spaces(sample, emb)
    log(f"Sample rows with embeddings: {len(sample):,}")

    log("Running proxy provider classifiers...")
    print(proxy_provider_classifier(spaces, providers).to_string(index=False), flush=True)

    log("Running distribution alignment metrics...")
    print(distribution_alignment(spaces, providers).to_string(index=False), flush=True)

    log("Running CP-NN threshold sensitivity...")
    print(cpnn_threshold_sensitivity(spaces, providers).to_string(index=False), flush=True)

    log("Running cluster-threshold sensitivity...")
    print(cluster_threshold_sensitivity(full_df).to_string(index=False), flush=True)

    log("Running raw-feature provider permutation importance...")
    print(feature_provider_importance(full_df).head(10).to_string(index=False), flush=True)

    log(f"Saved outputs in {RESULT}")


if __name__ == "__main__":
    main()
