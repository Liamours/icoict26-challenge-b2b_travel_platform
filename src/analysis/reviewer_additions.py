from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

ROOT = Path(__file__).parents[2]
RESULT = ROOT / "result" / "analysis"
RESULT.mkdir(parents=True, exist_ok=True)
EMB_PATH = ROOT / "result" / "c1_contrastive" / "property_embeddings.npy"
IDS_PATH = ROOT / "result" / "c1_contrastive" / "property_ids.npy"
PARQUET = ROOT / "dataset" / "preprocessed" / "accommodations.parquet"

SEED = 42
N_PER_PROVIDER = 1_000
N_BOOT = 30
K_VALUES = [1, 5, 10, 20, 50]


def build_continuous_features(df):
    star = df["star_rating"].values
    star_miss = np.isnan(star).astype(np.float32)
    star_norm = np.where(np.isnan(star), 0.0, star / 5.0).astype(np.float32)

    guest = df["guest_rating"].values
    guest_miss = np.isnan(guest).astype(np.float32)
    guest_norm = np.where(np.isnan(guest), 0.0, guest / 10.0).astype(np.float32)

    star_score = (df["star_rating_score"].fillna(0).values / 7.0).astype(np.float32)
    pop_score = (df["popularity_score"].fillna(0).values / 102.0).astype(np.float32)
    avail = (df["availability_score"].values / 200.0).astype(np.float32)

    price = df["price_in_aud"].values
    price_miss = np.isnan(price).astype(np.float32)
    price_norm = np.where(np.isnan(price), 0.0, np.log1p(np.where(np.isnan(price), 0, price)) / 15.0).astype(np.float32)

    lat_norm = (df["lat"].fillna(0).values / 90.0).astype(np.float32)
    lon_norm = (df["lon"].fillna(0).values / 180.0).astype(np.float32)

    return np.stack([
        star_norm, star_miss, guest_norm, guest_miss, star_score, pop_score,
        avail, price_norm, price_miss, lat_norm, lon_norm,
    ], axis=1)


def sample_balanced(df, rng, n_per_provider):
    indices = []
    for provider in sorted(df["provider"].dropna().unique()):
        pool = df.index[df["provider"] == provider].to_numpy()
        indices.append(rng.choice(pool, min(n_per_provider, len(pool)), replace=False))
    return np.concatenate(indices)


def load_embedding_index():
    all_ids = np.load(IDS_PATH, allow_pickle=True)
    id_to_idx = {str(pid): i for i, pid in enumerate(all_ids)}
    all_emb = np.load(EMB_PATH, mmap_mode="r")
    return all_emb, id_to_idx


def load_embeddings_for_ids(ids, all_emb, id_to_idx):
    rows = []
    keep = []
    for pid in ids:
        idx = id_to_idx.get(str(pid))
        keep.append(idx is not None)
        rows.append(all_emb[idx] if idx is not None else np.zeros(128, dtype=np.float32))
    return np.stack(rows), np.array(keep, dtype=bool)


def mean_dim_ks(X, providers):
    vals = []
    unique = sorted(pd.unique(providers))
    for d in range(X.shape[1]):
        for i, p in enumerate(unique):
            xp = X[providers == p, d]
            for q in unique[i + 1:]:
                vals.append(ks_2samp(xp, X[providers == q, d]).statistic)
    return float(np.mean(vals))


def pooled_ks(X, providers):
    vals = []
    unique = sorted(pd.unique(providers))
    for i, p in enumerate(unique):
        xp = X[providers == p].ravel()
        for q in unique[i + 1:]:
            vals.append(ks_2samp(xp, X[providers == q].ravel()).statistic)
    return float(np.mean(vals))


def cp_rate(X_normed, providers, k):
    sim = cosine_similarity(X_normed)
    np.fill_diagonal(sim, -np.inf)
    top = np.argpartition(sim, -k, axis=1)[:, -k:]
    return float(np.mean([(providers[top[i]] != providers[i]).mean() for i in range(len(providers))]))


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
    if len(pooled) == 0:
        out["star_rating_calibrated"] = out["star_rating"]
        return out
    pooled_sorted = np.sort(pooled)
    out["star_rating_calibrated"] = out["star_rating"]
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
        out.loc[idx, "star_rating_calibrated"] = calibrated
    return out


def build_calibrated_continuous_features(df):
    tmp = df.copy()
    tmp["star_rating"] = tmp["star_rating_calibrated"]
    return build_continuous_features(tmp)


def cluster_stats(df):
    tiers = np.floor(df["star_rating"].fillna(-1)).clip(-1, 4).astype(int)
    tmp = pd.DataFrame({
        "tier": tiers,
        "dest": df["destination_display_name"].fillna("__unk__").values,
    })
    tmp = tmp[tmp["tier"] >= 0]
    tmp["key"] = tmp["dest"] + "||" + tmp["tier"].astype(str)
    sizes_all = tmp.groupby("key").size()
    sizes_valid = sizes_all[sizes_all >= 5]
    covered = int(sizes_valid.sum())
    rows = [{
        "min_cluster_size": 5,
        "clusters_all": int(len(sizes_all)),
        "clusters_valid": int(len(sizes_valid)),
        "properties_total": int(len(df)),
        "properties_tiered": int(len(tmp)),
        "properties_covered": covered,
        "properties_excluded": int(len(df) - covered),
        "coverage_pct": covered / len(df) * 100,
        "median_cluster_size": float(sizes_valid.median()),
        "p90_cluster_size": float(sizes_valid.quantile(0.90)),
        "p99_cluster_size": float(sizes_valid.quantile(0.99)),
        "max_cluster_size": int(sizes_valid.max()),
    }]
    pd.DataFrame(rows).to_csv(RESULT / "cluster_stats.csv", index=False)


def plot_outputs():
    BLUE = "#1B4F8A"
    RED = "#C0392B"
    GRAY = "#A8A8A8"

    sens = pd.read_csv(RESULT / "cp_nn_k_sensitivity.csv")
    labels = {
        "raw_features_13d": "Raw 13D",
        "star_calibrated_raw_13d": "Star calibration",
        "contrastive_embedding": "Contrastive",
    }
    colors = {
        "raw_features_13d": GRAY,
        "star_calibrated_raw_13d": BLUE,
        "contrastive_embedding": RED,
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for space in ["raw_features_13d", "star_calibrated_raw_13d", "contrastive_embedding"]:
        g = sens[sens["space"] == space]
        ax.plot(g["k"], g["cp_nn_rate"], marker="o", linewidth=1.8,
                color=colors[space], label=labels[space])
    ax.axhline(0.75, color=RED, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("K neighbours")
    ax.set_ylabel("Cross-provider NN rate")
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT / "cp_nn_k_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = pd.read_csv(RESULT / "reviewer_bootstrap_summary.csv")
    ks = summary[summary["metric"] == "ks_per_dimension"].set_index("space")
    cp = summary[summary["metric"] == "cp_nn_k10_resampled"].set_index("space")
    methods = ["raw_features_11d", "star_calibrated_raw_11d", "contrastive_embedding"]
    method_labels = ["Raw", "Star calib.", "Contrastive"]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ks_means = [ks.loc[m, "mean"] for m in methods]
    ks_err = [[ks.loc[m, "mean"] - ks.loc[m, "ci_lo"] for m in methods],
              [ks.loc[m, "ci_hi"] - ks.loc[m, "mean"] for m in methods]]
    axes[0].bar(x, ks_means, yerr=ks_err, capsize=3,
                color=[GRAY, BLUE, RED], edgecolor="#444", linewidth=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_labels, rotation=15, ha="right")
    axes[0].set_ylabel("Per-dim KS")

    cp_methods = ["raw_features_13d", "star_calibrated_raw_13d", "contrastive_embedding"]
    cp_means = [cp.loc[m, "mean"] for m in cp_methods]
    cp_err = [[cp.loc[m, "mean"] - cp.loc[m, "ci_lo"] for m in cp_methods],
              [cp.loc[m, "ci_hi"] - cp.loc[m, "mean"] for m in cp_methods]]
    axes[1].bar(x, cp_means, yerr=cp_err, capsize=3,
                color=[GRAY, BLUE, RED], edgecolor="#444", linewidth=0.6)
    axes[1].axhline(0.75, color=RED, linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_labels, rotation=15, ha="right")
    axes[1].set_ylabel("CP-NN@10")
    axes[1].set_ylim(0, 0.8)
    fig.tight_layout()
    fig.savefig(RESULT / "reviewer_robustness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ab_labels = ["A", "B", "C"]
    quality = [0.862, 0.996, 0.995]
    cp_vals = [0.167, 0.583, 0.591]
    axes[0].bar(np.arange(3), quality, color=[GRAY, BLUE, RED], edgecolor="#444", linewidth=0.6)
    axes[0].set_xticks(np.arange(3))
    axes[0].set_xticklabels(ab_labels)
    axes[0].set_ylim(0.80, 1.02)
    axes[0].set_ylabel("Quality tier accuracy")
    axes[1].bar(np.arange(3), cp_vals, color=[GRAY, BLUE, RED], edgecolor="#444", linewidth=0.6)
    axes[1].axhline(0.75, color=RED, linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].set_xticks(np.arange(3))
    axes[1].set_xticklabels(ab_labels)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Cross-provider NN rate")
    fig.tight_layout()
    fig.savefig(RESULT / "ablation_consistent.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(PARQUET).reset_index(drop=True)
    cluster_stats(df)
    all_emb, id_to_idx = load_embedding_index()

    idx = sample_balanced(df, rng, N_PER_PROVIDER)
    sub = df.iloc[idx].reset_index(drop=True)
    emb, keep = load_embeddings_for_ids(sub["id"].astype(str).values, all_emb, id_to_idx)
    sub = sub.loc[keep].reset_index(drop=True)
    emb = emb[keep]
    providers = sub["provider"].to_numpy()

    raw = build_continuous_features(sub)
    sub_cal = quantile_calibrate_star(sub)
    cal = build_calibrated_continuous_features(sub_cal)
    raw_13 = append_category_country(sub, raw)
    cal_13 = append_category_country(sub_cal, cal)

    spaces = {
        "raw_features_13d": normalize(raw_13, norm="l2"),
        "star_calibrated_raw_13d": normalize(cal_13, norm="l2"),
        "contrastive_embedding": normalize(emb, norm="l2"),
    }

    sens_rows = []
    for name, X in spaces.items():
        for k in K_VALUES:
            sens_rows.append({"space": name, "k": k, "cp_nn_rate": cp_rate(X, providers, k)})
    pd.DataFrame(sens_rows).to_csv(RESULT / "cp_nn_k_sensitivity.csv", index=False)

    cp_rows = []
    for r in range(N_BOOT):
        ridx = sample_balanced(df, rng, N_PER_PROVIDER)
        rsub = df.iloc[ridx].reset_index(drop=True)
        remb, rkeep = load_embeddings_for_ids(rsub["id"].astype(str).values, all_emb, id_to_idx)
        rsub = rsub.loc[rkeep].reset_index(drop=True)
        remb = remb[rkeep]
        rproviders = rsub["provider"].to_numpy()
        rraw = build_continuous_features(rsub)
        rcal = build_calibrated_continuous_features(quantile_calibrate_star(rsub))
        rraw_13 = append_category_country(rsub, rraw)
        rcal_13 = append_category_country(rsub, rcal)
        for name, X in {
            "raw_features_13d": rraw_13,
            "star_calibrated_raw_13d": rcal_13,
            "contrastive_embedding": remb,
        }.items():
            cp_rows.append({
                "resample": r,
                "space": name,
                "cp_nn_k10": cp_rate(normalize(X, norm="l2"), rproviders, 10),
            })
    cp_resampled = pd.DataFrame(cp_rows)
    cp_resampled.to_csv(RESULT / "cp_nn_resampled_raw.csv", index=False)

    boot_rows = []
    for b in range(N_BOOT):
        boot_idx = []
        for provider in sorted(pd.unique(providers)):
            pool = np.where(providers == provider)[0]
            boot_idx.append(rng.choice(pool, len(pool), replace=True))
        bidx = np.concatenate(boot_idx)
        bprov = providers[bidx]
        for name, X in {
            "raw_features_11d": raw,
            "star_calibrated_raw_11d": cal,
            "contrastive_embedding": emb,
        }.items():
            Xb = X[bidx]
            boot_rows.append({
                "bootstrap": b,
                "space": name,
                "ks_per_dimension": mean_dim_ks(Xb, bprov),
                "ks_pooled": pooled_ks(Xb, bprov),
            })
        for name, X in {
            "raw_features_13d": raw_13,
            "star_calibrated_raw_13d": cal_13,
            "contrastive_embedding": emb,
        }.items():
            Xb = X[bidx]
            boot_rows.append({
                "bootstrap": b,
                "space": name,
                "cp_nn_k10": cp_rate(normalize(Xb, norm="l2"), bprov, 10),
            })
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(RESULT / "reviewer_bootstrap_raw.csv", index=False)

    summary = []
    for (space), g in boot.groupby("space"):
        for metric in ["ks_per_dimension", "ks_pooled", "cp_nn_k10"]:
            vals = g[metric].dropna().to_numpy()
            if len(vals) == 0:
                continue
            summary.append({
                "space": space,
                "metric": metric,
                "mean": float(vals.mean()),
                "ci_lo": float(np.quantile(vals, 0.025)),
                "ci_hi": float(np.quantile(vals, 0.975)),
                "n_boot": N_BOOT,
            })
    for space, g in cp_resampled.groupby("space"):
        vals = g["cp_nn_k10"].to_numpy()
        summary.append({
            "space": space,
            "metric": "cp_nn_k10_resampled",
            "mean": float(vals.mean()),
            "ci_lo": float(np.quantile(vals, 0.025)),
            "ci_hi": float(np.quantile(vals, 0.975)),
            "n_boot": N_BOOT,
        })
    pd.DataFrame(summary).to_csv(RESULT / "reviewer_bootstrap_summary.csv", index=False)
    plot_outputs()
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
