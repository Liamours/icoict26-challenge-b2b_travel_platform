import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parents[1] / "module1_contrastive"))
from dataset import AccommodationVocab, build_continuous_features, PARQUET

RESULT = Path(__file__).parents[2] / "result" / "analysis"
RESULT.mkdir(parents=True, exist_ok=True)
EMB_PATH = Path(__file__).parents[2] / "result" / "c1_contrastive" / "property_embeddings.npy"
IDS_PATH = Path(__file__).parents[2] / "result" / "c1_contrastive" / "property_ids.npy"

N_PER_PROVIDER = 2_000
SEED = 42

FEATURE_NAMES = [
    "star_norm", "star_miss", "guest_norm", "guest_miss",
    "star_score", "pop_score", "avail",
    "price_norm", "price_miss", "lat_norm", "lon_norm",
]


def mean_ks(data: dict) -> float:
    keys = list(data.keys())
    vals = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            e1 = data[keys[i]].flatten()
            e2 = data[keys[j]].flatten()
            vals.append(ks_2samp(e1, e2).statistic)
    return float(np.mean(vals))


def per_pair_ks(data: dict) -> dict:
    keys = list(data.keys())
    pairs = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k = f"{keys[i]} vs {keys[j]}"
            e1 = data[keys[i]].flatten()
            e2 = data[keys[j]].flatten()
            pairs[k] = ks_2samp(e1, e2).statistic
    return pairs


def main():
    rng = np.random.default_rng(SEED)

    print("Loading accommodations...")
    df = pd.read_parquet(PARQUET)
    df = df.reset_index(drop=True)
    vocab = AccommodationVocab(df)

    sampled = []
    for prov in sorted(df["provider"].dropna().unique()):
        pool = df.index[df["provider"] == prov].to_numpy()
        n = min(N_PER_PROVIDER, len(pool))
        sampled.append(df.iloc[rng.choice(pool, n, replace=False)].copy())
    sub = pd.concat(sampled).reset_index(drop=True)
    provider_labels = sub["provider"].values
    print(f"Sample: {len(sub):,}")

    cont = build_continuous_features(sub)

    print("Loading embeddings...")
    all_emb = np.load(EMB_PATH, mmap_mode="r")
    all_ids = np.load(IDS_PATH, allow_pickle=True)
    id_to_idx = {str(pid): i for i, pid in enumerate(all_ids)}
    sub_ids = sub["id"].astype(str).values

    emb_rows = []
    valid_mask = []
    for pid in sub_ids:
        idx = id_to_idx.get(pid)
        if idx is not None:
            emb_rows.append(all_emb[idx])
            valid_mask.append(True)
        else:
            emb_rows.append(np.zeros(128, dtype=np.float32))
            valid_mask.append(False)
    valid_mask = np.array(valid_mask)
    embeddings = np.stack(emb_rows, axis=0)

    sub_valid = sub[valid_mask].reset_index(drop=True)
    cont_valid = cont[valid_mask]
    emb_valid = embeddings[valid_mask]
    prov_valid = provider_labels[valid_mask]
    print(f"Valid embedding matches: {valid_mask.sum():,}")

    providers = sorted(np.unique(prov_valid))

    raw_by_prov = {p: cont_valid[prov_valid == p] for p in providers}
    emb_by_prov = {p: normalize(emb_valid[prov_valid == p], norm="l2") for p in providers}

    # Per-feature KS on raw
    feat_ks = {}
    for fi, fname in enumerate(FEATURE_NAMES):
        feat_data = {p: raw_by_prov[p][:, fi] for p in providers}
        feat_ks[fname] = mean_ks(feat_data)

    raw_ks_mean = float(np.mean(list(feat_ks.values())))
    emb_ks_mean = mean_ks(emb_by_prov)

    print(f"\nMean KS per feature (raw):")
    for f, v in sorted(feat_ks.items(), key=lambda x: -x[1]):
        print(f"  {f:20s}: {v:.4f}")
    print(f"\nMean KS across all raw features: {raw_ks_mean:.4f}")
    print(f"Mean KS on C1 embeddings:         {emb_ks_mean:.4f}")
    print(f"Reduction: {raw_ks_mean:.4f} → {emb_ks_mean:.4f}  ({(1-emb_ks_mean/raw_ks_mean)*100:.1f}% reduction)")

    pair_raw = per_pair_ks({p: raw_by_prov[p].mean(axis=1, keepdims=True) for p in providers})
    pair_emb = per_pair_ks(emb_by_prov)

    rows = []
    for pair in pair_raw:
        rows.append({"pair": pair, "ks_raw": pair_raw[pair], "ks_emb": pair_emb[pair]})
    df_ks = pd.DataFrame(rows)
    df_ks["reduction_pct"] = (1 - df_ks["ks_emb"] / df_ks["ks_raw"]) * 100
    df_ks.loc[len(df_ks)] = {
        "pair": "MEAN", "ks_raw": raw_ks_mean,
        "ks_emb": emb_ks_mean,
        "reduction_pct": (1 - emb_ks_mean / raw_ks_mean) * 100,
    }
    df_ks.to_csv(RESULT / "ks_comparison.csv", index=False)
    print(f"\nSaved ks_comparison.csv")

    # Figure: per-feature KS + provider-pair comparison
    BLUE = '#1B4F8A'
    RED  = '#C0392B'

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    feat_sorted = sorted(feat_ks.items(), key=lambda x: -x[1])
    axes[0].barh([f[0] for f in feat_sorted], [f[1] for f in feat_sorted],
                 color=BLUE, alpha=0.85)
    axes[0].axvline(emb_ks_mean, color=RED, linestyle="--", linewidth=2,
                    label=f"KS = {emb_ks_mean:.3f}")
    axes[0].set_xlabel("KS Statistic")
    axes[0].legend(fontsize=9)
    axes[0].tick_params(labelsize=8)

    pairs_plot = df_ks[df_ks["pair"] != "MEAN"]
    x = np.arange(len(pairs_plot))
    w = 0.35
    axes[1].bar(x - w/2, pairs_plot["ks_raw"], w, label="Raw", color=BLUE, alpha=0.85)
    axes[1].bar(x + w/2, pairs_plot["ks_emb"], w, label="Embedding", color=RED, alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pairs_plot["pair"], rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("KS Statistic")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULT / "ks_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved ks_comparison.png")

    print(f"\n=== PAPER NUMBERS ===")
    print(f"Raw feature KS (mean across features + pairs): {raw_ks_mean:.4f}")
    print(f"C1 embedding KS:                               {emb_ks_mean:.4f}")
    print(f"Reduction:                                     {(1-emb_ks_mean/raw_ks_mean)*100:.1f}%")


if __name__ == "__main__":
    main()
