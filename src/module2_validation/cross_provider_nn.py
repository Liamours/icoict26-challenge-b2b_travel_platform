"""
C2 — Cross-Provider Nearest Neighbour Purity Analysis

HYPOTHESIS:
  A provider-debiased encoder should produce embeddings where nearest
  neighbours cross provider boundaries at a higher rate than raw features.

METHOD:
  1. Stratified sample: N_PER_PROVIDER properties per provider
  2. Embed with C1 encoder (provider token zeroed = inference mode)
  3. Build raw feature matrix (continuous + category/country, no provider)
  4. For both spaces: compute top-K cosine NN, measure cross-provider rate
  5. Random baseline = 1 - sum(p_i^2)  [expected cross-provider rate if random]

METRICS:
  - cp_rate: mean % of K neighbours from a different provider
  - Per-provider cp_rate (row of confusion matrix)
  - Provider confusion heatmap: rows=query, cols=neighbour provider

OUTPUTS (result/c2_validation/):
  - cp_nn_results.csv
  - cp_barchart.png        main comparison: raw vs embedding vs random
  - cp_confusion_emb.png   NN provider confusion heatmap (embedding)
  - cp_confusion_raw.png   NN provider confusion heatmap (raw features)
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "module1_contrastive"))
from dataset import AccommodationVocab, build_continuous_features, PARQUET
from model import build_model

RESULT = Path(__file__).parents[2] / "result" / "c2_validation"
RESULT.mkdir(parents=True, exist_ok=True)
C1_RESULT = Path(__file__).parents[2] / "result" / "c1_contrastive"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PER_PROVIDER = 1_000   # stratified sample per provider
K_NEIGHBOURS = 10        # top-K NN (excluding self)
SEED = 42


# ── helpers ──────────────────────────────────────────────────────────────────

def embed_rows(model, cont, cat_idx, cntry_idx, device, chunk=5_000):
    N = len(cont)
    out = np.empty((N, 128), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            h = model.encode(
                torch.from_numpy(cont[s:e]).to(device),
                torch.zeros(e - s, dtype=torch.long, device=device),   # provider masked
                torch.from_numpy(cat_idx[s:e].astype(np.int64)).to(device),
                torch.from_numpy(cntry_idx[s:e].astype(np.int64)).to(device),
            )
            out[s:e] = h.cpu().numpy()
    return out


def cross_provider_rate(sim_matrix: np.ndarray, provider_labels: np.ndarray, k: int):
    """
    For each row, find top-K neighbours (excluding self),
    return mean fraction of those neighbours with a DIFFERENT provider label.
    Also return per-provider rates and full confusion matrix.
    """
    N = len(provider_labels)
    np.fill_diagonal(sim_matrix, -np.inf)   # exclude self

    top_k_idx = np.argpartition(sim_matrix, -k, axis=1)[:, -k:]  # (N, K)

    cp_flags = []   # 1 if different provider, 0 if same
    provider_cp = {p: [] for p in np.unique(provider_labels)}

    providers = np.unique(provider_labels)
    # confusion[i, j] = mean fraction of query-provider-i neighbours that are provider-j
    confusion = np.zeros((len(providers), len(providers)), dtype=np.float32)
    prov_to_idx = {p: i for i, p in enumerate(providers)}

    for row_i in range(N):
        nbrs = top_k_idx[row_i]
        nbr_provs = provider_labels[nbrs]
        qp = provider_labels[row_i]
        diff = (nbr_provs != qp).astype(float)
        cp_flags.append(diff.mean())
        provider_cp[qp].append(diff.mean())
        # confusion row
        for np_ in nbr_provs:
            confusion[prov_to_idx[qp], prov_to_idx[np_]] += 1

    # normalise confusion rows
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion = np.where(row_sums > 0, confusion / row_sums, 0)

    per_prov = {p: np.mean(v) for p, v in provider_cp.items()}
    return float(np.mean(cp_flags)), per_prov, confusion, providers


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    rng = np.random.default_rng(SEED)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading accommodations...")
    df = pd.read_parquet(PARQUET)
    df = df.reset_index(drop=True)
    vocab = AccommodationVocab(df)

    # Stratified sample: N_PER_PROVIDER per provider
    print(f"Sampling {N_PER_PROVIDER}/provider stratified...")
    sampled_indices = []
    for prov in sorted(df["provider"].dropna().unique()):
        pool = df.index[df["provider"] == prov].to_numpy()
        n = min(N_PER_PROVIDER, len(pool))
        chosen = rng.choice(pool, n, replace=False)
        sampled_indices.append(chosen)
        print(f"  {prov}: {n:,} sampled")
    sampled_indices = np.concatenate(sampled_indices)

    sub = df.iloc[sampled_indices].reset_index(drop=True)
    provider_labels = sub["provider"].values
    print(f"Total sample: {len(sub):,}")

    # ── Continuous + categorical features ────────────────────────────────────
    cont = build_continuous_features(sub)
    _, cat_idx, cntry_idx = vocab.encode(sub)

    # Raw feature matrix: continuous (11) + normalised category int + country int
    # Normalise ints to [0,1] so scale is comparable
    cat_norm = (cat_idx / max(vocab.n_categories, 1)).astype(np.float32).reshape(-1, 1)
    cntry_norm = (cntry_idx / max(vocab.n_countries, 1)).astype(np.float32).reshape(-1, 1)
    raw_features = np.concatenate([cont, cat_norm, cntry_norm], axis=1)  # (N, 13)
    raw_features_normed = normalize(raw_features, norm="l2")

    # ── Load C1 encoder ───────────────────────────────────────────────────────
    print("Loading C1 encoder...")

    class _FV:
        pass

    fv = _FV()
    fv.n_providers = vocab.n_providers
    fv.n_categories = vocab.n_categories
    fv.n_countries = vocab.n_countries

    model = build_model(fv).to(DEVICE)
    ckpt_path = C1_RESULT / "encoder_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {ckpt_path}\nRun src/module1_contrastive/train.py first.")
    model.encoder.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # ── Embed ─────────────────────────────────────────────────────────────────
    print("Embedding sample...")
    embeddings = embed_rows(model, cont, cat_idx, cntry_idx, DEVICE)
    embeddings_normed = normalize(embeddings, norm="l2")
    print(f"Embeddings: {embeddings.shape}")

    # ── Random baseline ───────────────────────────────────────────────────────
    counts = pd.Series(provider_labels).value_counts()
    fracs = (counts / counts.sum()).values
    random_baseline = float(1 - np.sum(fracs ** 2))
    print(f"\nRandom baseline cross-provider rate: {random_baseline:.4f}")

    # ── Cosine similarity matrices ────────────────────────────────────────────
    print("Computing cosine similarity (raw features)...")
    sim_raw = cosine_similarity(raw_features_normed)

    print("Computing cosine similarity (embeddings)...")
    sim_emb = cosine_similarity(embeddings_normed)

    # ── Cross-provider rate ───────────────────────────────────────────────────
    print(f"\nComputing cross-provider NN purity (K={K_NEIGHBOURS})...")
    cp_raw, pp_raw, conf_raw, providers = cross_provider_rate(sim_raw, provider_labels, K_NEIGHBOURS)
    cp_emb, pp_emb, conf_emb, _ = cross_provider_rate(sim_emb, provider_labels, K_NEIGHBOURS)

    print(f"\nCross-provider rate @ K={K_NEIGHBOURS}:")
    print(f"  Raw features:  {cp_raw:.4f}")
    print(f"  C1 embeddings: {cp_emb:.4f}")
    print(f"  Random:        {random_baseline:.4f}")
    print(f"  Δ (emb - raw): {cp_emb - cp_raw:+.4f}")

    print("\nPer-provider (embeddings):")
    for p, v in sorted(pp_emb.items()):
        print(f"  {p}: {v:.4f}  (raw: {pp_raw.get(p, float('nan')):.4f})")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows = [{"provider": "ALL", "cp_raw": cp_raw, "cp_emb": cp_emb,
             "random_baseline": random_baseline, "delta": cp_emb - cp_raw,
             "n": len(sub), "K": K_NEIGHBOURS}]
    for p in providers:
        rows.append({"provider": p, "cp_raw": pp_raw.get(p, np.nan),
                     "cp_emb": pp_emb.get(p, np.nan),
                     "random_baseline": random_baseline,
                     "delta": pp_emb.get(p, np.nan) - pp_raw.get(p, np.nan),
                     "n": int((provider_labels == p).sum()), "K": K_NEIGHBOURS})
    pd.DataFrame(rows).to_csv(RESULT / "cp_nn_results.csv", index=False)
    print(f"\nSaved cp_nn_results.csv")

    # ── Figure 1: Bar chart — raw vs embedding vs random ──────────────────────
    prov_list = list(providers)
    x = np.arange(len(prov_list) + 1)  # +1 for "ALL"
    labels = ["ALL"] + prov_list
    vals_raw = [cp_raw] + [pp_raw.get(p, 0) for p in prov_list]
    vals_emb = [cp_emb] + [pp_emb.get(p, 0) for p in prov_list]

    GRAY_L = '#D0D0D0'
    BLUE   = '#1B4F8A'
    RED    = '#C0392B'
    EDGE   = '#4A4A4A'

    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.35
    ax.bar(x - w / 2, vals_raw, w, label="Raw", color=GRAY_L, edgecolor=EDGE, linewidth=0.7)
    ax.bar(x + w / 2, vals_emb, w, label="Embedding", color=BLUE, edgecolor=EDGE, linewidth=0.7)
    ax.axhline(random_baseline, color=RED, linestyle="--", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Cross-Provider NN Rate")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(RESULT / "cp_barchart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved cp_barchart.png")

    # ── Figure 2 & 3: Confusion heatmaps ─────────────────────────────────────
    for conf, name, title in [
        (conf_emb, "cp_confusion_emb.png", "NN Provider Confusion — C1 Embeddings"),
        (conf_raw, "cp_confusion_raw.png", "NN Provider Confusion — Raw Features"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            pd.DataFrame(conf, index=providers, columns=providers),
            annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
            ax=ax, linewidths=0.5,
        )
        ax.set_xlabel("Neighbour Provider", fontsize=11)
        ax.set_ylabel("Query Provider", fontsize=11)
        ax.set_title(title, fontsize=12)
        plt.tight_layout()
        plt.savefig(RESULT / name, dpi=150)
        plt.close()
        print(f"Saved {name}")

    print(f"\nAll outputs in {RESULT}")
    print("\n=== PAPER NUMBERS ===")
    print(f"Cross-provider NN rate (raw):       {cp_raw:.4f} ({cp_raw*100:.1f}%)")
    print(f"Cross-provider NN rate (embedding): {cp_emb:.4f} ({cp_emb*100:.1f}%)")
    print(f"Random baseline:                    {random_baseline:.4f} ({random_baseline*100:.1f}%)")
    print(f"Improvement over raw:               {(cp_emb-cp_raw)*100:+.1f}pp")
    print(f"Improvement over random:            {(cp_emb-random_baseline)*100:+.1f}pp")


if __name__ == "__main__":
    main()
