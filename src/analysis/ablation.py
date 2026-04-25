import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parents[1] / "module1_contrastive"))
from dataset import AccommodationVocab, build_continuous_features, PARQUET
from model import build_model

RESULT = Path(__file__).parents[2] / "result" / "analysis"
RESULT.mkdir(parents=True, exist_ok=True)
C1_RESULT = Path(__file__).parents[2] / "result" / "c1_contrastive"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_PROBE = 10_000
N_NN = 1_000
K = 10
SEED = 42


def probe_acc(X, y, test_size=0.2):
    valid = y >= 0
    Xv, yv = X[valid], y[valid]
    if len(np.unique(yv)) < 2:
        return float("nan")
    Xtr, Xte, ytr, yte = train_test_split(Xv, yv, test_size=test_size, random_state=SEED)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    return accuracy_score(yte, clf.predict(Xte))


def cp_nn_rate(features_normed, provider_labels, k):
    np.fill_diagonal(sim := cosine_similarity(features_normed), -np.inf)
    top_k = np.argpartition(sim, -k, axis=1)[:, -k:]
    rates = []
    for i in range(len(provider_labels)):
        nbr_provs = provider_labels[top_k[i]]
        rates.append((nbr_provs != provider_labels[i]).mean())
    return float(np.mean(rates))


@torch.no_grad()
def embed_batch(model, cont, cat_idx, cntry_idx, prov_idx, device, chunk=5_000):
    N = len(cont)
    out = np.empty((N, 128), dtype=np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        h = model.encode(
            torch.from_numpy(cont[s:e]).to(device),
            torch.from_numpy(prov_idx[s:e].astype(np.int64)).to(device),
            torch.from_numpy(cat_idx[s:e].astype(np.int64)).to(device),
            torch.from_numpy(cntry_idx[s:e].astype(np.int64)).to(device),
        )
        out[s:e] = h.cpu().numpy()
    return out


def main():
    print(f"Device: {DEVICE}")
    rng = np.random.default_rng(SEED)

    print("Loading data...")
    df = pd.read_parquet(PARQUET)
    df = df.reset_index(drop=True)
    vocab = AccommodationVocab(df)

    # Stratified sample for NN test (250/provider × 4 = 1000)
    nn_indices = []
    for prov in sorted(df["provider"].dropna().unique()):
        pool = df.index[df["provider"] == prov].to_numpy()
        nn_indices.append(rng.choice(pool, min(N_NN, len(pool)), replace=False))
    nn_indices = np.concatenate(nn_indices)

    # Random sample for probe
    probe_indices = rng.choice(len(df), min(N_PROBE, len(df)), replace=False)

    sub_nn = df.iloc[nn_indices].reset_index(drop=True)
    sub_probe = df.iloc[probe_indices].reset_index(drop=True)

    prov_nn = sub_nn["provider"].values
    star = sub_probe["star_rating"].values
    tiers = np.floor(np.where(np.isnan(star), -1, star)).clip(-1, 4).astype(int)

    cont_nn = build_continuous_features(sub_nn)
    prov_idx_nn, cat_nn, cntry_nn = vocab.encode(sub_nn)
    cont_probe = build_continuous_features(sub_probe)
    prov_idx_probe, cat_probe, cntry_probe = vocab.encode(sub_probe)

    zeros_nn = np.zeros_like(prov_idx_nn)
    zeros_probe = np.zeros_like(prov_idx_probe)

    # Random baseline for CP-NN
    counts = pd.Series(prov_nn).value_counts()
    fracs = (counts / counts.sum()).values
    random_baseline = float(1 - np.sum(fracs ** 2))

    class _FV:
        pass
    fv = _FV()
    fv.n_providers = vocab.n_providers
    fv.n_categories = vocab.n_categories
    fv.n_countries = vocab.n_countries

    model = build_model(fv).to(DEVICE)
    model.encoder.load_state_dict(
        torch.load(C1_RESULT / "encoder_best.pth", map_location=DEVICE)
    )
    model.eval()

    results = []

    # ── Variant A: raw features (no encoder) ─────────────────────────────────
    print("\nVariant A: raw features (no encoder)...")
    raw_nn_normed = normalize(
        np.concatenate([cont_nn,
                        (prov_idx_nn / max(vocab.n_providers, 1)).reshape(-1, 1).astype(np.float32),
                        (cat_nn / max(vocab.n_categories, 1)).reshape(-1, 1).astype(np.float32),
                        (cntry_nn / max(vocab.n_countries, 1)).reshape(-1, 1).astype(np.float32)],
                       axis=1),
        norm="l2"
    )
    raw_probe_normed = normalize(cont_probe, norm="l2")
    cp_raw = cp_nn_rate(raw_nn_normed, prov_nn, K)
    acc_raw = probe_acc(raw_probe_normed, tiers)
    results.append({"variant": "A: Raw features (no encoder)",
                    "quality_acc": round(acc_raw, 4),
                    "cp_nn_rate": round(cp_raw, 4),
                    "provider_masked": "N/A"})
    print(f"  quality_acc={acc_raw:.4f}  cp_rate={cp_raw:.4f}")

    # ── Variant B: encoder, provider NOT masked ───────────────────────────────
    print("Variant B: encoder, provider NOT masked...")
    emb_b_nn = embed_batch(model, cont_nn, cat_nn, cntry_nn, prov_idx_nn, DEVICE)
    emb_b_probe = embed_batch(model, cont_probe, cat_probe, cntry_probe, prov_idx_probe, DEVICE)
    cp_b = cp_nn_rate(normalize(emb_b_nn, norm="l2"), prov_nn, K)
    acc_b = probe_acc(emb_b_probe, tiers)
    results.append({"variant": "B: Encoder, no masking",
                    "quality_acc": round(acc_b, 4),
                    "cp_nn_rate": round(cp_b, 4),
                    "provider_masked": "No"})
    print(f"  quality_acc={acc_b:.4f}  cp_rate={cp_b:.4f}")

    # ── Variant C: encoder, provider MASKED (proposed) ───────────────────────
    print("Variant C: encoder, provider masked (proposed)...")
    emb_c_nn = embed_batch(model, cont_nn, cat_nn, cntry_nn, zeros_nn, DEVICE)
    emb_c_probe = embed_batch(model, cont_probe, cat_probe, cntry_probe, zeros_probe, DEVICE)
    cp_c = cp_nn_rate(normalize(emb_c_nn, norm="l2"), prov_nn, K)
    acc_c = probe_acc(emb_c_probe, tiers)
    results.append({"variant": "C: Encoder + masking (proposed)",
                    "quality_acc": round(acc_c, 4),
                    "cp_nn_rate": round(cp_c, 4),
                    "provider_masked": "Yes"})
    print(f"  quality_acc={acc_c:.4f}  cp_rate={cp_c:.4f}")

    df_res = pd.DataFrame(results)
    df_res["random_baseline"] = round(random_baseline, 4)
    df_res.to_csv(RESULT / "ablation.csv", index=False)
    print(f"\nSaved ablation.csv")
    print(f"\n{df_res[['variant','quality_acc','cp_nn_rate']].to_string(index=False)}")
    print(f"Random baseline (cp): {random_baseline:.4f}")

    # Figure
    COLORS = ['#D0D0D0', '#1B4F8A', '#C0392B']   # A: gray, B: blue, C: red
    EDGE   = '#4A4A4A'
    RED    = '#C0392B'

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    labels = ["A", "B", "C"]
    x = np.arange(len(results))

    axes[0].bar(x, [r["quality_acc"] for r in results], color=COLORS,
                edgecolor=EDGE, linewidth=0.7, width=0.55)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylabel("Quality Tier Accuracy")
    axes[0].set_ylim(0.80, 1.02)
    axes[0].tick_params(labelsize=9)

    axes[1].bar(x, [r["cp_nn_rate"] for r in results], color=COLORS,
                edgecolor=EDGE, linewidth=0.7, width=0.55)
    axes[1].axhline(random_baseline, color=RED, linestyle="--", linewidth=1.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylabel("Cross-Provider NN Rate")
    axes[1].set_ylim(0, 1.0)
    axes[1].tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(RESULT / "ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved ablation.png")

    print(f"\n=== PAPER TABLE ===")
    for r in results:
        print(f"  {r['variant']:40s} | quality_acc={r['quality_acc']} | cp_rate={r['cp_nn_rate']}")
    print(f"  {'Random baseline':40s} | {'':12s} | cp_rate={random_baseline:.4f}")


if __name__ == "__main__":
    main()
