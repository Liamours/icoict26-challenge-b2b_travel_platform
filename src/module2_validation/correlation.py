import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

RESULT = Path(__file__).parents[2] / "result" / "c2_validation"
RESULT.mkdir(parents=True, exist_ok=True)

PROVIDER_COLORS = {
    "Agoda": "#e74c3c",
    "Expedia": "#3498db",
    "HotelBeds": "#2ecc71",
    "RateHawk": "#f39c12",
}


def bootstrap_spearman_ci(x, y, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(x[idx], y[idx])
        stats.append(r)
    lo = np.percentile(stats, (1 - ci) / 2 * 100)
    hi = np.percentile(stats, (1 + ci) / 2 * 100)
    return lo, hi


def main():
    # Load
    df = pd.read_csv(RESULT / "transaction_quality_scores.csv")
    embeddings = np.load(RESULT / "transaction_property_embeddings.npy")
    print(f"Loaded {len(df)} properties, embeddings {embeddings.shape}")

    x = df["quality_score"].values
    # Primary y: months_active (booking consistency — more variance than total_count)
    y = df["months_active"].values.astype(float)
    y_log = np.log1p(df["total_transactions"].values)  # kept for reference

    print(f"months_active: mean={y.mean():.2f}  std={y.std():.2f}  max={y.max()}")
    print(f"total_transactions: mean={df['total_transactions'].mean():.2f}  "
          f"std={df['total_transactions'].std():.2f}  max={df['total_transactions'].max()}")

    # ── Global Spearman ──────────────────────────────────────────────────────
    rho, pval = spearmanr(x, y)
    lo, hi = bootstrap_spearman_ci(x, y)
    rho_tx, pval_tx = spearmanr(x, y_log)
    print(f"\nGlobal Spearman (months_active):      ρ = {rho:.4f}  p = {pval:.4e}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"Global Spearman (log total_tx):        ρ = {rho_tx:.4f}  p = {pval_tx:.4e}")

    # ── Per-provider Spearman ────────────────────────────────────────────────
    provider_rhos = {}
    print("\nPer-provider (months_active):")
    for prov, grp in df.groupby("provider"):
        if len(grp) < 10:
            print(f"  {prov}: skipped (n={len(grp)} < 10)")
            continue
        r, p = spearmanr(grp["quality_score"], grp["months_active"])
        lo_p, hi_p = bootstrap_spearman_ci(
            grp["quality_score"].values,
            grp["months_active"].values.astype(float),
        )
        provider_rhos[prov] = {"rho": r, "p": p, "ci_lo": lo_p, "ci_hi": hi_p, "n": len(grp)}
        print(f"  {prov}: ρ={r:.4f}  p={p:.4e}  CI [{lo_p:.4f}, {hi_p:.4f}]  n={len(grp)}")

    # Save correlation summary
    rows = [{"provider": "ALL", "rho": rho, "p_value": pval, "ci_lo": lo, "ci_hi": hi, "n": len(df)}]
    for prov, v in provider_rhos.items():
        rows.append({"provider": prov, "rho": v["rho"], "p_value": v["p"],
                     "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"], "n": v["n"]})
    pd.DataFrame(rows).to_csv(RESULT / "spearman_results.csv", index=False)

    # ── Figure 1: Scatter quality_score vs log1p(transactions) ──────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for prov in df["provider"].unique():
        mask = df["provider"] == prov
        color = PROVIDER_COLORS.get(prov, "#888888")
        ax.scatter(x[mask], y[mask], c=color, s=10, alpha=0.5, label=prov)
    ax.set_xlabel("Quality Score (cosine similarity to high-quality centroid)", fontsize=12)
    ax.set_ylabel("Months Active (booking consistency)", fontsize=12)
    ax.set_title(
        f"Quality Score vs Booking Consistency\nSpearman ρ = {rho:.3f}, p = {pval:.2e}",
        fontsize=13,
    )
    ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(RESULT / "scatter_quality_vs_transactions.png", dpi=150)
    plt.close()
    print("Saved scatter_quality_vs_transactions.png")

    # ── Figure 2: Per-provider ρ bar chart ───────────────────────────────────
    provs = list(provider_rhos.keys())
    rhos = [provider_rhos[p]["rho"] for p in provs]
    errs_lo = [provider_rhos[p]["rho"] - provider_rhos[p]["ci_lo"] for p in provs]
    errs_hi = [provider_rhos[p]["ci_hi"] - provider_rhos[p]["rho"] for p in provs]
    colors = [PROVIDER_COLORS.get(p, "#888888") for p in provs]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(provs, rhos, color=colors, yerr=[errs_lo, errs_hi],
                  capsize=5, edgecolor="black", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(rho, color="gray", linewidth=1, linestyle=":", label=f"Global ρ={rho:.3f}")
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_title("Per-Provider Spearman Correlation\n(Quality Score vs Transaction Frequency)", fontsize=12)
    ax.legend()
    for bar, val in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(RESULT / "correlation_by_provider.png", dpi=150)
    plt.close()
    print("Saved correlation_by_provider.png")

    # ── Figure 3: PCA of embeddings colored by transaction bucket ────────────
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    e2d = pca.fit_transform(embeddings)

    tx_vals = df["total_transactions"].values
    # Bucket: 0 (no tx), low, mid, high
    buckets = np.zeros(len(tx_vals), dtype=int)
    buckets[tx_vals > 0] = 1
    q33 = np.percentile(tx_vals[tx_vals > 0], 33)
    q66 = np.percentile(tx_vals[tx_vals > 0], 66)
    buckets[tx_vals > q33] = 2
    buckets[tx_vals > q66] = 3

    bucket_labels = {0: "No transactions", 1: "Low", 2: "Mid", 3: "High"}
    bucket_colors = {0: "#cccccc", 1: "#f39c12", 2: "#3498db", 3: "#e74c3c"}

    fig, ax = plt.subplots(figsize=(8, 7))
    for b in [0, 1, 2, 3]:
        mask = buckets == b
        ax.scatter(e2d[mask, 0], e2d[mask, 1],
                   c=bucket_colors[b], s=8, alpha=0.4, label=bucket_labels[b])
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.set_title("PCA of Transaction Property Embeddings\nColored by Transaction Frequency", fontsize=12)
    ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(RESULT / "embedding_pca.png", dpi=150)
    plt.close()
    print("Saved embedding_pca.png")

    print(f"\nAll outputs in {RESULT}")
    print("\nSummary for paper:")
    print(f"  Global ρ = {rho:.4f}  (95% CI [{lo:.4f}, {hi:.4f}])  p = {pval:.2e}")
    for prov, v in provider_rhos.items():
        print(f"  {prov}: ρ = {v['rho']:.4f}  p = {v['p']:.2e}")


if __name__ == "__main__":
    main()
