"""Regenerate t-SNE plots from saved embeddings (no retraining needed)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.manifold import TSNE

RESULT = Path(__file__).parents[1] / "result" / "c1_contrastive"
PARQUET = Path(__file__).parents[1] / "dataset" / "preprocessed" / "accommodations.parquet"

embeddings = np.load(RESULT / "property_embeddings.npy", mmap_mode="r")
print(f"Embeddings shape: {embeddings.shape}")

df = pd.read_parquet(PARQUET, columns=["provider", "star_rating"])
provider_map = {p: i + 1 for i, p in enumerate(sorted(df["provider"].dropna().unique()))}
provider_labels = df["provider"].map(provider_map).fillna(0).astype(int).values
tier_labels = np.floor(df["star_rating"].fillna(-1)).clip(-1, 4).astype(int).values

n = 5000
idx = np.random.default_rng(42).choice(len(embeddings), n, replace=False)
e2d = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1).fit_transform(embeddings[idx])

provider_names = {i + 1: p for i, p in enumerate(sorted(df["provider"].dropna().unique()))}
PROV_PALETTE = ['#1B4F8A', '#C0392B', '#9E9E9E', '#4A4A4A']
TIER_CMAP = mcolors.LinearSegmentedColormap.from_list('tier', ['#D6EAF8', '#1B4F8A'], N=7)

# Provider t-SNE
fig, ax = plt.subplots(figsize=(7, 5))
for pid in sorted(np.unique(provider_labels[idx])):
    if pid == 0:
        continue
    mask = provider_labels[idx] == pid
    color = PROV_PALETTE[(pid - 1) % len(PROV_PALETTE)]
    name = provider_names.get(int(pid), f"Provider {pid}")
    ax.scatter(e2d[mask, 0], e2d[mask, 1], c=color, s=4, alpha=0.5,
               label=name, rasterized=True)
ax.legend(fontsize=10, markerscale=3, framealpha=0.85, loc="best", handletextpad=0.4)
ax.axis("off")
plt.tight_layout(pad=0.3)
plt.savefig(RESULT / "tsne_provider.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved tsne_provider.png")

# Quality t-SNE
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(e2d[:, 0], e2d[:, 1], c=tier_labels[idx], s=4, alpha=0.5,
                cmap=TIER_CMAP, vmin=-1, vmax=4, rasterized=True)
cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label("Quality tier", fontsize=10)
cbar.set_ticks([-1, 0, 1, 2, 3, 4])
cbar.ax.tick_params(labelsize=9)
ax.axis("off")
plt.tight_layout(pad=0.3)
plt.savefig(RESULT / "tsne_quality.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved tsne_quality.png")

print("Done.")
