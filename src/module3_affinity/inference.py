"""
C3 — Affinity Inference.

1. Load trained ActivityProjector
2. Project all activities → 128-dim
3. Compute cosine similarity to cluster embeddings
4. Top-K activities per cluster
5. Affinity heatmap: category × destination

OUTPUTS (result/c3_affinity/):
  - activity_embeddings.npy       (N_activities, 128)
  - top_activities_per_cluster.csv
  - affinity_heatmap.png          category × top-N dest heatmap
  - top_clusters_table.csv        paper Table 4
"""

import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    ActivityDataset, load_cluster_embeddings,
    build_activity_features, parse_primary_category,
    ActivityVocab, ACTIVITIES
)
from model import ActivityProjector

RESULT = Path(__file__).parents[2] / "result" / "c3_affinity"
RESULT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K_ACTIVITIES = 20      # top activities per cluster to save
TOP_N_DEST_HEATMAP = 15    # destinations shown in heatmap
CHUNK = 10_000


def embed_activities(model, cont, cat_idx, device):
    N = len(cont)
    out = np.empty((N, 128), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, N, CHUNK):
            e = min(s + CHUNK, N)
            proj = model(
                torch.from_numpy(cont[s:e]).to(device),
                torch.from_numpy(cat_idx[s:e].astype(np.int64)).to(device),
            )
            out[s:e] = proj.cpu().numpy()
    return out


def main():
    print(f"Device: {DEVICE}")

    # ── Load clusters + dataset ───────────────────────────────────────────────
    cluster_emb = load_cluster_embeddings()
    dataset = ActivityDataset(cluster_emb)  # use_osm matches train.py default

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading trained projector...")
    model = ActivityProjector(
        n_categories=dataset.vocab.n_categories,
        cont_dim=dataset.cont_dim,
    ).to(DEVICE)
    model.load_state_dict(torch.load(RESULT / "projector_best.pth", map_location=DEVICE))

    # ── Embed all activities ──────────────────────────────────────────────────
    print(f"Embedding {len(dataset):,} activities...")
    act_emb = embed_activities(model, dataset.cont, dataset.cat_idx, DEVICE)
    act_emb_norm = normalize(act_emb, norm="l2")   # (N_act, 128)
    np.save(RESULT / "activity_embeddings.npy", act_emb)
    print(f"Saved activity_embeddings.npy {act_emb.shape}")

    # ── Cluster embedding matrix ──────────────────────────────────────────────
    dest_names = list(cluster_emb.keys())
    clust_mat = np.stack([cluster_emb[d] for d in dest_names], axis=0)  # (N_clust, 128)
    clust_mat_norm = normalize(clust_mat, norm="l2")

    # ── Affinity: cosine sim (N_act × N_clust) — compute per-cluster to save RAM ─
    print("Computing affinity scores per cluster...")
    rows = []
    for ci, dest in enumerate(dest_names):
        c_vec = clust_mat_norm[ci]  # (128,)
        scores = act_emb_norm @ c_vec  # (N_act,)
        top_idx = np.argpartition(scores, -TOP_K_ACTIVITIES)[-TOP_K_ACTIVITIES:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        for rank, ai in enumerate(top_idx, 1):
            rows.append({
                "destination": dest,
                "rank": rank,
                "activity_name": dataset.activity_names[ai],
                "category": dataset.primary_cats_arr[ai],
                "affinity_score": round(float(scores[ai]), 4),
            })

    df_top = pd.DataFrame(rows)
    df_top.to_csv(RESULT / "top_activities_per_cluster.csv", index=False)
    print(f"Saved top_activities_per_cluster.csv ({len(df_top):,} rows)")

    # ── Affinity heatmap: category × destination ──────────────────────────────
    # For top-N destinations (by activity count), compute mean affinity per category
    print("Building affinity heatmap...")

    # Pick top destinations by number of activities in dataset
    dest_act_counts = pd.Series(dataset.dest_names).value_counts()
    top_dests = dest_act_counts.head(TOP_N_DEST_HEATMAP).index.tolist()

    # Top categories overall
    cat_counts = pd.Series(dataset.primary_cats_arr).value_counts()
    top_cats = cat_counts.head(12).index.tolist()

    heatmap_data = pd.DataFrame(index=top_cats, columns=top_dests, dtype=float)

    for ci, dest in enumerate(dest_names):
        if dest not in top_dests:
            continue
        c_vec = clust_mat_norm[ci]
        scores = act_emb_norm @ c_vec
        mask = dataset.dest_names == dest
        dest_scores = scores[mask]
        dest_cats = dataset.primary_cats_arr[mask]
        for cat in top_cats:
            cat_mask = dest_cats == cat
            if cat_mask.sum() > 0:
                heatmap_data.loc[cat, dest] = float(dest_scores[cat_mask].mean())

    # Shorten destination names for display
    def short_dest(d):
        return d.split(",")[0].strip()

    heatmap_data.columns = [short_dest(d) for d in heatmap_data.columns]
    heatmap_data = heatmap_data.astype(float)

    import matplotlib.colors as mcolors
    affinity_cmap = mcolors.LinearSegmentedColormap.from_list(
        'affinity', ['#F5F5F5', '#C0392B'], N=256
    )

    fig, ax = plt.subplots(figsize=(18, 7))
    sns.heatmap(
        heatmap_data,
        annot=True, fmt=".2f", cmap=affinity_cmap,
        ax=ax, linewidths=0.4,
        cbar_kws={"shrink": 0.75, "label": "Mean cosine affinity"},
        annot_kws={"size": 11},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=35, ha="right", fontsize=11)
    plt.yticks(fontsize=11)
    ax.collections[0].colorbar.ax.tick_params(labelsize=10)
    plt.tight_layout(pad=0.5)
    plt.savefig(RESULT / "affinity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved affinity_heatmap.png")

    # ── Paper table: top-5 activities for top-5 destinations ─────────────────
    top5_dests = dest_act_counts.head(5).index.tolist()
    paper_rows = df_top[df_top["destination"].isin(top5_dests) & (df_top["rank"] <= 5)].copy()
    paper_rows["destination"] = paper_rows["destination"].apply(short_dest)
    paper_rows.to_csv(RESULT / "top_clusters_table.csv", index=False)
    print("Saved top_clusters_table.csv")

    # Print for quick inspection
    print("\n=== TOP-5 ACTIVITIES PER TOP-5 DESTINATIONS ===")
    for dest in top5_dests:
        short = short_dest(dest)
        sub = df_top[df_top["destination"] == dest].head(5)
        print(f"\n{short}:")
        for _, r in sub.iterrows():
            print(f"  {r['rank']}. [{r['category']}] {r['activity_name'][:60]}  "
                  f"(affinity={r['affinity_score']:.4f})")

    print(f"\nAll outputs in {RESULT}")


if __name__ == "__main__":
    main()
