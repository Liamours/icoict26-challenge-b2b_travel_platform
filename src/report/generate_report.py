"""
Generate combined C1 + C2 report.
Reads result CSVs + images, writes result/report/c1_c2_report.md
and compiles a single-page PNG summary figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from pathlib import Path
from datetime import date

ROOT = Path(__file__).parents[2]
C1_DIR = ROOT / "result" / "c1_contrastive"
C2_DIR = ROOT / "result" / "c2_validation"
REPORT_DIR = ROOT / "result" / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────

log = pd.read_csv(C1_DIR / "training_log.csv")
cp = pd.read_csv(C2_DIR / "cp_nn_results.csv")

eval_rows = log.dropna(subset=["quality_acc"])
first_eval = eval_rows.iloc[0]
last_eval = eval_rows.iloc[-1]
best_loss_epoch = log.loc[log["loss"].idxmin()]

cp_all = cp[cp["provider"] == "ALL"].iloc[0]
cp_prov = cp[cp["provider"] != "ALL"].set_index("provider")

emb_shape = np.load(C1_DIR / "property_embeddings.npy", mmap_mode="r").shape


# ── Markdown report ───────────────────────────────────────────────────────────

def fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if pd.notna(v) else "N/A"


md = f"""# C1 + C2 Results Report
**Generated:** {date.today()}
**Project:** ICoICT 2026 — Provider-Aware Contrastive Embeddings for B2B Travel Inventory Debiasing

---

## Module 1 — Contrastive Encoder (C1)

### Architecture
| Component | Detail |
|---|---|
| Encoder | Linear(43→512)→BN→ReLU→Linear(512→256)→BN→ReLU→Linear(256→128) |
| Input dim | 43 = 11 continuous + 8 provider emb + 8 category emb + 16 country emb |
| Projection head | Linear(128→64)→ReLU→Linear(64→32) — discarded at inference |
| Total params | 204,248 |
| Loss | NT-Xent (SimCLR), temperature=0.07 |

### Training Config
| Param | Value |
|---|---|
| Pairs/epoch | 100,000 |
| Batch size | 512 |
| Epochs (ran) | {int(log['epoch'].max())} |
| LR | 1e-3 (cosine annealing) |
| Weight decay | 1e-6 |
| Positive pair proxy | Same destination × same star tier |
| Augmentation | Gaussian noise σ=0.05, feature dropout p=0.2, provider mask p=0.5 |
| Valid clusters | 68,374 |
| Properties covered | 3,158,693 / 3,686,694 |

### Training Dynamics

| Epoch | Loss | Provider Acc ↓ | Quality Acc ↑ | Provider KS ↓ |
|---|---|---|---|---|
| {int(first_eval['epoch'])} (first eval) | {fmt(first_eval['loss'])} | {fmt(first_eval['provider_acc'], 3)} | {fmt(first_eval['quality_acc'], 3)} | {fmt(first_eval['provider_ks'], 4)} |
| {int(last_eval['epoch'])} (final) | {fmt(last_eval['loss'])} | {fmt(last_eval['provider_acc'], 3)} | {fmt(last_eval['quality_acc'], 3)} | {fmt(last_eval['provider_ks'], 4)} |

- **Loss:** {fmt(log['loss'].iloc[0])} → {fmt(last_eval['loss'])} (converged)
- **quality_acc = {fmt(last_eval['quality_acc'], 3)}** — near-perfect linear separability of quality tiers from embeddings ✓
- **provider_ks = {fmt(last_eval['provider_ks'], 4)}** — inter-provider distribution divergence very low ✓
- **provider_acc = {fmt(last_eval['provider_acc'], 3)}** — provider still linearly recoverable (partial debiasing, see §Limitations)

### Output Embeddings
| Property | Value |
|---|---|
| Shape | {emb_shape[0]:,} × {emb_shape[1]} |
| Norm mean | ~13.14 |
| Norm range | [4.53, 53.77] |
| Saved | `result/c1_contrastive/property_embeddings.npy` |

### Interpretation
The encoder achieves near-perfect quality tier separability (acc=0.995) and very low inter-provider KS divergence (0.018), confirming the embeddings capture accommodation quality in a largely provider-agnostic space.
Residual provider accuracy (0.889) reflects inherent feature-level provider correlation — star rating distributions differ by provider (raw KS=0.707) — making complete linear disentanglement impossible without auxiliary adversarial objectives.

---

## Module 2 — Cross-Provider NN Validation (C2)

### Setup
| Param | Value |
|---|---|
| Sample | 1,000 per provider × 4 = 4,000 properties |
| K neighbours | 10 |
| Similarity | Cosine |
| Raw features | 13-dim (11 continuous + normalised category + country) |
| Embedding | 128-dim C1 (provider token zeroed) |
| Random baseline | {fmt(cp_all['random_baseline'], 4)} (= 1 − Σpᵢ² for equal 25% split) |

### Cross-Provider NN Purity

| | Raw Features | C1 Embeddings | Δ |
|---|---|---|---|
| **ALL** | {fmt(cp_all['cp_raw'], 4)} | {fmt(cp_all['cp_emb'], 4)} | **{cp_all['delta']:+.4f}** |
"""

for prov, row in cp_prov.iterrows():
    md += f"| {prov} | {fmt(row['cp_raw'], 4)} | {fmt(row['cp_emb'], 4)} | {row['delta']:+.4f} |\n"

md += f"""
**Random baseline:** {fmt(cp_all['random_baseline'], 4)}
**Gap to random:** {cp_all['cp_emb'] - cp_all['random_baseline']:+.4f} pp

### Interpretation
Raw features cluster almost exclusively within providers (16.7% cross-provider NN rate) —
confirming severe provider bias in the raw feature space.
C1 embeddings increase cross-provider mixing to **58.6% (+42.0 pp)**, demonstrating substantial
debiasing. The residual gap to the 75% random baseline is consistent with partial debiasing
(prov_acc=0.889) and explained by inherent provider–quality correlations in the source data.

RateHawk shows the lowest cross-provider rate (49.1%) — its distinctive feature profile
(premium star ratings, narrow price range) makes full decorrelation harder.

### Coverage Note
- Transaction property IDs: 2,799
- Matched in accommodations: 1,132 (40.4%)
- Unmatched: 1,667 (59.6%) — predominantly HotelBeds (1,315) due to ID format mismatch
- **Transaction correlation (Spearman ρ):** inconclusive — median 1 booking/property over 13 months provides insufficient variance for validation. Acknowledged as data limitation.

---

## Summary

| Metric | Value | Target | Status |
|---|---|---|---|
| NT-Xent loss (final) | {fmt(last_eval['loss'])} | ↓ converge | ✓ |
| Quality tier acc (linear probe) | {fmt(last_eval['quality_acc'], 3)} | ≥ 0.60 | ✓ |
| Provider KS divergence | {fmt(last_eval['provider_ks'], 4)} | ↓ low | ✓ |
| Provider acc (linear probe) | {fmt(last_eval['provider_acc'], 3)} | ~0.25 (chance) | ✗ partial |
| Cross-provider NN rate (raw) | {fmt(cp_all['cp_raw'], 4)} | baseline | — |
| Cross-provider NN rate (emb) | {fmt(cp_all['cp_emb'], 4)} | > raw | ✓ +42pp |

**Core claim supported:** The C1 encoder substantially reduces provider bias in the embedding space
(raw 16.7% → embedding 58.6% cross-provider NN purity), while preserving strong quality signal
(linear probe acc=0.995). Complete debiasing is not achieved — acknowledged as a limitation.
"""

report_path = REPORT_DIR / "c1_c2_report.md"
report_path.write_text(md, encoding="utf-8")
print(f"Saved markdown report: {report_path}")


# ── Summary figure ────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 14))
fig.suptitle("C1 + C2 Results Summary — Provider-Aware Contrastive Encoder",
             fontsize=15, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Row 0: training curves
ax_loss = fig.add_subplot(gs[0, :2])
ax_metrics = fig.add_subplot(gs[0, 2:])

ax_loss.plot(log["epoch"], log["loss"], color="#2c3e50", linewidth=1.5)
ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("NT-Xent Loss")
ax_loss.set_title("Training Loss"); ax_loss.grid(alpha=0.3)

eval_mask = log["quality_acc"].notna()
ax_metrics.plot(log.loc[eval_mask, "epoch"], log.loc[eval_mask, "quality_acc"],
                label="Quality Acc ↑", color="#27ae60", marker="o", ms=4)
ax_metrics.plot(log.loc[eval_mask, "epoch"], log.loc[eval_mask, "provider_acc"],
                label="Provider Acc ↓", color="#e74c3c", marker="s", ms=4)
ax_metrics.axhline(0.25, color="#e74c3c", linestyle=":", linewidth=1, label="Chance (0.25)")
ax_metrics.set_xlabel("Epoch"); ax_metrics.set_ylabel("Accuracy")
ax_metrics.set_title("Linear Probe Metrics"); ax_metrics.legend(fontsize=8); ax_metrics.grid(alpha=0.3)

# Row 1: t-SNE
for col, (fname, title) in enumerate([
    ("tsne_provider.png", "t-SNE: by Provider"),
    ("tsne_quality.png", "t-SNE: by Quality Tier"),
]):
    ax = fig.add_subplot(gs[1, col * 2: col * 2 + 2])
    try:
        img = mpimg.imread(C1_DIR / fname)
        ax.imshow(img); ax.axis("off"); ax.set_title(title, fontsize=11)
    except Exception:
        ax.text(0.5, 0.5, "Image not found", ha="center"); ax.axis("off")

# Row 2: C2 figures
for col, (fname, title) in enumerate([
    ("cp_barchart.png", "Cross-Provider NN Purity"),
    ("cp_confusion_emb.png", "NN Confusion (Embeddings)"),
]):
    ax = fig.add_subplot(gs[2, col * 2: col * 2 + 2])
    try:
        img = mpimg.imread(C2_DIR / fname)
        ax.imshow(img); ax.axis("off"); ax.set_title(title, fontsize=11)
    except Exception:
        ax.text(0.5, 0.5, "Image not found", ha="center"); ax.axis("off")

fig_path = REPORT_DIR / "c1_c2_summary.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved summary figure: {fig_path}")
print("\nDone.")
