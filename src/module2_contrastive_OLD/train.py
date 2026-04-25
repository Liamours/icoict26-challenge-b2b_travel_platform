# Module 2 — Contrastive Training
#
# Framework: PyTorch + custom training loop (no Lightning needed here)
#
# Training:
#   - Optimizer: Adam, lr=1e-3, weight_decay=1e-6
#   - Schedule: CosineAnnealingLR
#   - Epochs: 50–100 (early stop on val NT-Xent loss)
#   - Batch size: 512 (large batch critical for contrastive — Chen et al. 2020)
#
# Ablation variants (for paper Table):
#   1. Encoder with provider ID as feature (biased baseline)
#   2. Encoder without provider ID (naive debiasing baseline)
#   3. Full contrastive + provider masking (proposed)
#   4. Full contrastive + GRL adversarial (optional if 3 underperforms)
#
# Evaluation metrics:
#   - NT-Xent loss on val set
#   - t-SNE / UMAP visualization (colored by provider vs colored by star_rating)
#   - Spearman rank correlation: does embedding order match human-judged quality?
#   - KS divergence between provider embedding distributions (should decrease vs raw features)
#
# Outputs saved to result/module2/:
#   - encoder checkpoint
#   - property_embeddings.npy (N_properties × 128)
#   - property_ids.npy (aligned index)
#   - tsne_visualization.png
#
# Notes:
#   - RTX 4050 6GB — batch 512 should fit. If OOM, reduce to 256.
#   - Property embeddings feed into M3 (cross-attention) and unifier layer
