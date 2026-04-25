# Module 2 — Contrastive Encoder Model
#
# Architecture: Dual encoder with projection head (SimCLR-style, Chen et al. 2020)
#
# Encoder:
#   - Input: tabular property features (continuous + categorical embeddings)
#   - MLP backbone: [input_dim → 512 → 256 → 128] with BatchNorm + ReLU
#   - Output: 128-dim property embedding h
#
# Projection head (for contrastive training only, discarded at inference):
#   - MLP: [128 → 64 → 32]
#   - Output: z (used in NT-Xent loss)
#
# Loss: NT-Xent (normalized temperature-scaled cross-entropy)
#   - temperature τ = 0.07 (SimCLR default)
#   - Batch size: 512 pairs
#
# Provider debiasing mechanism:
#   - Add provider_id as an input during training
#   - At inference: zero out / mask provider_id → embedding becomes provider-agnostic
#   - Alternative: adversarial provider classifier on top of encoder (GRL approach)
#     Use GRL if NT-Xent alone doesn't sufficiently debias (check t-SNE plots)
#
# Output at inference:
#   - 128-dim embedding per property (provider-normalized quality vector)
#   - Used by M3 cross-attention and unifier layer
#
# Notes:
#   - Evaluate embedding quality with: provider-blind ranking correlation (Spearman)
#   - Visualize with t-SNE / UMAP colored by provider and by quality tier
