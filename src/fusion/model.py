# Fusion Layer — Cross-Module Attention Unifier
#
# TASK: Combine M1 demand score + M2 property embeddings + M3 affinity scores
#       → ranked property-activity bundles for partner storefronts
#
# Input per candidate bundle (accommodation + activity set):
#   - demand_score: scalar from M1 (cluster-level, broadcast to properties in cluster)
#   - property_embedding: 128-dim from M2
#   - affinity_scores: top-K activity affinities from M3
#   - price_tier: discretized reference_price bucket (optional)
#
# Architecture:
#   - Concat [demand_score, property_embedding, max_affinity, mean_affinity, top_activity_category_emb]
#   - MLP: [input_dim → 128 → 64 → 1] → bundle_score (scalar)
#   - Rank bundles by bundle_score descending
#
# Alternative: cross-module attention
#   - Q = property_embedding
#   - K, V = stack([demand_embedding, activity_embeddings])
#   - Attend over M1 and M3 signals jointly
#   - More powerful, but harder to explain — only use if MLP fusion underperforms
#
# Output:
#   - ranked list of (property_id, [activity_id, ...], bundle_score) per destination cluster
#   - Ready for partner storefront injection via Travlr's Data & Insights Dashboard API
#
# Notes:
#   - This layer is the "system contribution" that ties all modules together for the paper
#   - Keep architecture simple — main novelty is in M1/M2/M3, not the fusion layer
#   - Must be explainable: SHAP on fusion MLP inputs (demand vs quality vs affinity contribution)
