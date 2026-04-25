# Fusion — Final Bundle Output
#
# TASK: Combine C1 quality scores + C2 transaction validation + C3 affinity
#       → ranked property-activity bundles per destination cluster
#
# PIPELINE:
#   1. Load per-destination cluster: mean C1 embedding, top-K activities from C3
#   2. For each property in cluster: compute quality_score = cosine_sim(embedding, cluster_mean)
#   3. Bundle score = α × quality_score + β × affinity_score
#      where α, β tuned by correlation with transaction frequency (from C2)
#   4. Rank properties within cluster by bundle_score
#   5. For each top property: attach top-3 activity recommendations from C3
#
# OUTPUT format (JSON per cluster):
#   {
#     "cluster": "Ubud, Bali, Indonesia",
#     "n_properties": 500,
#     "top_bundles": [
#       {
#         "property_id": "Expedia-12345",
#         "quality_score": 0.87,
#         "provider": "Expedia",
#         "top_activities": [
#           {"name": "...", "category": "Cultural", "affinity": 0.83},
#           {"name": "...", "category": "Tour And Sight Seeing", "affinity": 0.71}
#         ],
#         "bundle_score": 0.85
#       }
#     ]
#   }
#
# PAPER USE:
#   - This output IS the "Data & Insights Dashboard" Travlr needs
#   - Figure 5: sample bundle output for an Indonesian destination cluster
#   - Connects all three components into one deployable system output
#
# NOTES:
#   - α, β weights from C2 Spearman correlation (data-driven weighting)
#   - Keep simple — fusion is not the contribution, C1 encoder is
