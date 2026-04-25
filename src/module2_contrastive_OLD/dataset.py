# Module 2 — Contrastive Learning Dataset
#
# TASK: Build provider-aware contrastive pairs from 3.7M accommodation records
#
# Core idea (provider debiasing):
#   - Expedia/Agoda/HotelBeds/RateHawk each have different rating scales and coverage bias
#   - EDA KS test should confirm distribution divergence (see eda_accommodations.py)
#   - Goal: learn embeddings where same-property-different-provider → closer,
#     and different-property-same-provider → further (if different quality)
#
# Input features per property (to be confirmed after EDA):
#   - star_rating, guest_rating, guest_rating_count
#   - popularity, popularity_score
#   - reference_price (log-normalized)
#   - country, city (embeddings)
#   - destination_coordinate (lat/lon → sinusoidal encoding)
#
# Positive pair strategy:
#   - OPTION A: Same property ID across providers (if same property exists in multiple providers)
#   - OPTION B: Properties in same destination cluster with similar star_rating bucket
#   - Use OPTION A if multi-provider property overlap exists (check in EDA)
#
# Augmentation (adapted from SimCLR — Chen et al. 2020):
#   - Dropout random features (simulate missing fields across providers)
#   - Add Gaussian noise to continuous features
#   - Randomly mask provider ID (force provider-agnostic representation)
#
# Notes:
#   - 3.7M rows — use streaming DataLoader, do NOT load all into RAM
#   - Sample strategy: mine hard negatives (same destination, different quality tier)
