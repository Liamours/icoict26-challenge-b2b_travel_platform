# Module 1 — TFT Dataset
#
# TASK: Transform raw transaction data into TFT-ready panel format
#
# Input:
#   - dataset/raw/Accommodation - Transactions by HotelCode.xlsx
#   - External: Bank Indonesia FX rates (IDR/AUD/USD), holiday calendar, UNWTO SE Asia stats
#
# Key steps:
#   1. Aggregate transactions to destination clusters × monthly timestep
#      - Cluster candidates: by destination_display_name (join with accommodations.csv)
#        OR by provider prefix (quick fallback if join coverage is low)
#      - Target variable: sum(Number Transactions) per cluster per month
#   2. Build static covariates (time-invariant per cluster):
#      - cluster size (# properties), dominant provider, country/region
#   3. Build known future inputs (calendar features):
#      - month_of_year, is_holiday, school_holiday_AU (Australian traveler seasonality)
#   4. Build observed past inputs:
#      - FX rates (IDR/AUD, IDR/USD), lagged transactions
#   5. Output: PyTorch Geometric Temporal or PyTorch Forecasting TimeSeriesDataSet
#
# Notes:
#   - Only 13 monthly timesteps total (2025-03 → 2026-03)
#   - TFT needs min ~20 timesteps per series — use global model across all clusters
#   - Transfer learning from accommodation metadata embeddings (M2) as static covariate input
#   - Reference: Lim et al. 2021 (IJF) — TimeSeriesDataSet from pytorch-forecasting
