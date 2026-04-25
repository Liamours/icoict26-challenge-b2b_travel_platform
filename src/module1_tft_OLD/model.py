# Module 1 — TFT Model
#
# Architecture: Temporal Fusion Transformer (Lim et al., 2021)
# Library: pytorch-forecasting (TemporalFusionTransformer class)
#
# Key components:
#   - Variable Selection Networks: learn which inputs matter per timestep
#   - LSTM encoder/decoder: local temporal processing
#   - Interpretable multi-head self-attention: long-range dependencies
#   - Quantile output heads: predict P10/P50/P90 demand (uncertainty quantification)
#
# Input features (to be confirmed after EDA):
#   Static:  cluster_id (embedding), n_properties, dominant_provider (embedding)
#   Known future: month_sin, month_cos, is_holiday, is_school_holiday
#   Observed past: transaction_count (target), fx_idr_aud, fx_idr_usd
#
# Output: multi-horizon demand forecast (1, 2, 3 months ahead) per cluster
#
# Interpretability outputs (for paper):
#   - Variable importance scores (VSN attention weights)
#   - Temporal self-attention patterns
#   - Encoder/decoder variable selection
#
# Notes:
#   - Use pytorch-forecasting's built-in TFT — do not re-implement from scratch
#   - Benchmark against: ARIMA, LSTM (Law et al. 2019 baseline), Prophet
#   - Metric: MAPE (primary), MAE, RMSE (secondary)
#   - Hardware: RTX 4050 6GB — batch_size=32, gradient_clip_val=0.1
