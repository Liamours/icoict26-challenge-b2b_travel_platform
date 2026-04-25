# Module 1 — TFT Training
#
# Framework: PyTorch Lightning + pytorch-forecasting
#
# Training strategy:
#   - Global model: train on ALL destination clusters simultaneously (panel paradigm)
#     Justification: Lim 2021 shows global TFT outperforms per-series models on sparse data
#   - 70/15/15 train/val/test split by time (not random — avoid leakage)
#   - Loss: QuantileLoss([0.1, 0.5, 0.9])
#
# Baselines to run (for ablation table in paper):
#   1. Naive seasonal (last year same month)
#   2. ARIMA per cluster
#   3. LSTM (no attention, no variable selection)
#   4. TFT — full model
#   5. TFT — no exogenous signals (ablation)
#   6. TFT — no static covariates (ablation)
#
# Outputs saved to result/module1/:
#   - model checkpoint (.ckpt)
#   - predictions DataFrame (cluster, month, actual, pred_p50, pred_p10, pred_p90)
#   - variable_importance.csv
#   - attention_weights.npy
#
# Notes:
#   - Log with TensorBoard or W&B
#   - Early stopping: patience=10, monitor=val_loss
#   - Transfer learning option: pretrain on accommodation time patterns, fine-tune on transactions
