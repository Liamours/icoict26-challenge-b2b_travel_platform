# Provider-Aware Contrastive Learning for Debiased Accommodation Quality Embeddings in Multi-Provider B2B Travel Platforms

Submitted to ICoICT 2026.

## Author

**Rifqi Luay Adani**  
Informatics, Telkom University

## Structure

```
src/
  module1_contrastive/   # contrastive encoder (C1)
  module2_validation/    # cross-provider NN validation (C2)
  module3_affinity/      # activity-accommodation affinity (C3)
  analysis/              # KS divergence, ablation
  eda/                   # exploratory analysis
  preprocessing/         # data preprocessing
scripts/                 # utility scripts
result/                  # figures and CSVs
```

## Run

```bash
# 1. preprocess
python src/preprocessing/preprocessing.py

# 2. train encoder
python src/module1_contrastive/train.py

# 3. validate
python src/module2_validation/cross_provider_nn.py

# 4. affinity module
python src/module3_affinity/train.py
python src/module3_affinity/inference.py

# 5. analysis
python src/analysis/ks_comparison.py
python src/analysis/ablation.py
```

Dataset not included (proprietary).
