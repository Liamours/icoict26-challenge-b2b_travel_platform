# Provider-Aware Contrastive Learning for Accommodation Quality Embeddings

**ICoICT 2026 — Travlr Data Challenge (Challenge Track)**

Contrastive encoder that learns provider-invariant accommodation quality embeddings from 3.69M records across four B2B travel providers. Reduces cross-provider KS divergence by 57.2% and raises CP-NN@10 from 16.7% to 59.1%.

---

## Repository Structure

```
src/
  module1_contrastive/   # encoder architecture, dataset, training loop
  module2_validation/    # CP-NN protocol, KS evaluation, embedding export
  module3_affinity/      # activity-accommodation affinity model
  analysis/              # baselines (CORAL, MMD, GRL), ablations, robustness checks
  eda/                   # exploratory analysis
  preprocessing/         # data preprocessing pipeline
  fusion/                # inference and embedding fusion
scripts/                 # utility notebooks
manuscript/              # paper source (LaTeX)
```

---

## Setup

```bash
conda activate pytorch_gpu
uv pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.0+, CUDA-capable GPU.

---

## Usage

**1. Preprocess**
```bash
python src/preprocessing/preprocessing.py
```

**2. Train contrastive encoder**
```bash
python src/module1_contrastive/train.py
```

**3. Export embeddings**
```bash
python src/module2_validation/embed_properties.py
```

**4. Run CP-NN validation**
```bash
python src/module2_validation/cross_provider_nn.py
```

**5. Run baselines and ablations**
```bash
python src/analysis/reviewer_action_suite.py
python src/analysis/ablation.py
```

---

## Dataset

Datasets are provided by Travlr through the ICoICT 2026 Travlr Data Challenge and are not included in this repository. Place raw files under `dataset/raw/` before running preprocessing.

---

## Paper

`manuscript/main.tex` — full paper source. Compile with:
```bash
tectonic manuscript/main.tex
```

---

## Results

Key results from the paper:

| Method | KS | CP-NN@10 |
|--------|----|----------|
| Raw features | 0.304 | 16.7% |
| Star calibration | 0.289 | 10.6% |
| GRL baseline | — | 58.4% |
| **Proposed encoder** | **0.130** | **59.1%** |
| Random mixing (upper bound) | — | 75.0% |
