import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parents[1] / "module1_contrastive"))
from dataset import AccommodationVocab, build_continuous_features, PARQUET
from model import build_model

RESULT = Path(__file__).parents[2] / "result" / "analysis"
RESULT.mkdir(parents=True, exist_ok=True)
C1_RESULT = Path(__file__).parents[2] / "result" / "c1_contrastive"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 10_000
SEED = 42

FEATURE_NAMES = [
    "star_norm", "star_miss", "guest_norm", "guest_miss",
    "star_score", "pop_score", "avail",
    "price_norm", "price_miss", "lat_norm", "lon_norm",
]


@torch.no_grad()
def embed(model, cont, cat_idx, cntry_idx, device, chunk=5_000):
    N = len(cont)
    out = np.empty((N, 128), dtype=np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        h = model.encode(
            torch.from_numpy(cont[s:e]).to(device),
            torch.zeros(e - s, dtype=torch.long, device=device),
            torch.from_numpy(cat_idx[s:e].astype(np.int64)).to(device),
            torch.from_numpy(cntry_idx[s:e].astype(np.int64)).to(device),
        )
        out[s:e] = h.cpu().numpy()
    return out


def probe_quality_acc(embeddings, tiers):
    valid = tiers >= 0
    h_v, t_v = embeddings[valid], tiers[valid]
    if len(np.unique(t_v)) < 2:
        return float("nan")
    X_tr, X_te, y_tr, y_te = train_test_split(h_v, t_v, test_size=0.2, random_state=SEED)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    return accuracy_score(y_te, clf.predict(X_te))


def main():
    print(f"Device: {DEVICE}")
    rng = np.random.default_rng(SEED)

    print("Loading data...")
    df = pd.read_parquet(PARQUET)
    df = df.reset_index(drop=True)
    vocab = AccommodationVocab(df)

    indices = rng.choice(len(df), min(N_SAMPLES, len(df)), replace=False)
    sub = df.iloc[indices].reset_index(drop=True)

    star = sub["star_rating"].values
    tiers = np.floor(np.where(np.isnan(star), -1, star)).clip(-1, 4).astype(int)

    cont = build_continuous_features(sub)
    _, cat_idx, cntry_idx = vocab.encode(sub)

    class _FV:
        pass
    fv = _FV()
    fv.n_providers = vocab.n_providers
    fv.n_categories = vocab.n_categories
    fv.n_countries = vocab.n_countries

    model = build_model(fv).to(DEVICE)
    model.encoder.load_state_dict(
        torch.load(C1_RESULT / "encoder_best.pth", map_location=DEVICE)
    )
    model.eval()

    # Baseline
    print("Computing baseline...")
    base_emb = embed(model, cont, cat_idx, cntry_idx, DEVICE)
    baseline_acc = probe_quality_acc(base_emb, tiers)
    print(f"Baseline quality_acc: {baseline_acc:.4f}")

    # Perturb each feature
    results = []
    for fi, fname in enumerate(FEATURE_NAMES):
        cont_perturbed = cont.copy()
        cont_perturbed[:, fi] = 0.0
        emb_p = embed(model, cont_perturbed, cat_idx, cntry_idx, DEVICE)
        acc_p = probe_quality_acc(emb_p, tiers)
        drop = baseline_acc - acc_p
        results.append({
            "feature": fname,
            "quality_acc_perturbed": round(acc_p, 4),
            "quality_acc_drop": round(drop, 4),
        })
        print(f"  {fname:20s}: acc={acc_p:.4f}  drop={drop:+.4f}")

    df_res = pd.DataFrame(results).sort_values("quality_acc_drop", ascending=False)
    df_res["baseline_acc"] = round(baseline_acc, 4)
    df_res.to_csv(RESULT / "feature_attribution.csv", index=False)
    print(f"\nSaved feature_attribution.csv")

    # Figure
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in df_res["quality_acc_drop"]]
    ax.barh(df_res["feature"], df_res["quality_acc_drop"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Quality Accuracy Drop (baseline − perturbed)", fontsize=11)
    ax.set_title(
        f"Feature Attribution via Input Perturbation\n"
        f"Baseline quality_acc = {baseline_acc:.4f}  |  Red = hurts quality, Blue = negligible",
        fontsize=12,
    )
    for i, (_, row) in enumerate(df_res.iterrows()):
        ax.text(row["quality_acc_drop"] + 0.0005, i, f"{row['quality_acc_drop']:+.4f}",
                va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULT / "feature_attribution.png", dpi=150)
    plt.close()
    print("Saved feature_attribution.png")

    print(f"\n=== PAPER NUMBERS ===")
    print(f"Baseline quality_acc: {baseline_acc:.4f}")
    print("Top 3 most important features:")
    for _, row in df_res.head(3).iterrows():
        print(f"  {row['feature']:20s}: drop = {row['quality_acc_drop']:+.4f}")


if __name__ == "__main__":
    main()
