import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import ks_2samp

from dataset import AccommodationPairDataset, PARQUET
from model import build_model, nt_xent_loss

RESULT = Path(__file__).parents[2] / "result" / "c1_contrastive"
RESULT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = {
    "n_pairs_per_epoch": 100_000,
    "batch_size": 512,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "temperature": 0.07,
    "patience": 10,
    "eval_every": 5,
    "probe_n_samples": 10_000,
    "embed_chunk_size": 10_000,
}


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, device, temperature) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad()
        z1, z2 = model(batch)
        loss = nt_xent_loss(z1, z2, temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def linear_probe(model, dataset, device, n_samples=10_000) -> dict:
    """
    Train logistic regression on frozen embeddings.
    Metrics:
      provider_acc: target ~25% (chance) — good debiasing = low provider accuracy
      quality_acc:  target ~60%+ — good quality capture = high tier accuracy
    """
    model.eval()
    n = min(n_samples, len(dataset.cont))
    indices = np.random.choice(len(dataset.cont), size=n, replace=False)

    cont, prov, cat, cntry = dataset.get_full_batch(indices, device, mask_provider=False)
    h = model.encode(cont, prov, cat, cntry).cpu().numpy()

    provider_labels = dataset.provider_idx[indices].astype(int)
    tiers = np.floor(
        np.where(np.isnan(
            pd.read_parquet(PARQUET, columns=["star_rating"])["star_rating"].values[indices]
        ), -1,
            pd.read_parquet(PARQUET, columns=["star_rating"])["star_rating"].values[indices]
        )
    ).clip(-1, 4).astype(int)
    valid = tiers >= 0
    h_v, t_v = h[valid], tiers[valid]
    p_v = provider_labels[valid]

    results = {}
    for label_name, labels in [("provider_acc", p_v), ("quality_acc", t_v)]:
        if len(np.unique(labels)) < 2:
            results[label_name] = float("nan")
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(h_v, labels, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr, y_tr)
        results[label_name] = accuracy_score(y_te, clf.predict(X_te))

    return results


@torch.no_grad()
def provider_ks_divergence(model, dataset, device, n_per_provider=2000) -> float:
    """
    Mean KS statistic between embedding distributions of all provider pairs.
    Should decrease during training (debiasing effect).
    """
    model.eval()
    provider_embeddings = {}
    for prov_id in range(1, dataset.vocab.n_providers + 1):
        mask = dataset.provider_idx == prov_id
        indices = np.where(mask)[0]
        if len(indices) < 10:
            continue
        indices = indices[np.random.choice(len(indices), min(n_per_provider, len(indices)), replace=False)]
        cont, prov, cat, cntry = dataset.get_full_batch(indices, device, mask_provider=True)
        h = model.encode(cont, prov, cat, cntry).cpu().numpy()
        provider_embeddings[prov_id] = h

    if len(provider_embeddings) < 2:
        return float("nan")

    keys = list(provider_embeddings.keys())
    ks_vals = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            e1 = provider_embeddings[keys[i]].flatten()
            e2 = provider_embeddings[keys[j]].flatten()
            ks_vals.append(ks_2samp(e1, e2).statistic)
    return float(np.mean(ks_vals))


@torch.no_grad()
def generate_all_embeddings(model, dataset, device) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed all 3.7M properties (provider masked = inference mode).
    Returns (embeddings (N, 128), property_ids (N,)).
    """
    model.eval()
    df_ids = pd.read_parquet(PARQUET, columns=["id"])["id"].values
    N = len(dataset.cont)
    chunk = CFG["embed_chunk_size"]
    embeddings = np.empty((N, 128), dtype=np.float32)

    for start in tqdm(range(0, N, chunk), desc="Generating embeddings"):
        end = min(start + chunk, N)
        indices = np.arange(start, end)
        cont, prov, cat, cntry = dataset.get_full_batch(indices, device, mask_provider=True)
        h = model.encode(cont, prov, cat, cntry)
        embeddings[start:end] = h.cpu().numpy()

    return embeddings, df_ids


def visualize_tsne(embeddings: np.ndarray, provider_labels: np.ndarray, tier_labels: np.ndarray):
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        n = min(5000, len(embeddings))
        idx = np.random.choice(len(embeddings), n, replace=False)
        e2d = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1).fit_transform(embeddings[idx])

        provider_names = {1: "Agoda", 2: "Expedia", 3: "HotelBeds", 4: "RateHawk"}
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        PROV_PALETTE = ['#1B4F8A', '#C0392B', '#9E9E9E', '#4A4A4A']
        TIER_CMAP = mcolors.LinearSegmentedColormap.from_list(
            'tier', ['#D6EAF8', '#1B4F8A'], N=7
        )

        # --- Provider t-SNE ---
        fig, ax = plt.subplots(figsize=(7, 5))
        prov_ids = sorted(np.unique(provider_labels[idx]))
        for pid in prov_ids:
            mask = provider_labels[idx] == pid
            color = PROV_PALETTE[int(pid) % len(PROV_PALETTE)]
            name = provider_names.get(int(pid), f"Provider {pid}")
            ax.scatter(e2d[mask, 0], e2d[mask, 1], c=color, s=4, alpha=0.5,
                       label=name, rasterized=True)
        ax.legend(fontsize=10, markerscale=3, framealpha=0.85,
                  loc="best", handletextpad=0.4)
        ax.axis("off")
        plt.tight_layout(pad=0.3)
        plt.savefig(RESULT / "tsne_provider.png", dpi=150, bbox_inches="tight")
        plt.close()

        # --- Quality t-SNE ---
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(e2d[:, 0], e2d[:, 1], c=tier_labels[idx], s=4, alpha=0.5,
                        cmap=TIER_CMAP, vmin=-1, vmax=4, rasterized=True)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("Quality tier", fontsize=10)
        cbar.set_ticks([-1, 0, 1, 2, 3, 4])
        cbar.ax.tick_params(labelsize=9)
        ax.axis("off")
        plt.tight_layout(pad=0.3)
        plt.savefig(RESULT / "tsne_quality.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"t-SNE plots saved to {RESULT}")
    except Exception as e:
        print(f"t-SNE failed: {e}")


def main():
    print(f"Device: {DEVICE}")

    dataset = AccommodationPairDataset(
        parquet_path=PARQUET,
        n_pairs_per_epoch=CFG["n_pairs_per_epoch"],
        min_cluster_size=5,
    )
    loader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = build_model(dataset.vocab).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])

    best_loss = float("inf")
    patience_counter = 0
    log_rows = []

    for epoch in range(1, CFG["epochs"] + 1):
        loss = train_one_epoch(model, loader, optimizer, DEVICE, CFG["temperature"])
        scheduler.step()

        row = {"epoch": epoch, "loss": round(loss, 6)}

        if epoch % CFG["eval_every"] == 0:
            probe = linear_probe(model, dataset, DEVICE, CFG["probe_n_samples"])
            ks = provider_ks_divergence(model, dataset, DEVICE)
            row.update(probe)
            row["provider_ks"] = round(ks, 4) if not np.isnan(ks) else None
            print(
                f"Epoch {epoch:3d} | loss={loss:.4f} | "
                f"prov_acc={probe.get('provider_acc', 0):.3f} | "
                f"qual_acc={probe.get('quality_acc', 0):.3f} | "
                f"ks={ks:.4f}"
            )
        else:
            print(f"Epoch {epoch:3d} | loss={loss:.4f}")

        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(RESULT / "training_log.csv", index=False)

        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            torch.save(model.encoder.state_dict(), RESULT / "encoder_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= CFG["patience"]:
                print(f"Early stop at epoch {epoch}")
                break

    print("Generating full property embeddings...")
    model.encoder.load_state_dict(torch.load(RESULT / "encoder_best.pth", map_location=DEVICE))
    embeddings, property_ids = generate_all_embeddings(model, dataset, DEVICE)
    np.save(RESULT / "property_embeddings.npy", embeddings)
    np.save(RESULT / "property_ids.npy", property_ids)
    print(f"Saved embeddings: {embeddings.shape}")

    df = pd.read_parquet(PARQUET, columns=["provider", "star_rating"])
    provider_labels = dataset.provider_idx
    tier_labels = np.floor(df["star_rating"].fillna(-1)).clip(-1, 4).astype(int).values
    visualize_tsne(embeddings, provider_labels, tier_labels)

    print(f"Done. All outputs in {RESULT}")


if __name__ == "__main__":
    main()
