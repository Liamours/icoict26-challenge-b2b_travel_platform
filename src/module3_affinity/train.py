"""
C3 — Train ActivityProjector.

Supervision: for each activity, target = mean C1 embedding of its destination cluster.
Loss: 1 - cosine_similarity(proj(activity_feat), cluster_embedding)

Weak geographic co-occurrence supervision — no manual labels needed.
Aligns activity feature space with C1 property embedding space.
"""

import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset import ActivityDataset, load_cluster_embeddings
from model import ActivityProjector, cosine_loss

RESULT = Path(__file__).parents[2] / "result" / "c3_affinity"
RESULT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = {
    "epochs": 30,
    "batch_size": 1024,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 5,
}


def main():
    print(f"Device: {DEVICE}")

    cluster_emb = load_cluster_embeddings()
    dataset = ActivityDataset(cluster_emb)
    loader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    model = ActivityProjector(
        n_categories=dataset.vocab.n_categories,
        cont_dim=dataset.cont_dim,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])

    print(f"ActivityProjector params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset: {len(dataset):,} activities | {len(cluster_emb):,} clusters")
    print(f"Categories: {dataset.vocab.n_categories}")

    best_loss = float("inf")
    patience_counter = 0
    log_rows = []

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for cont, cat_idx, target in loader:
            cont, cat_idx, target = cont.to(DEVICE), cat_idx.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            proj = model(cont, cat_idx)
            loss = cosine_loss(proj, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()
        log_rows.append({"epoch": epoch, "loss": round(avg_loss, 6)})
        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), RESULT / "projector_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= CFG["patience"]:
                print(f"Early stop at epoch {epoch}")
                break

    pd.DataFrame(log_rows).to_csv(RESULT / "c3_training_log.csv", index=False)
    print(f"\nBest loss: {best_loss:.4f} | Saved to {RESULT}")


if __name__ == "__main__":
    main()
