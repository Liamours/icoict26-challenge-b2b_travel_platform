"""
C3 — ActivityProjector: maps 25-dim activity features → 128-dim embedding space.
Trained to align with C1 property embeddings via cosine loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CONT_DIM_BASE, EMB_DIM


class ActivityProjector(nn.Module):
    def __init__(
        self,
        n_categories: int,
        cont_dim: int = CONT_DIM_BASE,
        emb_cat_dim: int = EMB_DIM,
        hidden_dim: int = 64,
        output_dim: int = 128,
    ):
        super().__init__()
        self.cat_emb = nn.Embedding(n_categories + 1, emb_cat_dim, padding_idx=0)
        input_dim = cont_dim + emb_cat_dim  # 25

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, cont: torch.Tensor, cat_idx: torch.Tensor) -> torch.Tensor:
        c = self.cat_emb(cat_idx)
        x = torch.cat([cont, c], dim=1)
        return self.net(x)  # (N, 128)


def cosine_loss(proj: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean 1 - cosine_similarity(proj, target). Minimise → align spaces."""
    proj_n = F.normalize(proj, dim=1)
    tgt_n = F.normalize(target, dim=1)
    return (1 - (proj_n * tgt_n).sum(dim=1)).mean()
