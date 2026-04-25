import torch
import torch.nn as nn
import torch.nn.functional as F


class AccommodationEncoder(nn.Module):
    def __init__(
        self,
        n_providers: int,
        n_categories: int,
        n_countries: int,
        cont_dim: int = 11,
        emb_provider: int = 8,
        emb_category: int = 8,
        emb_country: int = 16,
        hidden_dims: tuple = (512, 256),
        output_dim: int = 128,
    ):
        super().__init__()
        # index 0 = masked/unknown token for all categorical embeddings
        self.provider_emb = nn.Embedding(n_providers + 1, emb_provider, padding_idx=0)
        self.category_emb = nn.Embedding(n_categories + 1, emb_category, padding_idx=0)
        self.country_emb = nn.Embedding(n_countries + 1, emb_country, padding_idx=0)

        input_dim = cont_dim + emb_provider + emb_category + emb_country

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.backbone = nn.Sequential(*layers)

    def forward(
        self,
        cont: torch.Tensor,
        provider_idx: torch.Tensor,
        category_idx: torch.Tensor,
        country_idx: torch.Tensor,
    ) -> torch.Tensor:
        p = self.provider_emb(provider_idx)
        c = self.category_emb(category_idx)
        r = self.country_emb(country_idx)
        x = torch.cat([cont, p, c, r], dim=1)
        return self.backbone(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


class ContrastiveModel(nn.Module):
    def __init__(self, encoder: AccommodationEncoder, head: ProjectionHead):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = self.encoder(batch["cont1"], batch["provider1"], batch["category1"], batch["country1"])
        h2 = self.encoder(batch["cont2"], batch["provider2"], batch["category2"], batch["country2"])
        z1 = self.head(h1)
        z2 = self.head(h2)
        return z1, z2

    def encode(self, cont, provider_idx, category_idx, country_idx) -> torch.Tensor:
        return self.encoder(cont, provider_idx, category_idx, country_idx)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    NT-Xent loss (Chen et al. 2020).
    z1, z2: (N, dim) — one view per positive pair.
    For each i, positive is (i, i+N); all other 2(N-1) pairs are negatives.
    """
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)              # (2N, dim)
    z = F.normalize(z, dim=1)

    sim = torch.mm(z, z.T) / temperature         # (2N, 2N)
    sim.fill_diagonal_(float("-inf"))            # mask self-similarity

    # Positive pair for row i (in z1) is row i+N (in z2), and vice versa
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(N, device=z.device),
    ])

    return F.cross_entropy(sim, labels)


def build_model(vocab) -> ContrastiveModel:
    encoder = AccommodationEncoder(
        n_providers=vocab.n_providers,
        n_categories=vocab.n_categories,
        n_countries=vocab.n_countries,
    )
    head = ProjectionHead()
    return ContrastiveModel(encoder, head)
