"""
Short-run domain-adversarial baseline for provider invariance.

This baseline keeps the same positive-pair construction and NT-Xent objective as
the main encoder, then adds a provider classifier through a gradient reversal
layer (GRL). It is intended as a minimal comparison, not a fully tuned DANN.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from torch.autograd import Function
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1] / "module1_contrastive"))
from dataset import AccommodationPairDataset, AccommodationVocab, PARQUET, build_continuous_features
from model import AccommodationEncoder, ProjectionHead, nt_xent_loss


ROOT = Path(__file__).parents[2]
RESULT = ROOT / "result" / "analysis"
RESULT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

CFG = {
    "epochs": 20,
    "n_pairs_per_epoch": 50_000,
    "batch_size": 512,
    "lr": 1e-3,
    "weight_decay": 1e-6,
    "temperature": 0.07,
    "probe_n": 8_000,
    "nn_per_provider": 500,
    "k": 10,
}

DEFAULT_LAMBDAS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.50]


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_):
    return GradientReversal.apply(x, lambda_)


class AdversarialContrastiveModel(nn.Module):
    def __init__(self, vocab, output_dim=128, proj_hidden=64, proj_dim=32):
        super().__init__()
        self.encoder = AccommodationEncoder(
            n_providers=vocab.n_providers,
            n_categories=vocab.n_categories,
            n_countries=vocab.n_countries,
            hidden_dims=(512, 256),
            output_dim=output_dim,
        )
        self.head = ProjectionHead(input_dim=output_dim, hidden_dim=proj_hidden, output_dim=proj_dim)
        self.provider_clf = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, vocab.n_providers + 1),
        )

    def encode(self, cont, provider_idx, category_idx, country_idx):
        return self.encoder(cont, provider_idx, category_idx, country_idx)

    def forward(self, batch, lambda_):
        h1 = self.encode(batch["cont1"], batch["provider1"], batch["category1"], batch["country1"])
        h2 = self.encode(batch["cont2"], batch["provider2"], batch["category2"], batch["country2"])
        z1, z2 = self.head(h1), self.head(h2)
        h = torch.cat([h1, h2], dim=0)
        provider_y = torch.cat([batch["true_provider1"], batch["true_provider2"]], dim=0)
        provider_logits = self.provider_clf(grad_reverse(h, lambda_))
        return z1, z2, provider_logits, provider_y


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


def grl_weight(lambda_, epoch, total_epochs, schedule):
    if schedule == "linear":
        return lambda_ * epoch / max(total_epochs, 1)
    return lambda_


def train_model(model, loader, lambda_, schedule):
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    logs = []
    for epoch in tqdm(range(1, CFG["epochs"] + 1), desc=f"lambda={lambda_} epochs", leave=False):
        model.train()
        total_loss = total_ntx = total_provider = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            batch = move_batch(batch)
            opt.zero_grad()
            active_lambda = grl_weight(lambda_, epoch, CFG["epochs"], schedule)
            z1, z2, provider_logits, provider_y = model(batch, active_lambda)
            contrastive = nt_xent_loss(z1, z2, CFG["temperature"])
            provider_loss = F.cross_entropy(provider_logits, provider_y)
            loss = contrastive + provider_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            total_ntx += contrastive.item()
            total_provider += provider_loss.item()
        row = {
            "epoch": epoch,
            "lambda_grl": lambda_,
            "active_lambda": grl_weight(lambda_, epoch, CFG["epochs"], schedule),
            "schedule": schedule,
            "loss": total_loss / len(loader),
            "nt_xent": total_ntx / len(loader),
            "provider_loss": total_provider / len(loader),
        }
        logs.append(row)
        tqdm.write(
            f"lambda={lambda_:.2f} epoch={epoch:02d} "
            f"loss={row['loss']:.4f} ntx={row['nt_xent']:.4f} provider={row['provider_loss']:.4f}"
        )
    return pd.DataFrame(logs)


@torch.no_grad()
def encode_rows(model, df, vocab, mask_provider=True, chunk=4096):
    cont = build_continuous_features(df)
    prov, cat, country = vocab.encode(df)
    provider_input = np.zeros_like(prov) if mask_provider else prov
    out = np.empty((len(df), model.encoder.backbone[-1].out_features), dtype=np.float32)
    model.eval()
    for start in tqdm(range(0, len(df), chunk), desc=f"encode mask={mask_provider}", leave=False):
        end = min(start + chunk, len(df))
        h = model.encode(
            torch.from_numpy(cont[start:end]).to(DEVICE),
            torch.from_numpy(provider_input[start:end].astype(np.int64)).to(DEVICE),
            torch.from_numpy(cat[start:end].astype(np.int64)).to(DEVICE),
            torch.from_numpy(country[start:end].astype(np.int64)).to(DEVICE),
        )
        out[start:end] = h.cpu().numpy()
    return out, prov


def probe_accuracy(emb, labels):
    valid = labels >= 0
    X, y = emb[valid], labels[valid]
    if len(np.unique(y)) < 2:
        return np.nan
    counts = pd.Series(y).value_counts()
    stratify = y if counts.min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=stratify)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    return accuracy_score(yte, clf.predict(Xte))


def cp_nn_rate(emb, providers):
    sim = cosine_similarity(normalize(emb, norm="l2"))
    np.fill_diagonal(sim, -np.inf)
    top = np.argpartition(sim, -CFG["k"], axis=1)[:, -CFG["k"]:]
    return float(np.mean([(providers[top[i]] != providers[i]).mean() for i in range(len(providers))]))


def sample_eval_frames(df):
    rng = np.random.default_rng(SEED)
    probe_idx = rng.choice(len(df), min(CFG["probe_n"], len(df)), replace=False)
    probe = df.iloc[probe_idx].reset_index(drop=True)
    nn_idx = []
    for provider in sorted(df["provider"].dropna().unique()):
        pool = df.index[df["provider"] == provider].to_numpy()
        nn_idx.append(rng.choice(pool, min(CFG["nn_per_provider"], len(pool)), replace=False))
    nn = df.iloc[np.concatenate(nn_idx)].reset_index(drop=True)
    return probe, nn


def evaluate(model, df, vocab):
    probe, nn = sample_eval_frames(df)
    emb_probe, provider_idx = encode_rows(model, probe, vocab, mask_provider=True)
    star = probe["star_rating"].to_numpy()
    tier = np.floor(np.where(np.isnan(star), -1, star)).clip(-1, 4).astype(int)

    emb_nn, _ = encode_rows(model, nn, vocab, mask_provider=True)
    return {
        "quality_acc_masked": probe_accuracy(emb_probe, tier),
        "provider_acc_masked": probe_accuracy(emb_probe, provider_idx.astype(int)),
        "cp_nn_k10_masked": cp_nn_rate(emb_nn, nn["provider"].to_numpy()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--epochs", type=int, default=CFG["epochs"])
    parser.add_argument("--schedule", choices=["constant", "linear"], default="constant")
    return parser.parse_args()


def main():
    args = parse_args()
    CFG["epochs"] = args.epochs
    set_seed(SEED)
    print(f"Device: {DEVICE}", flush=True)
    dataset = AccommodationPairDataset(n_pairs_per_epoch=CFG["n_pairs_per_epoch"])
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=CFG["batch_size"], shuffle=True, drop_last=True, generator=generator)
    df = pd.read_parquet(PARQUET).reset_index(drop=True)
    vocab = AccommodationVocab(df)

    rows = []
    for lambda_ in args.lambdas:
        print(f"\n=== GRL lambda={lambda_} schedule={args.schedule} ===", flush=True)
        set_seed(SEED)
        model = AdversarialContrastiveModel(dataset.vocab).to(DEVICE)
        logs = train_model(model, loader, lambda_, args.schedule)
        logs.to_csv(RESULT / f"domain_adversarial_training_{args.schedule}_lambda_{lambda_:.2f}.csv", index=False)
        metrics = evaluate(model, df, vocab)
        row = {
            "lambda_grl": lambda_,
            "schedule": args.schedule,
            "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "epochs": CFG["epochs"],
            "final_loss": logs.iloc[-1]["loss"],
            "final_nt_xent": logs.iloc[-1]["nt_xent"],
            "final_provider_loss": logs.iloc[-1]["provider_loss"],
            **metrics,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULT / "domain_adversarial_baseline.csv", index=False)
        print(pd.DataFrame([row]).to_string(index=False), flush=True)

    print(f"\nSaved: {RESULT / 'domain_adversarial_baseline.csv'}", flush=True)


if __name__ == "__main__":
    main()
