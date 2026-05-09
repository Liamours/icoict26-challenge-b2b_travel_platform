"""
Lightweight architecture sweep for the accommodation contrastive encoder.

This script intentionally keeps the same training principles as the main C1
pipeline: destination-tier positive pairs, NT-Xent loss, stochastic provider
masking from the dataset, frozen-embedding linear probes, and CP-NN validation.
It is not a replacement for full training; it is a short-run comparison to
justify whether the selected architecture is reasonable.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1] / "module1_contrastive"))
from dataset import AccommodationPairDataset, AccommodationVocab, PARQUET, build_continuous_features
from model import AccommodationEncoder, ContrastiveModel, ProjectionHead, nt_xent_loss


ROOT = Path(__file__).parents[2]
RESULT = ROOT / "result" / "analysis" / "architecture_sweep"
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
    "k_values": [5, 10, 20],
    "num_workers": 0,
}

ARCH_GRID = [
    {"name": "small_64_p16", "hidden_dims": (256, 128), "output_dim": 64, "proj_hidden": 32, "proj_dim": 16},
    {"name": "small_128_p32", "hidden_dims": (256, 128), "output_dim": 128, "proj_hidden": 64, "proj_dim": 32},
    {"name": "base_128_p32", "hidden_dims": (512, 256), "output_dim": 128, "proj_hidden": 64, "proj_dim": 32},
    {"name": "base_256_p64", "hidden_dims": (512, 256), "output_dim": 256, "proj_hidden": 128, "proj_dim": 64},
    {"name": "wide_128_p32", "hidden_dims": (1024, 512), "output_dim": 128, "proj_hidden": 64, "proj_dim": 32},
    {"name": "wide_256_p64", "hidden_dims": (1024, 512), "output_dim": 256, "proj_hidden": 128, "proj_dim": 64},
]


def log(msg):
    print(msg, flush=True)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sweep_model(vocab, arch):
    encoder = AccommodationEncoder(
        n_providers=vocab.n_providers,
        n_categories=vocab.n_categories,
        n_countries=vocab.n_countries,
        hidden_dims=arch["hidden_dims"],
        output_dim=arch["output_dim"],
    )
    head = ProjectionHead(
        input_dim=arch["output_dim"],
        hidden_dim=arch["proj_hidden"],
        output_dim=arch["proj_dim"],
    )
    return ContrastiveModel(encoder, head)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def move_batch(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


def train_short(model, loader):
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    losses = []
    for epoch in tqdm(range(1, CFG["epochs"] + 1), desc="epochs", leave=False):
        model.train()
        total = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            batch = move_batch(batch)
            opt.zero_grad()
            z1, z2 = model(batch)
            loss = nt_xent_loss(z1, z2, CFG["temperature"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        losses.append(total / len(loader))
        tqdm.write(f"  epoch {epoch:02d}/{CFG['epochs']} loss={losses[-1]:.4f}")
    return losses


@torch.no_grad()
def encode_rows(model, cont, provider_idx, category_idx, country_idx, chunk=4096, desc="encode"):
    model.eval()
    out = np.empty((len(cont), model.encoder.backbone[-1].out_features), dtype=np.float32)
    for start in tqdm(range(0, len(cont), chunk), desc=desc, leave=False):
        end = min(start + chunk, len(cont))
        h = model.encode(
            torch.from_numpy(cont[start:end]).to(DEVICE),
            torch.from_numpy(provider_idx[start:end].astype(np.int64)).to(DEVICE),
            torch.from_numpy(category_idx[start:end].astype(np.int64)).to(DEVICE),
            torch.from_numpy(country_idx[start:end].astype(np.int64)).to(DEVICE),
        )
        out[start:end] = h.cpu().numpy()
    return out


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


def cp_nn_rate(emb, providers, k):
    if len(emb) <= k:
        return np.nan
    sim = cosine_similarity(normalize(emb, norm="l2"))
    np.fill_diagonal(sim, -np.inf)
    top = np.argpartition(sim, -k, axis=1)[:, -k:]
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


def evaluate_probe(model, probe, vocab, mask_provider):
    cont = build_continuous_features(probe)
    prov, cat, country = vocab.encode(probe)
    provider_input = np.zeros_like(prov) if mask_provider else prov
    emb = encode_rows(model, cont, provider_input, cat, country, desc=f"probe mask={mask_provider}")

    star = probe["star_rating"].to_numpy()
    tier = np.floor(np.where(np.isnan(star), -1, star)).clip(-1, 4).astype(int)
    return {
        "quality_acc": probe_accuracy(emb, tier),
        "provider_acc": probe_accuracy(emb, prov.astype(int)),
    }


def evaluate_cpnn(model, nn, vocab, mask_provider):
    cont_nn = build_continuous_features(nn)
    prov_nn, cat_nn, country_nn = vocab.encode(nn)
    provider_input = np.zeros_like(prov_nn) if mask_provider else prov_nn
    emb_nn = encode_rows(model, cont_nn, provider_input, cat_nn, country_nn, desc=f"cpnn mask={mask_provider}")
    return {
        f"cp_nn_k{k}": cp_nn_rate(emb_nn, nn["provider"].to_numpy(), k)
        for k in CFG["k_values"]
    }


def evaluate(model, probe, nn, vocab):
    masked_probe = evaluate_probe(model, probe, vocab, mask_provider=True)
    unmasked_probe = evaluate_probe(model, probe, vocab, mask_provider=False)
    masked_cp = evaluate_cpnn(model, nn, vocab, mask_provider=True)
    unmasked_cp = evaluate_cpnn(model, nn, vocab, mask_provider=False)
    out = {
        "quality_acc_masked": masked_probe["quality_acc"],
        "provider_acc_masked": masked_probe["provider_acc"],
        "quality_acc_unmasked": unmasked_probe["quality_acc"],
        "provider_acc_unmasked": unmasked_probe["provider_acc"],
    }
    out.update({f"{k}_masked": v for k, v in masked_cp.items()})
    out.update({f"{k}_unmasked": v for k, v in unmasked_cp.items()})
    return out


def main():
    set_seed(SEED)
    log(f"Device: {DEVICE}")
    log(f"Config: {CFG}")
    dataset = AccommodationPairDataset(n_pairs_per_epoch=CFG["n_pairs_per_epoch"])
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=CFG["num_workers"],
        generator=generator,
    )
    df = pd.read_parquet(PARQUET).reset_index(drop=True)
    vocab = AccommodationVocab(df)
    probe, nn = sample_eval_frames(df)
    log(f"Probe rows: {len(probe):,} | CP-NN rows: {len(nn):,}")

    rows = []
    for arch in tqdm(ARCH_GRID, desc="architectures"):
        log(f"\n=== {arch['name']} ===")
        set_seed(SEED)
        model = build_sweep_model(dataset.vocab, arch).to(DEVICE)
        log(f"Params: {count_params(model):,} | hidden={arch['hidden_dims']} | output={arch['output_dim']} | proj={arch['proj_dim']}")
        losses = train_short(model, loader)
        metrics = evaluate(model, probe, nn, vocab)
        row = {
            **arch,
            "params": count_params(model),
            "epochs": CFG["epochs"],
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            **metrics,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULT / "architecture_sweep.csv", index=False)
        log(pd.DataFrame([row]).to_string(index=False))

    log(f"\nSaved: {RESULT / 'architecture_sweep.csv'}")


if __name__ == "__main__":
    main()
