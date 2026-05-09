"""
Reviewer-requested validation suite.

Commands:
  provider_removed     Train contrastive encoder with provider token always masked.
  alignment_baseline   Train contrastive encoder with CORAL or MMD provider-alignment penalty.
  stratified_cpnn      Stratify CP-NN by destination size and price band using saved embeddings.
  activity_geo         Compare activity-affinity retrieval with a geography-only baseline.

Run from repo root:
  python src\\analysis\\reviewer_action_suite.py stratified_cpnn
  python src\\analysis\\reviewer_action_suite.py activity_geo
  python src\\analysis\\reviewer_action_suite.py provider_removed
  python src\\analysis\\reviewer_action_suite.py alignment_baseline --penalty coral --lambdas 0.01 0.05
"""

import argparse
import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1] / "module1_contrastive"))
from dataset import AccommodationPairDataset, AccommodationVocab, PARQUET, build_continuous_features
from model import AccommodationEncoder, ProjectionHead, nt_xent_loss

AFFINITY_DATASET_PATH = Path(__file__).parents[1] / "module3_affinity" / "dataset.py"
spec = importlib.util.spec_from_file_location("module3_affinity_dataset", AFFINITY_DATASET_PATH)
affinity_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(affinity_dataset)
ActivityDataset = affinity_dataset.ActivityDataset
load_cluster_embeddings = affinity_dataset.load_cluster_embeddings


ROOT = Path(__file__).parents[2]
RESULT = ROOT / "result" / "analysis" / "reviewer_action_suite"
RESULT.mkdir(parents=True, exist_ok=True)

EMB_PATH = ROOT / "result" / "c1_contrastive" / "property_embeddings.npy"
IDS_PATH = ROOT / "result" / "c1_contrastive" / "property_ids.npy"
ACT_EMB_PATH = ROOT / "result" / "c3_affinity" / "activity_embeddings.npy"

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


def log(msg):
    print(msg, flush=True)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InspectableContrastiveModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.encoder = AccommodationEncoder(
            n_providers=vocab.n_providers,
            n_categories=vocab.n_categories,
            n_countries=vocab.n_countries,
            hidden_dims=(512, 256),
            output_dim=128,
        )
        self.head = ProjectionHead(input_dim=128, hidden_dim=64, output_dim=32)

    def forward(self, batch):
        h1 = self.encoder(batch["cont1"], batch["provider1"], batch["category1"], batch["country1"])
        h2 = self.encoder(batch["cont2"], batch["provider2"], batch["category2"], batch["country2"])
        return h1, h2, self.head(h1), self.head(h2)

    def encode(self, cont, provider_idx, category_idx, country_idx):
        return self.encoder(cont, provider_idx, category_idx, country_idx)


def move_batch(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


def covariance(x):
    x = x - x.mean(dim=0, keepdim=True)
    denom = max(x.shape[0] - 1, 1)
    return (x.T @ x) / denom


def coral_penalty(features, providers):
    vals = []
    for p, q in combinations(torch.unique(providers).tolist(), 2):
        xp = features[providers == p]
        xq = features[providers == q]
        if xp.shape[0] < 2 or xq.shape[0] < 2:
            continue
        mean_loss = torch.mean((xp.mean(dim=0) - xq.mean(dim=0)) ** 2)
        cov_loss = torch.mean((covariance(xp) - covariance(xq)) ** 2)
        vals.append(mean_loss + cov_loss)
    if not vals:
        return features.new_tensor(0.0)
    return torch.stack(vals).mean()


def rbf_mmd_penalty(features, providers):
    vals = []
    x = nn.functional.normalize(features, dim=1)
    with torch.no_grad():
        d = torch.cdist(x[: min(len(x), 512)], x[: min(len(x), 512)]) ** 2
        gamma = 1.0 / torch.clamp(torch.median(d[d > 0]), min=1e-6)
    for p, q in combinations(torch.unique(providers).tolist(), 2):
        xp = x[providers == p]
        xq = x[providers == q]
        if xp.shape[0] < 2 or xq.shape[0] < 2:
            continue
        kpp = torch.exp(-gamma * torch.cdist(xp, xp) ** 2).mean()
        kqq = torch.exp(-gamma * torch.cdist(xq, xq) ** 2).mean()
        kpq = torch.exp(-gamma * torch.cdist(xp, xq) ** 2).mean()
        vals.append(kpp + kqq - 2 * kpq)
    if not vals:
        return features.new_tensor(0.0)
    return torch.stack(vals).mean()


def train_model(dataset, penalty_name="none", lambda_align=0.0):
    set_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    model = InspectableContrastiveModel(dataset.vocab).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    rows = []
    for epoch in tqdm(range(1, CFG["epochs"] + 1), desc=f"{penalty_name} lambda={lambda_align}"):
        model.train()
        totals = {"loss": 0.0, "nt_xent": 0.0, "align": 0.0}
        for batch in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            batch = move_batch(batch)
            opt.zero_grad()
            h1, h2, z1, z2 = model(batch)
            nt = nt_xent_loss(z1, z2, CFG["temperature"])
            h = torch.cat([h1, h2], dim=0)
            providers = torch.cat([batch["true_provider1"], batch["true_provider2"]], dim=0)
            if penalty_name == "coral":
                align = coral_penalty(h, providers)
            elif penalty_name == "mmd":
                align = rbf_mmd_penalty(h, providers)
            else:
                align = h.new_tensor(0.0)
            loss = nt + lambda_align * align
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            totals["loss"] += float(loss.item())
            totals["nt_xent"] += float(nt.item())
            totals["align"] += float(align.item())
        row = {k: v / len(loader) for k, v in totals.items()}
        row.update({"epoch": epoch, "penalty": penalty_name, "lambda_align": lambda_align})
        rows.append(row)
        tqdm.write(
            f"{penalty_name} lambda={lambda_align} epoch={epoch:02d} "
            f"loss={row['loss']:.4f} ntx={row['nt_xent']:.4f} align={row['align']:.4f}"
        )
    return model, pd.DataFrame(rows)


@torch.no_grad()
def encode_rows(model, df, vocab, mask_provider=True, chunk=4096):
    cont = build_continuous_features(df)
    prov, cat, country = vocab.encode(df)
    provider_input = np.zeros_like(prov) if mask_provider else prov
    out = np.empty((len(df), 128), dtype=np.float32)
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
    counts = pd.Series(y).value_counts()
    stratify = y if counts.min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=stratify)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    return float(accuracy_score(yte, clf.predict(Xte)))


def cp_nn_rate(emb, providers, k=10):
    if len(emb) <= k:
        return np.nan
    sim = cosine_similarity(normalize(emb, norm="l2"))
    np.fill_diagonal(sim, -np.inf)
    top = np.argpartition(sim, -k, axis=1)[:, -k:]
    return float(np.mean([(providers[top[i]] != providers[i]).mean() for i in range(len(providers))]))


def sample_eval_frames(df):
    rng = np.random.default_rng(SEED)
    probe_idx = rng.choice(len(df), min(CFG["probe_n"], len(df)), replace=False)
    nn_idx = []
    for provider in sorted(df["provider"].dropna().unique()):
        pool = df.index[df["provider"] == provider].to_numpy()
        nn_idx.append(rng.choice(pool, min(CFG["nn_per_provider"], len(pool)), replace=False))
    return df.iloc[probe_idx].reset_index(drop=True), df.iloc[np.concatenate(nn_idx)].reset_index(drop=True)


def evaluate_trained_model(model, df, vocab):
    probe, nn = sample_eval_frames(df)
    emb_probe, provider_idx = encode_rows(model, probe, vocab, mask_provider=True)
    star = probe["star_rating"].to_numpy()
    tier = np.floor(np.where(np.isnan(star), -1, star)).clip(-1, 4).astype(int)
    emb_nn, _ = encode_rows(model, nn, vocab, mask_provider=True)
    return {
        "quality_acc_masked": probe_accuracy(emb_probe, tier),
        "provider_acc_masked": probe_accuracy(emb_probe, provider_idx.astype(int)),
        "cp_nn_k10_masked": cp_nn_rate(emb_nn, nn["provider"].to_numpy(), CFG["k"]),
    }


def command_provider_removed(_args):
    log(f"Device: {DEVICE}")
    dataset = AccommodationPairDataset(
        n_pairs_per_epoch=CFG["n_pairs_per_epoch"],
        p_provider_mask=1.0,
    )
    df = pd.read_parquet(PARQUET).reset_index(drop=True)
    vocab = AccommodationVocab(df)
    model, logs = train_model(dataset, penalty_name="provider_removed", lambda_align=0.0)
    logs.to_csv(RESULT / "provider_removed_training.csv", index=False)
    metrics = evaluate_trained_model(model, df, vocab)
    out = pd.DataFrame([{**metrics, "epochs": CFG["epochs"], "params": sum(p.numel() for p in model.parameters() if p.requires_grad)}])
    out.to_csv(RESULT / "provider_removed_ablation.csv", index=False)
    print(out.to_string(index=False), flush=True)


def command_alignment_baseline(args):
    log(f"Device: {DEVICE}")
    rows = []
    for lambda_align in args.lambdas:
        dataset = AccommodationPairDataset(n_pairs_per_epoch=CFG["n_pairs_per_epoch"])
        df = pd.read_parquet(PARQUET).reset_index(drop=True)
        vocab = AccommodationVocab(df)
        model, logs = train_model(dataset, penalty_name=args.penalty, lambda_align=lambda_align)
        logs.to_csv(RESULT / f"{args.penalty}_lambda_{lambda_align:g}_training.csv", index=False)
        metrics = evaluate_trained_model(model, df, vocab)
        row = {
            "penalty": args.penalty,
            "lambda_align": lambda_align,
            "epochs": CFG["epochs"],
            "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "final_loss": logs.iloc[-1]["loss"],
            "final_nt_xent": logs.iloc[-1]["nt_xent"],
            "final_align": logs.iloc[-1]["align"],
            **metrics,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(RESULT / f"{args.penalty}_alignment_baseline.csv", index=False)
        print(pd.DataFrame([row]).to_string(index=False), flush=True)


def load_saved_embedding_frame():
    df = pd.read_parquet(
        PARQUET,
        columns=["id", "provider", "destination_display_name", "price_in_aud", "star_rating", "guest_rating", "category", "country"],
    ).reset_index(drop=True)
    ids = np.load(IDS_PATH, allow_pickle=True).astype(str)
    emb = np.load(EMB_PATH, mmap_mode="r")
    id_to_pos = {pid: i for i, pid in enumerate(ids)}
    pos = df["id"].astype(str).map(id_to_pos)
    keep = pos.notna().to_numpy()
    frame = df.loc[keep].copy().reset_index(drop=True)
    emb_sel = np.asarray(emb[pos[keep].astype(int).to_numpy()], dtype=np.float32)
    return frame, emb_sel


def command_stratified_cpnn(_args):
    frame, emb = load_saved_embedding_frame()
    rng = np.random.default_rng(SEED)
    idx = []
    for provider in sorted(frame["provider"].dropna().unique()):
        pool = frame.index[frame["provider"] == provider].to_numpy()
        idx.append(rng.choice(pool, min(1_000, len(pool)), replace=False))
    idx = np.concatenate(idx)
    sub = frame.iloc[idx].reset_index(drop=True)
    X = emb[idx]
    dest_size = frame["destination_display_name"].value_counts()
    sub["destination_size"] = sub["destination_display_name"].map(dest_size).fillna(0).astype(int)
    sub["destination_size_band"] = pd.qcut(sub["destination_size"].rank(method="first"), 3, labels=["small", "medium", "large"])
    price = sub["price_in_aud"].replace([np.inf, -np.inf], np.nan)
    sub["price_band"] = pd.qcut(price.rank(method="first"), 3, labels=["low", "mid", "high"])

    rows = [{"segment": "all", "value": "all", "n": len(sub), "cp_nn_k10": cp_nn_rate(X, sub["provider"].to_numpy())}]
    for col in ["destination_size_band", "price_band"]:
        for value, part in sub.groupby(col, observed=True):
            part_idx = part.index.to_numpy()
            rows.append({
                "segment": col,
                "value": str(value),
                "n": len(part),
                "cp_nn_k10": cp_nn_rate(X[part_idx], part["provider"].to_numpy()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULT / "stratified_cpnn.csv", index=False)
    print(out.to_string(index=False), flush=True)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def command_activity_geo(args):
    if not ACT_EMB_PATH.exists():
        raise FileNotFoundError(f"Missing activity embeddings: {ACT_EMB_PATH}")
    cluster_emb = load_cluster_embeddings()
    dataset = ActivityDataset(cluster_emb)
    act_emb = normalize(np.load(ACT_EMB_PATH), norm="l2")
    clust_names = list(cluster_emb.keys())
    clust_emb = normalize(np.stack([cluster_emb[d] for d in clust_names], axis=0), norm="l2")

    act = pd.DataFrame({
        "destination": dataset.dest_names,
        "lat": dataset.cont[:, 5] * 90.0,
        "lon": dataset.cont[:, 6] * 180.0,
    })
    acc = pd.read_parquet(PARQUET, columns=["destination_display_name", "lat", "lon"])
    centroids = (
        acc.dropna(subset=["lat", "lon"])
        .groupby("destination_display_name")[["lat", "lon"]]
        .median()
    )

    rows = []
    for ci, dest in enumerate(tqdm(clust_names, desc="activity geography baseline")):
        if dest not in centroids.index:
            continue
        learned_scores = act_emb @ clust_emb[ci]
        learned_top = np.argpartition(learned_scores, -args.k)[-args.k:]
        lat, lon = centroids.loc[dest, ["lat", "lon"]]
        dist = haversine_km(lat, lon, act["lat"].to_numpy(), act["lon"].to_numpy())
        geo_top = np.argpartition(-dist, -args.k)[-args.k:]
        geo_top = geo_top[np.argsort(dist[geo_top])]
        rows.append({
            "destination": dest,
            "learned_topk_local_rate": float(np.mean(act.iloc[learned_top]["destination"].to_numpy() == dest)),
            "geo_topk_local_rate": float(np.mean(act.iloc[geo_top]["destination"].to_numpy() == dest)),
            "learned_topk_median_km": float(np.median(dist[learned_top])),
            "geo_topk_median_km": float(np.median(dist[geo_top])),
        })
    out = pd.DataFrame(rows)
    summary = out.drop(columns=["destination"]).mean(numeric_only=True).to_frame("mean").T
    out.to_csv(RESULT / "activity_geography_baseline.csv", index=False)
    summary.to_csv(RESULT / "activity_geography_baseline_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("provider_removed").set_defaults(func=command_provider_removed)
    align = sub.add_parser("alignment_baseline")
    align.add_argument("--penalty", choices=["coral", "mmd"], default="coral")
    align.add_argument("--lambdas", nargs="+", type=float, default=[0.01, 0.05])
    align.set_defaults(func=command_alignment_baseline)
    sub.add_parser("stratified_cpnn").set_defaults(func=command_stratified_cpnn)
    act = sub.add_parser("activity_geo")
    act.add_argument("--k", type=int, default=20)
    act.set_defaults(func=command_activity_geo)
    return parser


def main():
    set_seed(SEED)
    args = build_parser().parse_args()
    args.func(args)
    log(f"Saved outputs in {RESULT}")


if __name__ == "__main__":
    main()
