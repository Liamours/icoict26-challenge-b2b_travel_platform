import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "module1_contrastive"))
from dataset import AccommodationVocab, build_continuous_features, PARQUET
from model import build_model

RESULT = Path(__file__).parents[2] / "result" / "c2_validation"
RESULT.mkdir(parents=True, exist_ok=True)

C1_RESULT = Path(__file__).parents[2] / "result" / "c1_contrastive"
TRANSACTIONS = Path(__file__).parents[2] / "dataset" / "preprocessed" / "transactions.parquet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embed_chunk(model, cont, provider_idx, category_idx, country_idx, device, chunk=5_000):
    N = len(cont)
    embeddings = np.empty((N, 128), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            idx = np.arange(start, end)
            cont_t = torch.from_numpy(cont[idx]).to(device)
            prov_t = torch.zeros(len(idx), dtype=torch.long, device=device)
            cat_t = torch.from_numpy(category_idx[idx].astype(np.int64)).to(device)
            cntry_t = torch.from_numpy(country_idx[idx].astype(np.int64)).to(device)
            h = model.encode(cont_t, prov_t, cat_t, cntry_t)
            embeddings[start:end] = h.cpu().numpy()
    return embeddings


def main():
    print(f"Device: {DEVICE}")

    # ── Transactions ─────────────────────────────────────────────────────────
    print("Loading transactions...")
    tx = pd.read_parquet(TRANSACTIONS)
    print(f"Shape: {tx.shape} | Cols: {tx.columns.tolist()}")

    tx_agg = (
        tx.groupby("property_id")
        .agg(
            total_transactions=("transaction_count", "sum"),
            months_active=("year_month", "nunique"),
        )
        .reset_index()
    )
    tx_agg["property_id"] = tx_agg["property_id"].astype(str)
    tx_ids_set = set(tx_agg["property_id"].values)
    print(f"Unique property IDs: {len(tx_ids_set)}")
    print(f"months_active stats:\n{tx_agg['months_active'].describe()}")

    # ── Accommodations ────────────────────────────────────────────────────────
    print("\nLoading accommodations...")
    df = pd.read_parquet(PARQUET)
    df["id_str"] = df["id"].astype(str)
    matched = df[df["id_str"].isin(tx_ids_set)].copy().reset_index(drop=True)
    print(f"Matched: {len(matched)} / {len(tx_ids_set)} ({100*len(matched)/len(tx_ids_set):.1f}%)")

    # ── Vocab + features ──────────────────────────────────────────────────────
    vocab = AccommodationVocab(df)
    cont_matched = build_continuous_features(matched)
    prov_m, cat_m, cntry_m = vocab.encode(matched)

    # ── Load C1 encoder ───────────────────────────────────────────────────────
    print("Loading C1 encoder...")

    class _FakeVocab:
        pass

    fv = _FakeVocab()
    fv.n_providers = vocab.n_providers
    fv.n_categories = vocab.n_categories
    fv.n_countries = vocab.n_countries

    model = build_model(fv).to(DEVICE)
    model.encoder.load_state_dict(
        torch.load(C1_RESULT / "encoder_best.pth", map_location=DEVICE)
    )
    model.eval()

    # ── Embed matched properties ───────────────────────────────────────────────
    print("Embedding matched properties...")
    embeddings = embed_chunk(model, cont_matched, prov_m, cat_m, cntry_m, DEVICE)
    print(f"Embeddings shape: {embeddings.shape}")

    # ── Quality score: cosine sim to high-quality centroid ────────────────────
    # Build centroid from ALL accommodations with star_rating >= 3 (tier 3+4)
    print("Building high-quality centroid from full accommodations...")
    star = df["star_rating"].values
    hq_mask = (~np.isnan(star)) & (star >= 3.0)
    print(f"High-quality properties (star≥3): {hq_mask.sum():,} / {len(df):,}")

    if hq_mask.sum() < 100:
        print("WARNING: too few high-quality properties, falling back to L2 norm")
        quality_scores = np.linalg.norm(embeddings, axis=1).astype(np.float32)
    else:
        # Embed high-quality subset in chunks
        hq_idx = np.where(hq_mask)[0]
        # Sample up to 50k for centroid to keep memory reasonable
        if len(hq_idx) > 50_000:
            hq_idx = hq_idx[np.random.default_rng(42).choice(len(hq_idx), 50_000, replace=False)]

        hq_df = df.iloc[hq_idx].reset_index(drop=True)
        cont_hq = build_continuous_features(hq_df)
        prov_hq, cat_hq, cntry_hq = vocab.encode(hq_df)

        print(f"Embedding {len(hq_idx):,} high-quality properties for centroid...")
        hq_emb = embed_chunk(model, cont_hq, prov_hq, cat_hq, cntry_hq, DEVICE)

        centroid = hq_emb.mean(axis=0)  # (128,)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

        # Cosine similarity = dot(emb_normalized, centroid_normalized)
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        quality_scores = emb_norm @ centroid_norm  # (N,)
        quality_scores = quality_scores.astype(np.float32)

        print(f"Quality score (cosine to HQ centroid) stats:")
        print(f"  mean={quality_scores.mean():.4f}  std={quality_scores.std():.4f}  "
              f"min={quality_scores.min():.4f}  max={quality_scores.max():.4f}")

    # ── Build result df ───────────────────────────────────────────────────────
    matched_ids = matched["id"].astype(str).values
    result_df = pd.DataFrame({
        "property_id": matched_ids,
        "provider": matched["provider"].values,
        "quality_score": quality_scores,
        "quality_score_l2": np.linalg.norm(embeddings, axis=1),  # keep for reference
    })
    result_df = result_df.merge(tx_agg, on="property_id", how="left")
    result_df["total_transactions"] = result_df["total_transactions"].fillna(0).astype(int)
    result_df["months_active"] = result_df["months_active"].fillna(0).astype(int)

    print(f"\nFinal df shape: {result_df.shape}")
    print(f"quality_score stats:\n{result_df['quality_score'].describe()}")
    print(f"\nmonths_active stats:\n{result_df['months_active'].describe()}")
    print(f"\nPer provider:")
    print(result_df.groupby("provider")[["quality_score", "total_transactions", "months_active"]].mean())

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(RESULT / "transaction_property_embeddings.npy", embeddings)
    np.save(RESULT / "transaction_property_ids.npy", matched_ids)
    result_df.to_csv(RESULT / "transaction_quality_scores.csv", index=False)
    print(f"\nSaved to {RESULT}")

    # ── Coverage report ───────────────────────────────────────────────────────
    print("\n--- Coverage Report ---")
    print(f"Transaction property IDs: {len(tx_ids_set)}")
    print(f"Matched: {len(matched)} ({100*len(matched)/len(tx_ids_set):.1f}%)")
    print(f"Unmatched: {len(tx_ids_set)-len(matched)} ({100*(len(tx_ids_set)-len(matched))/len(tx_ids_set):.1f}%)")
    unmatched = tx_ids_set - set(matched_ids)
    tx["id_str"] = tx["property_id"].astype(str)
    if "provider" in tx.columns:
        print(f"Unmatched by provider:\n{tx[tx['id_str'].isin(unmatched)]['provider'].value_counts()}")


if __name__ == "__main__":
    main()
