#!/usr/bin/env python3
"""Encode 2M Amazebay product titles with gte-small and save as a HF dataset.

Usage:
    python3 scripts/build_embeddings.py \
        --catalog data/amazebay-2M \
        --out data/amazebay-2M-embeddings \
        --batch-size 1024 \
        --device cuda:0

Output: same columns as input + 'embedding' (list[float], 384-dim, L2-normalized).

Load with:
    from datasets import load_from_disk
    ds = load_from_disk("data/amazebay-2M-embeddings")
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode product titles with gte-small")
    parser.add_argument("--catalog", type=str, default="data/amazebay-2M")
    parser.add_argument("--model", type=str, default="thenlper/gte-small")
    parser.add_argument("--out", type=str, default="data/amazebay-2M-embeddings")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    from datasets import load_from_disk
    from sentence_transformers import SentenceTransformer

    # ── Load catalog ──────────────────────────────────────────────────
    print(f"Loading catalog from {args.catalog} ...")
    ds = load_from_disk(args.catalog)
    print(f"  {ds.num_rows:,} rows, columns: {ds.column_names}")

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading model: {args.model} on {args.device} ...")
    model = SentenceTransformer(args.model, device=args.device)
    dim = model.get_sentence_embedding_dimension()
    print(f"  dim={dim}")

    # ── Encode via .map() ─────────────────────────────────────────────
    def encode_batch(batch):
        titles = [t or "" for t in batch["title"]]
        embs = model.encode(
            titles,
            normalize_embeddings=True,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return {"embedding": embs.astype(np.float32).tolist()}

    print(f"\nEncoding titles (batch_size={args.batch_size}) ...")
    t0 = time.time()
    ds = ds.map(encode_batch, batched=True, batch_size=args.batch_size, desc="Encoding")
    elapsed = time.time() - t0
    print(f"  Done: {ds.num_rows:,} rows in {elapsed / 60:.1f} min "
          f"({ds.num_rows / elapsed:,.0f} texts/s)")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_path} ...")
    ds.save_to_disk(str(out_path))
    print(f"  Saved. Columns: {ds.column_names}")

    # ── Validate ──────────────────────────────────────────────────────
    sample = np.array(ds[0]["embedding"])
    print(f"\n--- Validation ---")
    print(f"  Shape: ({ds.num_rows}, {len(ds[0]['embedding'])})")
    print(f"  Sample norm: {np.linalg.norm(sample):.4f}")
    print(f"  Sample title: {ds[0]['title'][:80]}")

    total = time.time() - t0
    print(f"\nTotal: {total / 60:.1f} min")


if __name__ == "__main__":
    main()
