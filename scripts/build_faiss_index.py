#!/usr/bin/env python3
"""Build a FAISS index from pre-computed embeddings and save to disk.

Usage:
    python scripts/build_faiss_index.py \
        --embeddings data/amazebay-2M-embeddings \
        --out data/faiss-index \
        --factory Flat

Output:
    data/faiss-index/index.faiss   – the FAISS binary index
    data/faiss-index/ids.txt       – product IDs (one per line, positional)

Load later:
    import faiss, numpy as np
    index = faiss.read_index("data/faiss-index/index.faiss")
    ids = open("data/faiss-index/ids.txt").read().splitlines()
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import faiss
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings")
    parser.add_argument("--embeddings", type=str, default="data/amazebay-2M-embeddings",
                        help="Path to HF dataset with 'embedding' and 'parent_asin' columns")
    parser.add_argument("--out", type=str, default="data/faiss-index",
                        help="Output directory")
    parser.add_argument("--factory", type=str, default="Flat",
                        help="FAISS index factory string (Flat, IVF4096,PQ32, HNSW32, etc.)")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Build index on GPU (faster for IVF training)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load embeddings ───────────────────────────────────────
    from datasets import load_from_disk

    print(f"Loading embeddings from {args.embeddings} ...")
    t0 = time.time()
    ds = load_from_disk(args.embeddings)
    n = ds.num_rows
    dim = len(ds[0]["embedding"])
    print(f"  {n:,} vectors, dim={dim}, loaded in {time.time() - t0:.1f}s")

    # ── Step 2: Extract numpy matrix ──────────────────────────────────
    print("Extracting embedding matrix ...")
    t1 = time.time()
    embeddings = np.array(ds["embedding"], dtype=np.float32)
    ids = ds["parent_asin"]
    print(f"  Matrix shape: {embeddings.shape}, extracted in {time.time() - t1:.1f}s")

    # Verify normalization
    norms = np.linalg.norm(embeddings[:1000], axis=1)
    print(f"  L2 norms (first 1000): mean={norms.mean():.4f}, min={norms.min():.4f}, max={norms.max():.4f}")

    # ── Step 3: Build FAISS index ─────────────────────────────────────
    print(f"\nBuilding FAISS index (factory='{args.factory}') ...")
    t2 = time.time()

    index = faiss.index_factory(dim, args.factory, faiss.METRIC_INNER_PRODUCT)

    # Move to GPU if requested (speeds up IVF training)
    gpu_index = None
    if args.use_gpu:
        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            print("  Using GPU for index building")
            index = gpu_index
        except Exception as e:
            print(f"  GPU transfer failed ({e}), using CPU")

    # Train if needed (IVF, PQ, etc.)
    if not index.is_trained:
        print("  Training index ...")
        train_t = time.time()
        # Use a subset for training if dataset is large
        train_n = min(n, 500_000)
        train_data = embeddings[:train_n]
        index.train(train_data)
        print(f"  Training done in {time.time() - train_t:.1f}s")

    # Add vectors
    print(f"  Adding {n:,} vectors ...")
    add_t = time.time()
    # Add in chunks to show progress
    chunk = 200_000
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        index.add(embeddings[start:end])
        print(f"    {end:>10,} / {n:,} added")
    print(f"  All vectors added in {time.time() - add_t:.1f}s")

    # Move back to CPU if was on GPU
    if gpu_index is not None:
        index = faiss.index_gpu_to_cpu(index)

    build_time = time.time() - t2
    print(f"  Index built: {index.ntotal:,} vectors in {build_time:.1f}s")

    # ── Step 4: Save to disk ──────────────────────────────────────────
    index_path = out_path / "index.faiss"
    ids_path = out_path / "ids.txt"

    print(f"\nSaving index to {index_path} ...")
    faiss.write_index(index, str(index_path))
    index_size_mb = index_path.stat().st_size / 1e6
    print(f"  Index file: {index_size_mb:.0f} MB")

    print(f"Saving IDs to {ids_path} ...")
    with open(ids_path, "w") as f:
        for pid in ids:
            f.write(pid + "\n")
    print(f"  {len(ids):,} IDs saved")

    # ── Step 5: Validation ────────────────────────────────────────────
    print(f"\n--- Validation ---")

    # Reload index from disk
    index2 = faiss.read_index(str(index_path))
    print(f"  Reloaded index: {index2.ntotal:,} vectors")

    # Search with a sample query
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("thenlper/gte-small")
    queries = [
        "wireless bluetooth headphones noise cancelling",
        "dinosaur toy for kids",
        "organic green tea bags",
        "phone case iphone 15 pro max",
        "gaming mechanical keyboard rgb",
    ]
    q_embs = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    k = 5
    distances, indices = index2.search(q_embs, k)

    titles = ds["title"]
    for i, query in enumerate(queries):
        print(f"\n  Query: '{query}'")
        for j in range(k):
            idx = indices[i, j]
            score = distances[i, j]
            print(f"    [{j+1}] score={score:.4f}  {titles[idx][:70]}")

    total_time = time.time() - t0
    print(f"\n✅ Done! Total time: {total_time:.1f}s")
    print(f"   Index: {index_path} ({index_size_mb:.0f} MB)")
    print(f"   IDs:   {ids_path}")


if __name__ == "__main__":
    main()
