#!/usr/bin/env python3
"""Download the first 2M rows of Amazebay-catalog and save locally as a HF dataset.

Usage:
    uv run python scripts/make_mini_catalog.py [--max-rows 2000000] [--out data/amazebay-2M]

This streams the dataset so it never loads the full 1.97M+ dataset into memory.
The output is saved in Arrow/Parquet format and can be loaded with:
    ds = load_dataset("data/amazebay-2M")
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from datasets import Dataset, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a mini Amazebay catalog")
    parser.add_argument(
        "--max-rows", type=int, default=2_000_000,
        help="Number of rows to take (default: 2,000,000)",
    )
    parser.add_argument(
        "--out", type=str, default="data/amazebay-2M",
        help="Output directory for the saved dataset",
    )
    parser.add_argument(
        "--source", type=str, default="thebajajra/Amazebay-catalog",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10_000,
        help="Rows to accumulate before printing progress",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    print(f"Streaming {args.source} → first {args.max_rows:,} rows → {out_path}")

    t0 = time.time()
    ds_stream = load_dataset(args.source, split="train", streaming=True)

    # Collect rows in batches for progress reporting
    rows: list[dict] = []
    for i, row in enumerate(ds_stream):
        if i >= args.max_rows:
            break
        rows.append(row)
        if (i + 1) % args.batch_size == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (args.max_rows - i - 1) / rate if rate > 0 else 0
            print(
                f"  {i + 1:>10,} / {args.max_rows:,} rows  "
                f"({100 * (i + 1) / args.max_rows:5.1f}%)  "
                f"{rate:,.0f} rows/s  ETA {eta / 60:.1f}min"
            )

    elapsed = time.time() - t0
    print(f"\nCollected {len(rows):,} rows in {elapsed:.1f}s")

    # Convert to HF Dataset and save
    print("Converting to Arrow dataset...")
    ds = Dataset.from_list(rows)
    print(f"  columns: {ds.column_names}")
    print(f"  num_rows: {ds.num_rows:,}")

    print(f"Saving to {out_path} ...")
    out_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))

    # Also save as parquet for easy inspection
    parquet_path = out_path / "catalog.parquet"
    ds.to_parquet(str(parquet_path))
    print(f"Also saved as parquet: {parquet_path}")

    # Quick stats
    print("\n--- Quick Stats ---")
    cats = {}
    valid_prices = 0
    for row in rows:
        cat = row.get("main_category", "unknown")
        cats[cat] = cats.get(cat, 0) + 1
        p = row.get("price")
        if p and str(p) != "None":
            valid_prices += 1

    print(f"Total rows:     {len(rows):,}")
    print(f"Valid prices:   {valid_prices:,} ({100 * valid_prices / len(rows):.1f}%)")
    print(f"Categories ({len(cats)}):")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        cat_name = str(cat) if cat is not None else "(None)"
        print(f"  {cat_name:45s} {cnt:>10,}  ({100 * cnt / len(rows):5.1f}%)")

    total_sec = time.time() - t0
    print(f"\nDone in {total_sec / 60:.1f} min total.")


if __name__ == "__main__":
    main()
