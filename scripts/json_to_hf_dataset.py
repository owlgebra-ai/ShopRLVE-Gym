#!/usr/bin/env python3
"""Convert cart_samples_llm.json to a Hugging Face Dataset and save to disk.

Usage:
    uv run python scripts/json_to_hf_dataset.py [--input FILE] [--output DIR] [--push REPO]

Options:
    --input   Path to JSON file (default: data/cart_samples_llm.json)
    --output  Directory to save HF dataset (default: data/cart_samples_hf)
    --push    HuggingFace repo id to push to (e.g. "user/ecom-rlve-cart-samples")
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets


def flatten_samples(json_path: str | Path) -> list[dict]:
    """Flatten the nested JSON into a list of flat records for tabular storage.

    Complex nested fields (dicts/lists) are stored as JSON strings so the
    dataset schema stays clean and works with Arrow/Parquet.
    """
    with open(json_path) as f:
        data = json.load(f)

    rows: list[dict] = []
    for _d_key, samples in data["samples_by_difficulty"].items():
        for s in samples:
            row = {
                # ---- Scalars ----
                "env_id": s["env_id"],
                "difficulty": s["difficulty"],
                "seed": s["seed"],
                "initial_user_message": s["initial_user_message"],
                "llm_message": s["llm_message"],
                "template_message": s["template_message"],
                "message_source": s["message_source"],
                # ---- Persona (flatten top-level keys) ----
                "w_price": s["persona_weights"]["w_price"],
                "w_rating": s["persona_weights"]["w_rating"],
                "w_ship": s["persona_weights"]["w_ship"],
                "w_brand": s["persona_weights"]["w_brand"],
                "w_similarity": s["persona_weights"]["w_similarity"],
                # ---- Difficulty params (flatten) ----
                "n_constraints": s["difficulty_params"]["m"],
                "k_rec": s["difficulty_params"]["k_rec"],
                "T_max": s["difficulty_params"]["T_max"],
                "p_missing": s["difficulty_params"]["p_missing"],
                "p_noise": s["difficulty_params"]["p_noise"],
                "p_switch": s["difficulty_params"]["p_switch"],
                "top_k": s["difficulty_params"]["top_k"],
                "eps_rank": s["difficulty_params"]["eps_rank"],
                "p_oos": s["difficulty_params"]["p_oos"],
                "H_orders": s["difficulty_params"]["H_orders"],
                "B_branch": s["difficulty_params"]["B_branch"],
                "B_tool": s["difficulty_params"]["B_tool"],
                # ---- Cart params (flatten) ----
                "n_items": s["cart_params"]["n_items"],
                "p_var": s["cart_params"]["p_var"],
                "p_qty": s["cart_params"]["p_qty"],
                # ---- Structured fields (as JSON strings) ----
                "required_items": json.dumps(s["required_items"]),
                "variant_reqs": json.dumps(s["variant_reqs"]),
                "target_products": json.dumps(s["target_products"]),
                "item_details": json.dumps(s["item_details"]),
                "synthetic_variants": json.dumps(s["synthetic_variants"]),
                "visit_history": json.dumps(s["visit_history"]),
                # ---- Attr tracking ----
                "mentioned_attrs": s["mentioned_attrs"],
                "omitted_attrs": s["omitted_attrs"],
            }
            rows.append(row)

    return rows


def build_dataset(rows: list[dict]) -> datasets.Dataset:
    """Create a typed HF Dataset from flat rows."""
    features = datasets.Features(
        {
            "env_id": datasets.Value("string"),
            "difficulty": datasets.Value("int32"),
            "seed": datasets.Value("int64"),
            "initial_user_message": datasets.Value("string"),
            "llm_message": datasets.Value("string"),
            "template_message": datasets.Value("string"),
            "message_source": datasets.Value("string"),
            # Persona weights
            "w_price": datasets.Value("float64"),
            "w_rating": datasets.Value("float64"),
            "w_ship": datasets.Value("float64"),
            "w_brand": datasets.Value("float64"),
            "w_similarity": datasets.Value("float64"),
            # Difficulty params
            "n_constraints": datasets.Value("int32"),
            "k_rec": datasets.Value("int32"),
            "T_max": datasets.Value("int32"),
            "p_missing": datasets.Value("float64"),
            "p_noise": datasets.Value("float64"),
            "p_switch": datasets.Value("float64"),
            "top_k": datasets.Value("int32"),
            "eps_rank": datasets.Value("float64"),
            "p_oos": datasets.Value("float64"),
            "H_orders": datasets.Value("int32"),
            "B_branch": datasets.Value("int32"),
            "B_tool": datasets.Value("int32"),
            # Cart params
            "n_items": datasets.Value("int32"),
            "p_var": datasets.Value("float64"),
            "p_qty": datasets.Value("float64"),
            # Structured (JSON strings)
            "required_items": datasets.Value("string"),
            "variant_reqs": datasets.Value("string"),
            "target_products": datasets.Value("string"),
            "item_details": datasets.Value("string"),
            "synthetic_variants": datasets.Value("string"),
            "visit_history": datasets.Value("string"),
            # Attr tracking (lists of strings)
            "mentioned_attrs": datasets.Sequence(datasets.Value("string")),
            "omitted_attrs": datasets.Sequence(datasets.Value("string")),
        }
    )
    return datasets.Dataset.from_list(rows, features=features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cart samples JSON to HF Dataset")
    parser.add_argument("--input", default="data/cart_samples_llm.json", help="Input JSON file")
    parser.add_argument("--output", default="data/cart_samples_hf", help="Output directory for HF dataset")
    parser.add_argument("--push", default=None, help="HuggingFace repo id to push (e.g. user/repo)")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    rows = flatten_samples(args.input)
    print(f"  Flattened {len(rows)} samples")

    ds = build_dataset(rows)
    print(f"\nDataset:\n{ds}")
    print(f"\nFeatures:\n{ds.features}")

    # Save to disk (Arrow format)
    output_path = Path(args.output)
    ds.save_to_disk(str(output_path))
    print(f"\nSaved to {output_path}/")

    # Also save as Parquet for easy sharing
    parquet_path = output_path / "cart_samples.parquet"
    ds.to_parquet(str(parquet_path))
    print(f"Saved Parquet to {parquet_path}")

    # Optionally push to HuggingFace Hub
    if args.push:
        print(f"\nPushing to HuggingFace Hub: {args.push} ...")
        ds.push_to_hub(args.push, split="train")
        print("Done!")

    # Quick preview
    print("\n--- Preview (first 3 rows) ---")
    for i in range(min(3, len(ds))):
        row = ds[i]
        print(f"\n[{i}] d={row['difficulty']} seed={row['seed']}")
        print(f"    msg: {row['llm_message'][:80]}...")
        print(f"    items: {row['required_items']}")
        print(f"    variants: {row['variant_reqs']}")


if __name__ == "__main__":
    main()
