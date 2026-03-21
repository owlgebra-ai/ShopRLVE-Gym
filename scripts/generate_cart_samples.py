#!/usr/bin/env python3
"""Generate diverse E_CART data samples at each difficulty level.

For each difficulty d in [0..10], generates N episodes via OpenEnv.reset()
and captures the full episode initialization data:
  - Trigger dialog (initial user message from Ollama LLM)
  - Persona weights (Dirichlet-sampled)
  - Verifiable target products (IDs, titles, variants, quantities)
  - Difficulty parameters (12-axis vector)
  - Synthetic variants injected into the catalog
  - Visit history (recently viewed products)
  - All ProblemParams metadata

Usage:
    uv run python scripts/generate_cart_samples.py
    uv run python scripts/generate_cart_samples.py --samples 5 --min-d 0 --max-d 5
    uv run python scripts/generate_cart_samples.py --output data/cart_samples.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ecom_rlve.data.catalog_loader import generate_synthetic_catalog
from ecom_rlve.difficulty.mapping import DifficultyParams, map_difficulty
from ecom_rlve.server.openenv import EcomRLVEEnv
from ecom_rlve.simulator.llm_backend import is_ollama_available, verbalize_cart_request
from ecom_rlve.simulator.persona import PersonaWeights

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("generate_cart_samples")


def extract_sample(
    env: EcomRLVEEnv,
    difficulty: int,
    seed: int,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Run a single reset() for CART at the given difficulty and extract all data."""
    obs = env.reset(env_id="CART", difficulty=difficulty, seed=seed)
    state = env._state
    assert state is not None

    problem = state.hidden_goal
    diff_params = map_difficulty(difficulty)
    persona = state.persona_weights

    # The server's reset() uses template-based initial messages.
    # Call the LLM verbalization path directly to get natural language.
    template_message = obs.conversation[0]["content"] if obs.conversation else ""
    llm_message = None
    mentioned_attrs: set[str] = set()
    omitted_attrs: set[str] = set()

    if use_llm:
        details = problem.extra.get("item_details", [])
        p_missing = problem.extra.get("p_missing", 0.0)
        text, mentioned, omitted = verbalize_cart_request(
            item_details=details,
            p_missing=p_missing,
            seed=seed + 1,
        )
        if text is not None:
            llm_message = text
            mentioned_attrs = mentioned
            omitted_attrs = omitted

    # Collect target product details
    target_products = []
    for pid in problem.target_product_ids:
        product = state.products_by_id.get(pid)
        if product is not None:
            target_products.append({
                "product_id": product.id,
                "title": product.title,
                "category": product.cat,
                "brand": product.brand,
                "price": product.price,
                "rating": product.rating,
                "stock_qty": product.stock_qty,
                "features": product.features[:5] if product.features else [],
                "attrs": dict(product.attrs) if product.attrs else {},
            })

    # Collect synthetic variants
    synthetic_variants = []
    for vd in problem.extra.get("synthetic_variants", []):
        synthetic_variants.append(vd if isinstance(vd, dict) else vd.model_dump())

    # Build the sample
    sample: dict[str, Any] = {
        # Identifiers
        "env_id": "CART",
        "difficulty": difficulty,
        "seed": seed,

        # Trigger dialog — LLM-generated (Ollama) and template fallback
        "initial_user_message": llm_message if llm_message else template_message,
        "llm_message": llm_message,
        "template_message": template_message,
        "message_source": "ollama_llm" if llm_message else "template",

        # Persona
        "persona_weights": persona.as_dict() if persona else None,

        # Difficulty parameters (12-axis vector)
        "difficulty_params": {
            "m": diff_params.m_val,
            "k_rec": diff_params.k_rec_val,
            "T_max": diff_params.T_max_val,
            "p_missing": round(diff_params.p_missing_val, 4),
            "p_noise": round(diff_params.p_noise_val, 4),
            "p_switch": round(diff_params.p_switch_val, 4),
            "top_k": diff_params.top_k_val,
            "eps_rank": round(diff_params.eps_rank_val, 4),
            "p_oos": round(diff_params.p_oos_val, 4),
            "H_orders": diff_params.H_orders_val,
            "B_branch": diff_params.B_branch_val,
            "B_tool": diff_params.B_tool_val,
        },

        # CART-specific difficulty axes
        "cart_params": {
            "n_items": problem.extra.get("n_items"),
            "p_var": round(problem.extra.get("p_var", 0.0), 4),
            "p_qty": round(problem.extra.get("p_qty", 0.0), 4),
        },

        # Verifiable ground truth
        "required_items": problem.extra.get("required_items", {}),
        "variant_reqs": problem.extra.get("variant_reqs", {}),

        # Target products (full details)
        "target_products": target_products,

        # Item details (what the user simulator knows)
        "item_details": problem.extra.get("item_details", []),

        # Synthetic variants injected for this episode
        "synthetic_variants": synthetic_variants,

        # Visit history (user.get_visit_history returns this)
        "visit_history": problem.extra.get("visit_history", []),

        # LLM verbalization metadata
        "mentioned_attrs": list(mentioned_attrs or problem.extra.get("mentioned_attrs", [])),
        "omitted_attrs": list(omitted_attrs or problem.extra.get("omitted_attrs", [])),
    }

    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate E_CART data samples per difficulty")
    parser.add_argument("--samples", type=int, default=10, help="Samples per difficulty level")
    parser.add_argument("--min-d", type=int, default=0, help="Min difficulty (inclusive)")
    parser.add_argument("--max-d", type=int, default=10, help="Max difficulty (inclusive)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--catalog-size", type=int, default=1000, help="Synthetic catalog size")
    parser.add_argument("--output", type=str, default="data/cart_samples.json", help="Output path")
    parser.add_argument("--no-llm", action="store_true", help="Skip Ollama LLM verbalization")
    args = parser.parse_args()

    # Check Ollama availability
    use_llm = not args.no_llm and is_ollama_available()

    print("=" * 70)
    print(f"Generating {args.samples} E_CART samples per difficulty d=[{args.min_d}..{args.max_d}]")
    print(f"  LLM verbalization: {'ON (Ollama qwen3.5)' if use_llm else 'OFF (template fallback)'}")
    print("=" * 70)

    # Initialize OpenEnv with CART environment
    t0 = time.time()
    env = EcomRLVEEnv(
        collection="C8",
        seed=args.seed,
        config={
            "n_synthetic_products": args.catalog_size,
            "disclose_env_id": True,
            "disclose_difficulty": True,
        },
    )
    t_init = time.time() - t0
    print(f"\n  Environment initialized in {t_init:.1f}s "
          f"(catalog: {args.catalog_size} products)\n")

    all_samples: dict[str, list[dict[str, Any]]] = {}
    total_generated = 0
    total_time = 0.0

    for d in range(args.min_d, args.max_d + 1):
        diff_params = map_difficulty(d)
        print(f"  Difficulty {d:>2}: n_items={min(5, 1 + d // 3)}, "
              f"T_max={diff_params.T_max_val}, "
              f"p_var={round(float(1 / (1 + __import__('math').exp(-(d - 2) / 1.5))), 2)}, "
              f"p_qty={min(0.5, 0.1 * d):.1f}")

        samples: list[dict[str, Any]] = []
        for i in range(args.samples):
            ep_seed = args.seed * 10000 + d * 100 + i
            t_start = time.time()
            try:
                sample = extract_sample(env, difficulty=d, seed=ep_seed, use_llm=use_llm)
                elapsed = time.time() - t_start
                total_time += elapsed
                samples.append(sample)
                total_generated += 1
            except Exception as exc:
                logger.error("  Failed d=%d sample=%d seed=%d: %s", d, i, ep_seed, exc)

        all_samples[f"d{d}"] = samples

        # Print a brief summary for this difficulty
        if samples:
            avg_items = sum(s["cart_params"]["n_items"] or 0 for s in samples) / len(samples)
            has_variants = sum(1 for s in samples if any(v is not None for v in s["variant_reqs"].values()))
            llm_count = sum(1 for s in samples if s["message_source"] == "ollama_llm")
            msg_lens = [len(s["initial_user_message"]) for s in samples]
            print(f"           → {len(samples)} samples, "
                  f"avg_items={avg_items:.1f}, "
                  f"with_variants={has_variants}/{len(samples)}, "
                  f"llm={llm_count}/{len(samples)}, "
                  f"msg_len=[{min(msg_lens)}..{max(msg_lens)}]")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "env_id": "CART",
            "min_difficulty": args.min_d,
            "max_difficulty": args.max_d,
            "samples_per_difficulty": args.samples,
            "total_samples": total_generated,
            "base_seed": args.seed,
            "catalog_size": args.catalog_size,
            "generation_time_s": round(total_time, 2),
            "llm_verbalization": use_llm,
            "llm_model": "qwen3.5 (Ollama)" if use_llm else None,
        },
        "samples_by_difficulty": all_samples,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"  Done! {total_generated} samples written to {output_path}")
    print(f"  Total generation time: {total_time:.1f}s")
    print(f"{'=' * 70}")

    # Print a preview of one sample
    if all_samples.get("d0"):
        print("\n--- Sample Preview (d=0, first sample) ---")
        s = all_samples["d0"][0]
        print(f"  Initial message: {s['initial_user_message'][:200]}")
        print(f"  Persona: {s['persona_weights']}")
        print(f"  Required items: {s['required_items']}")
        print(f"  Variant reqs: {s['variant_reqs']}")
        print(f"  Target products: {len(s['target_products'])} items")
        for tp in s['target_products']:
            print(f"    - {tp['title'][:60]} (${tp['price']}, {tp['category']})")
        print(f"  Synthetic variants: {len(s['synthetic_variants'])} total")
        print(f"  Visit history: {len(s['visit_history'])} entries")


if __name__ == "__main__":
    main()
