#!/usr/bin/env python3
"""ShopRLVE Evaluation Script -- compare performance across collections.

Usage:
    python scripts/evaluate.py --episodes 50 --seed 42
    python scripts/evaluate.py --episodes 200 --output results/eval_results.json

Runs episodes across all collections (C1, C2, C4, C8) and compares:
    - Success rate per env
    - Average reward per env
    - Average turns per env
    - Hallucination rate per env

Prints a comparison table using rich and saves results as JSON for plotting.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from shop_rlve.server.openenv import ShopRLVEEnv
from shop_rlve.training.collections import COLLECTIONS, get_collection
from shop_rlve.training.rollout import DummyModelFn, RolloutResult, run_rollout

logger = logging.getLogger(__name__)
console = Console()


def evaluate_collection(
    collection: str,
    n_episodes: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Run evaluation episodes for a single collection.

    Args:
        collection: Collection name (C1, C2, C4, C8).
        n_episodes: Total episodes (distributed across envs in the collection).
        seed:       Random seed.

    Returns:
        Dict mapping env_id -> stats dict with keys:
            n_episodes, mean_reward, std_reward, success_rate,
            mean_turns, hallucination_rate.
    """
    env = ShopRLVEEnv(collection=collection, seed=seed)
    env.dump_dir = ""

    env_ids = get_collection(collection)
    product_ids = [p.id for p in env._products[:30]]

    per_env: dict[str, list[RolloutResult]] = defaultdict(list)

    for i in range(n_episodes):
        ep_env_id = env_ids[i % len(env_ids)]
        ep_seed = seed + i

        dummy = DummyModelFn(
            env_id=ep_env_id,
            product_ids=product_ids,
            seed=ep_seed,
        )

        result = run_rollout(
            env=env,
            model_fn=dummy,
            env_id=ep_env_id,
            seed=ep_seed,
            collect_trace=True,
        )
        per_env[ep_env_id].append(result)

    env.close()

    # Aggregate
    stats: dict[str, dict[str, Any]] = {}
    for eid in env_ids:
        results = per_env.get(eid, [])
        if not results:
            stats[eid] = {
                "n_episodes": 0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "success_rate": 0.0,
                "mean_turns": 0.0,
                "hallucination_rate": 0.0,
            }
            continue

        rewards = np.array([r.reward for r in results], dtype=np.float64)
        turns = [r.turns for r in results]
        correct = sum(1 for r in results if r.is_correct)

        # Hallucination rate from reward breakdowns
        hall_rates: list[float] = []
        for r in results:
            bd = r.reward_breakdown
            details = bd.get("details", {}) if isinstance(bd, dict) else {}
            h_rate = details.get("hallucination_rate", 0.0)
            hall_rates.append(float(h_rate))

        stats[eid] = {
            "n_episodes": len(results),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate": correct / len(results),
            "mean_turns": float(np.mean(turns)),
            "hallucination_rate": float(np.mean(hall_rates)) if hall_rates else 0.0,
        }

    return stats


def run_evaluation(
    n_episodes: int,
    seed: int,
    output_path: str | None,
) -> dict[str, Any]:
    """Run full evaluation across all collections.

    Args:
        n_episodes: Episodes per collection.
        seed:       Random seed.
        output_path: Optional path to save results JSON.

    Returns:
        Full evaluation results dict.
    """
    all_results: dict[str, Any] = {
        "n_episodes_per_collection": n_episodes,
        "seed": seed,
        "collections": {},
    }

    t0 = time.monotonic()

    for coll_name in sorted(COLLECTIONS.keys()):
        console.print(
            f"\n[bold cyan]Evaluating collection {coll_name}[/bold cyan] "
            f"({n_episodes} episodes)"
        )
        coll_stats = evaluate_collection(coll_name, n_episodes, seed)
        all_results["collections"][coll_name] = coll_stats

    elapsed = time.monotonic() - t0
    all_results["elapsed_seconds"] = elapsed

    # Print comparison table
    comparison = Table(
        title=f"Evaluation Comparison ({n_episodes} episodes/collection, {elapsed:.1f}s)",
        show_header=True,
        header_style="bold cyan",
    )
    comparison.add_column("Collection", style="bold", width=12)
    comparison.add_column("Env", style="cyan", width=10)
    comparison.add_column("Success", justify="right", width=10)
    comparison.add_column("Avg Reward", justify="right", width=11)
    comparison.add_column("Avg Turns", justify="right", width=10)
    comparison.add_column("Hall Rate", justify="right", width=10)

    for coll_name in sorted(COLLECTIONS.keys()):
        coll_stats = all_results["collections"].get(coll_name, {})
        first_row = True
        for eid in get_collection(coll_name):
            s = coll_stats.get(eid, {})
            if not s or s.get("n_episodes", 0) == 0:
                continue

            success_rate = s.get("success_rate", 0.0)
            color = "green" if success_rate > 0.5 else ("yellow" if success_rate > 0.1 else "red")

            comparison.add_row(
                coll_name if first_row else "",
                eid,
                f"[{color}]{success_rate:.2%}[/{color}]",
                f"{s.get('mean_reward', 0.0):.4f}",
                f"{s.get('mean_turns', 0.0):.2f}",
                f"{s.get('hallucination_rate', 0.0):.4f}",
            )
            first_row = False

    console.print(comparison)

    # Save results
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        console.print(f"\n[dim]Results saved to {output_path}[/dim]")

    return all_results


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="ShopRLVE Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes per collection (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_results.json",
        help="Output JSON path",
    )

    args = parser.parse_args()
    run_evaluation(
        n_episodes=args.episodes,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
