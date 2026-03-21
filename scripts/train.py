#!/usr/bin/env python3
"""EcomRLVE Training Script -- run rollouts with synthetic data.

Usage:
    python scripts/train.py --collection C1 --episodes 100 --seed 42
    python scripts/train.py --collection C8 --episodes 1000 --model Qwen/Qwen2.5-1.5B-Instruct

Creates a EcomRLVEEnv with a synthetic catalog and runs batch rollouts
with DummyModelFn (or a real model if ``--model`` is specified).  Prints
per-env stats and saves results to JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from ecom_rlve.server.openenv import EcomRLVEEnv
from ecom_rlve.training.collections import COLLECTIONS, get_collection
from ecom_rlve.training.rollout import DummyModelFn, RolloutResult, run_rollout

logger = logging.getLogger(__name__)
console = Console()


def _build_dummy_model(
    env_ids: list[str],
    product_ids: list[str],
    seed: int,
) -> DummyModelFn:
    """Build a DummyModelFn that cycles through env_ids.

    Args:
        env_ids:     List of environment IDs.
        product_ids: Product IDs to recommend from.
        seed:        Random seed.

    Returns:
        DummyModelFn instance.
    """
    # The dummy model needs an env_id; for multi-env collections we default
    # to the first env since DummyModelFn uses it for the answer schema.
    return DummyModelFn(
        env_id=env_ids[0],
        product_ids=product_ids,
        seed=seed,
    )


def run_training(
    collection: str,
    n_episodes: int,
    seed: int,
    output_path: str | None,
) -> dict[str, Any]:
    """Run training rollouts and return aggregated results.

    Args:
        collection: Collection name (C1, C2, C4, C8).
        n_episodes: Number of episodes to run.
        seed:       Master random seed.
        output_path: Optional path to save JSON results.

    Returns:
        Dict with per-env stats and overall stats.
    """
    console.print(
        f"[bold cyan]Training[/bold cyan] collection=[bold]{collection}[/bold] "
        f"episodes=[bold]{n_episodes}[/bold] seed=[bold]{seed}[/bold]"
    )

    t0 = time.monotonic()

    env = EcomRLVEEnv(collection=collection, seed=seed)
    env.dump_dir = ""  # Disable trace dumping

    env_ids = get_collection(collection)
    product_ids = [p.id for p in env._products[:30]]

    # Per-env accumulators
    per_env: dict[str, list[RolloutResult]] = defaultdict(list)

    for i in range(n_episodes):
        ep_env_id = env_ids[i % len(env_ids)]
        ep_seed = seed + i

        # Create a fresh DummyModelFn per episode with the correct env_id
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
        )
        per_env[ep_env_id].append(result)

        if (i + 1) % max(1, n_episodes // 10) == 0:
            console.print(f"  [{i + 1}/{n_episodes}] episodes completed")

    elapsed = time.monotonic() - t0

    # Aggregate stats
    stats_table = Table(
        title=f"Training Results: {collection} ({n_episodes} episodes, {elapsed:.1f}s)",
        show_header=True,
        header_style="bold cyan",
    )
    stats_table.add_column("Env", style="cyan", width=10)
    stats_table.add_column("Episodes", justify="right", width=10)
    stats_table.add_column("Mean Reward", justify="right", width=12)
    stats_table.add_column("Std Reward", justify="right", width=12)
    stats_table.add_column("Success Rate", justify="right", width=13)
    stats_table.add_column("Mean Turns", justify="right", width=11)

    all_results: dict[str, Any] = {
        "collection": collection,
        "n_episodes": n_episodes,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "per_env": {},
    }

    total_rewards: list[float] = []
    total_correct = 0

    for eid in env_ids:
        results = per_env.get(eid, [])
        if not results:
            continue

        rewards = [r.reward for r in results]
        turns = [r.turns for r in results]
        correct = sum(1 for r in results if r.is_correct)

        rewards_arr = np.array(rewards, dtype=np.float64)
        mean_r = float(np.mean(rewards_arr))
        std_r = float(np.std(rewards_arr))
        mean_t = float(np.mean(turns))
        success = correct / len(results)

        total_rewards.extend(rewards)
        total_correct += correct

        env_stats = {
            "n_episodes": len(results),
            "mean_reward": mean_r,
            "std_reward": std_r,
            "success_rate": success,
            "mean_turns": mean_t,
        }
        all_results["per_env"][eid] = env_stats

        stats_table.add_row(
            eid,
            str(len(results)),
            f"{mean_r:.4f}",
            f"{std_r:.4f}",
            f"{success:.2%}",
            f"{mean_t:.2f}",
        )

    # Overall row
    if total_rewards:
        overall_arr = np.array(total_rewards, dtype=np.float64)
        stats_table.add_row(
            "[bold]TOTAL[/bold]",
            str(n_episodes),
            f"[bold]{float(np.mean(overall_arr)):.4f}[/bold]",
            f"{float(np.std(overall_arr)):.4f}",
            f"[bold]{total_correct / n_episodes:.2%}[/bold]",
            "",
        )

    console.print(stats_table)

    # Difficulty progression
    engine_state = env.adaptive_engine.get_all_states()
    diff_table = Table(title="Difficulty Progression", show_header=True)
    diff_table.add_column("Env", style="cyan", width=10)
    diff_table.add_column("Low", justify="right", width=6)
    diff_table.add_column("High", justify="right", width=6)

    for eid in env_ids:
        state = engine_state.get(eid)
        if state:
            diff_table.add_row(eid, str(state.low), str(state.high))

    console.print(diff_table)

    # Save results
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        console.print(f"\n[dim]Results saved to {output_path}[/dim]")

    env.close()
    return all_results


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="EcomRLVE Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="C1",
        choices=sorted(COLLECTIONS.keys()),
        help="Environment collection (default: C1)",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: DummyModelFn)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/train_results.json",
        help="Output JSON path",
    )

    args = parser.parse_args()

    if args.model is not None:
        console.print(
            f"[yellow]Note: --model {args.model} specified but real model "
            f"integration is not yet implemented.  Using DummyModelFn.[/yellow]"
        )

    run_training(
        collection=args.collection,
        n_episodes=args.episodes,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
