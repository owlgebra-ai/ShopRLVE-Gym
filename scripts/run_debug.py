#!/usr/bin/env python3
"""EcomRLVE Debug CLI -- main entry point for environment debugging.

Usage:
    python scripts/run_debug.py probe --env PD --difficulty 3 --episodes 20
    python scripts/run_debug.py validate --env all --max-difficulty 10
    python scripts/run_debug.py episode --env PD --difficulty 0 --seed 42 --verbose
    python scripts/run_debug.py difficulty --env PD --range 0-15
    python scripts/run_debug.py smoke-test

Subcommands:
    probe       Run probe_env() and print reward statistics.
    validate    Run solvability checks across envs and difficulties.
    episode     Run one episode with DummyModelFn and show full trace.
    difficulty  Show the difficulty parameter table for a range of d values.
    smoke-test  Quick test that all 8 envs can reset+step without crashing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def cmd_probe(args: argparse.Namespace) -> None:
    """Run probe_env() and print results."""
    from ecom_rlve.debug.replay import probe_env

    console.print(
        f"[bold cyan]Probing[/bold cyan] env=[bold]{args.env}[/bold] "
        f"difficulty=[bold]{args.difficulty}[/bold] "
        f"episodes=[bold]{args.episodes}[/bold]"
    )

    t0 = time.monotonic()
    stats: dict[str, Any] = probe_env(
        env_id=args.env,
        difficulty=args.difficulty,
        n_episodes=args.episodes,
        seed=args.seed,
    )
    elapsed = time.monotonic() - t0

    # Results table
    table = Table(title=f"Probe Results: {args.env} @ d={args.difficulty}", show_header=True)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", justify="right", width=15)

    table.add_row("Episodes", str(stats["n_episodes"]))
    table.add_row("Mean Reward", f"{stats['mean_reward']:.4f}")
    table.add_row("Std Reward", f"{stats['std_reward']:.4f}")
    table.add_row("Min Reward", f"{stats['min_reward']:.4f}")
    table.add_row("Max Reward", f"{stats['max_reward']:.4f}")
    table.add_row("Mean Turns", f"{stats['mean_turns']:.2f}")
    table.add_row("Success Rate", f"{stats['success_rate']:.2%}")
    table.add_row("Time (s)", f"{elapsed:.2f}")

    console.print(table)

    # Histogram
    hist: list[int] = stats["reward_histogram"]
    bin_edges = [f"{-1.0 + i * 0.2:.1f}" for i in range(11)]
    console.print("\n[bold]Reward Histogram:[/bold]")
    for i, count in enumerate(hist):
        bar = "#" * count
        console.print(f"  [{bin_edges[i]:>5} .. {bin_edges[i + 1]:>5}] {count:>3} {bar}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Run solvability validation."""
    from ecom_rlve.data.catalog_loader import generate_synthetic_catalog
    from ecom_rlve.debug.validators import validate_all_envs, validate_env_solvability
    from ecom_rlve.envs.base import get_env

    console.print("[bold cyan]Generating synthetic catalog...[/bold cyan]")
    products, _ = generate_synthetic_catalog(n_products=500, seed=args.seed)

    if args.env == "all":
        console.print(
            f"[bold cyan]Validating all envs[/bold cyan] "
            f"max_difficulty={args.max_difficulty} n_trials={args.trials}"
        )
        results = validate_all_envs(
            catalog=products,
            max_difficulty=args.max_difficulty,
            n_trials=args.trials,
        )

        for env_id, env_results in sorted(results.items()):
            table = Table(
                title=f"Solvability: {env_id}",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Difficulty", justify="right", width=10)
            table.add_column("Solvable Rate", justify="right", width=15)
            table.add_column("Failures", justify="right", width=10)

            for r in env_results:
                rate = r["solvable_rate"]
                color = "green" if rate >= 0.9 else ("yellow" if rate >= 0.5 else "red")
                table.add_row(
                    str(r["difficulty"]),
                    f"[{color}]{rate:.2%}[/{color}]",
                    str(r["n_failures"]),
                )
            console.print(table)
            console.print()
    else:
        env_instance = get_env(args.env)
        console.print(
            f"[bold cyan]Validating {args.env}[/bold cyan] "
            f"max_difficulty={args.max_difficulty}"
        )

        table = Table(
            title=f"Solvability: {args.env}",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Difficulty", justify="right", width=10)
        table.add_column("Solvable Rate", justify="right", width=15)
        table.add_column("Failures", justify="right", width=10)

        for d in range(args.max_difficulty + 1):
            result = validate_env_solvability(
                env=env_instance,
                catalog=products,
                difficulty=d,
                n_trials=args.trials,
            )
            rate = result["solvable_rate"]
            color = "green" if rate >= 0.9 else ("yellow" if rate >= 0.5 else "red")
            table.add_row(
                str(d),
                f"[{color}]{rate:.2%}[/{color}]",
                str(len(result["failures"])),
            )

        console.print(table)


def cmd_episode(args: argparse.Namespace) -> None:
    """Run one episode with DummyModelFn and show full trace."""
    from ecom_rlve.debug.inspector import EpisodeInspector
    from ecom_rlve.server.openenv import EcomRLVEEnv
    from ecom_rlve.training.rollout import DummyModelFn, run_rollout

    console.print(
        f"[bold cyan]Running episode[/bold cyan] "
        f"env=[bold]{args.env}[/bold] "
        f"difficulty=[bold]{args.difficulty}[/bold] "
        f"seed=[bold]{args.seed}[/bold]"
    )

    env = EcomRLVEEnv(collection="C8", seed=args.seed)
    env.dump_dir = ""
    if args.verbose:
        env.trace_episodes = True

    product_ids = [p.id for p in env._products[:20]]
    dummy = DummyModelFn(env_id=args.env, product_ids=product_ids, seed=args.seed)

    result = run_rollout(
        env=env,
        model_fn=dummy,
        env_id=args.env,
        difficulty=args.difficulty,
        seed=args.seed,
        collect_trace=True,
    )

    inspector = EpisodeInspector(env=env)

    if result.episode_trace:
        output = inspector.inspect_episode(result.episode_trace)
        console.print(output)
    else:
        console.print(f"[yellow]No trace collected.[/yellow]")
        console.print(f"Reward: {result.reward:.4f}, Correct: {result.is_correct}")

    env.close()


def cmd_difficulty(args: argparse.Namespace) -> None:
    """Show difficulty parameter table for a range of d values."""
    from ecom_rlve.difficulty.mapping import map_difficulty

    parts = args.range.split("-")
    d_min = int(parts[0])
    d_max = int(parts[1]) if len(parts) > 1 else d_min

    table = Table(
        title=f"Difficulty Parameters (d={d_min}..{d_max})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("d", justify="right", width=4)
    table.add_column("m", justify="right", width=5)
    table.add_column("k_rec", justify="right", width=6)
    table.add_column("T_max", justify="right", width=6)
    table.add_column("p_miss", justify="right", width=7)
    table.add_column("p_noise", justify="right", width=7)
    table.add_column("p_switch", justify="right", width=8)
    table.add_column("top_k", justify="right", width=6)
    table.add_column("eps_rnk", justify="right", width=8)
    table.add_column("p_oos", justify="right", width=7)
    table.add_column("H_ord", justify="right", width=6)
    table.add_column("B_br", justify="right", width=5)
    table.add_column("B_tool", justify="right", width=6)

    for d in range(d_min, d_max + 1):
        p = map_difficulty(d)
        table.add_row(
            str(d),
            str(p.m_val),
            str(p.k_rec_val),
            str(p.T_max_val),
            f"{p.p_missing_val:.3f}",
            f"{p.p_noise_val:.3f}",
            f"{p.p_switch_val:.3f}",
            str(p.top_k_val),
            f"{p.eps_rank_val:.3f}",
            f"{p.p_oos_val:.3f}",
            str(p.H_orders_val),
            str(p.B_branch_val),
            str(p.B_tool_val),
        )

    console.print(table)


def cmd_smoke_test(args: argparse.Namespace) -> None:
    """Quick test that all 8 envs can reset+step without crashing."""
    from ecom_rlve.server.openenv import EcomRLVEEnv
    from ecom_rlve.training.collections import COLLECTIONS

    console.print("[bold cyan]Smoke Test: all 8 environments[/bold cyan]")

    env = EcomRLVEEnv(collection="C8", seed=42)
    env.dump_dir = ""

    all_env_ids = COLLECTIONS["C8"]
    results_table = Table(
        title="Smoke Test Results", show_header=True, header_style="bold green"
    )
    results_table.add_column("Env ID", style="cyan", width=10)
    results_table.add_column("Reset", justify="center", width=8)
    results_table.add_column("Step", justify="center", width=8)
    results_table.add_column("Reward", justify="right", width=10)
    results_table.add_column("Error", style="red", width=40)

    n_pass = 0
    for env_id in all_env_ids:
        reset_ok = False
        step_ok = False
        reward_str = "N/A"
        error_str = ""

        try:
            obs = env.reset(env_id=env_id, difficulty=0, seed=42)
            reset_ok = True

            action = json.dumps({
                "assistant_message": "Here are my recommendations.",
                "tool_calls": [],
                "answer": {"env": env_id, "done": True, "recommended_product_ids": []},
            })
            obs, reward, done, info = env.step(action)
            step_ok = True
            reward_str = f"{reward:.4f}"
            n_pass += 1
        except Exception as exc:
            error_str = f"{type(exc).__name__}: {str(exc)[:60]}"

        reset_icon = "[green]PASS[/green]" if reset_ok else "[red]FAIL[/red]"
        step_icon = "[green]PASS[/green]" if step_ok else "[red]FAIL[/red]"
        results_table.add_row(env_id, reset_icon, step_icon, reward_str, error_str)

    console.print(results_table)
    console.print(
        f"\n[bold]Result: {n_pass}/{len(all_env_ids)} environments passed.[/bold]"
    )

    env.close()

    if n_pass < len(all_env_ids):
        sys.exit(1)


def main() -> None:
    """Entry point for the debug CLI."""
    parser = argparse.ArgumentParser(
        description="EcomRLVE Debug CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Debug subcommand")

    # probe
    p_probe = subparsers.add_parser("probe", help="Run probe_env() and print results")
    p_probe.add_argument("--env", type=str, required=True, help="Env ID (e.g. PD, SUB)")
    p_probe.add_argument("--difficulty", type=int, default=0, help="Difficulty level")
    p_probe.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    p_probe.add_argument("--seed", type=int, default=42, help="Random seed")

    # validate
    p_validate = subparsers.add_parser("validate", help="Run solvability validation")
    p_validate.add_argument(
        "--env", type=str, default="all", help="Env ID or 'all'"
    )
    p_validate.add_argument("--max-difficulty", type=int, default=10, help="Max difficulty")
    p_validate.add_argument("--trials", type=int, default=20, help="Trials per level")
    p_validate.add_argument("--seed", type=int, default=42, help="Random seed")

    # episode
    p_episode = subparsers.add_parser("episode", help="Run one episode and show trace")
    p_episode.add_argument("--env", type=str, default="PD", help="Env ID")
    p_episode.add_argument("--difficulty", type=int, default=0, help="Difficulty level")
    p_episode.add_argument("--seed", type=int, default=42, help="Random seed")
    p_episode.add_argument("--verbose", action="store_true", help="Enable trace logging")

    # difficulty
    p_diff = subparsers.add_parser("difficulty", help="Show difficulty parameter table")
    p_diff.add_argument("--range", type=str, default="0-15", help="Range e.g. '0-15'")

    # smoke-test
    subparsers.add_parser("smoke-test", help="Quick test all 8 envs can reset+step")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch: dict[str, Any] = {
        "probe": cmd_probe,
        "validate": cmd_validate,
        "episode": cmd_episode,
        "difficulty": cmd_difficulty,
        "smoke-test": cmd_smoke_test,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
