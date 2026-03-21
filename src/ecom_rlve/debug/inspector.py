"""Episode inspector for debugging EcomRLVE-GYM episodes.

Provides pretty-printing, comparison, and serialization of episode traces
using the ``rich`` library for colored terminal output.

Usage:
    from ecom_rlve.debug.inspector import EpisodeInspector
    from ecom_rlve.server import EcomRLVEEnv

    inspector = EpisodeInspector()
    trace = env.get_episode_trace()
    print(inspector.inspect_episode(trace))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ecom_rlve.difficulty.adaptive import AdaptiveDifficultyEngine

logger = logging.getLogger(__name__)


class EpisodeInspector:
    """Debug inspector for episode traces produced by EcomRLVEEnv.

    Renders episode data as richly-formatted terminal output including
    conversation flow, tool calls, reward breakdowns, and difficulty state.

    Args:
        env: Optional EcomRLVEEnv instance (used for live inspection).
    """

    def __init__(self, env: Any | None = None) -> None:
        """Initialise the inspector.

        Args:
            env: Optional EcomRLVEEnv instance. When provided, the inspector
                 can pull live episode state directly from the env.
        """
        self._env = env
        self._console = Console(record=True)

    # ------------------------------------------------------------------
    # inspect_episode
    # ------------------------------------------------------------------

    def inspect_episode(self, trace: dict[str, Any]) -> str:
        """Pretty-print a full episode trace.

        Shows:
            - Env ID, difficulty, seed
            - Problem params (target products, constraints)
            - Turn-by-turn: user message, assistant response, tool calls + results
            - Reward breakdown (r_task, r_eff, r_hall, r_total)
            - IsCorrect flag

        Args:
            trace: Episode trace dict (from ``EcomRLVEEnv.get_episode_trace()``).

        Returns:
            Rich-formatted string suitable for terminal display.
        """
        self._console = Console(record=True, width=120)

        # Header
        env_id: str = trace.get("env_id", "?")
        difficulty: int = trace.get("difficulty", 0)
        seed: int = trace.get("seed", 0)
        done: bool = trace.get("done", False)
        reward: float | None = trace.get("reward")

        header_text = (
            f"[bold cyan]Episode Trace[/bold cyan]  "
            f"env=[bold]{env_id}[/bold]  "
            f"difficulty=[bold]{difficulty}[/bold]  "
            f"seed=[bold]{seed}[/bold]  "
            f"done=[bold]{done}[/bold]  "
            f"reward=[bold {'green' if reward is not None and reward > 0 else 'red'}]"
            f"{reward if reward is not None else 'N/A'}[/]"
        )
        self._console.print(Panel(header_text, title="Episode Inspector", border_style="blue"))

        # Hidden goal / problem params
        hidden_goal: dict[str, Any] = trace.get("hidden_goal", {})
        if hidden_goal:
            goal_table = Table(title="Problem Params", show_header=True, header_style="bold magenta")
            goal_table.add_column("Key", style="cyan", width=25)
            goal_table.add_column("Value", style="white")
            for key, value in hidden_goal.items():
                goal_table.add_row(key, str(value))
            self._console.print(goal_table)

        # Conversation turn-by-turn
        conversation: list[dict[str, str]] = trace.get("conversation", [])
        tool_history: list[dict[str, Any]] = trace.get("tool_results_history", [])

        self._console.print("\n[bold underline]Conversation[/bold underline]")
        tool_idx = 0
        for msg in conversation:
            role: str = msg.get("role", "?")
            content: str = msg.get("content", "")

            if role == "user":
                self._console.print(
                    Panel(content, title=f"[bold blue]User[/bold blue]", border_style="blue")
                )
            elif role == "assistant":
                self._console.print(
                    Panel(content, title=f"[bold green]Assistant[/bold green]", border_style="green")
                )
                # Show tool calls that occurred during this assistant turn
                # (heuristic: tool results in order)
                while tool_idx < len(tool_history):
                    tr = tool_history[tool_idx]
                    tool_name: str = tr.get("name", "?")
                    tool_args: dict[str, Any] = tr.get("args", {})
                    tool_result: Any = tr.get("result")
                    tool_error: str | None = tr.get("error")

                    result_str = (
                        f"[red]ERROR: {tool_error}[/red]"
                        if tool_error
                        else str(tool_result)[:200]
                    )
                    self._console.print(
                        f"  [dim]Tool:[/dim] [yellow]{tool_name}[/yellow]"
                        f"({json.dumps(tool_args, default=str)[:100]})"
                        f" -> {result_str}"
                    )
                    tool_idx += 1
                    # Stop if we reach a boundary (next message will be a user turn)
                    break

        # Reward breakdown
        breakdown: dict[str, Any] = trace.get("reward_breakdown", {})
        if breakdown:
            self._console.print()
            reward_table = Table(
                title="Reward Breakdown", show_header=True, header_style="bold yellow"
            )
            reward_table.add_column("Component", style="cyan", width=20)
            reward_table.add_column("Value", style="white", justify="right")
            reward_table.add_row("r_task", f"{breakdown.get('r_task', 0.0):.4f}")
            reward_table.add_row("r_eff", f"{breakdown.get('r_eff', 0.0):.4f}")
            reward_table.add_row("r_hall", f"{breakdown.get('r_hall', 0.0):.4f}")
            reward_table.add_row(
                "r_total",
                f"[bold]{breakdown.get('r_total', 0.0):.4f}[/bold]",
            )
            reward_table.add_row("format_valid", str(breakdown.get("format_valid", True)))
            reward_table.add_row("tool_valid", str(breakdown.get("tool_valid", True)))
            reward_table.add_row("safety_valid", str(breakdown.get("safety_valid", True)))
            reward_table.add_row(
                "is_correct",
                f"[bold {'green' if breakdown.get('is_correct') else 'red'}]"
                f"{breakdown.get('is_correct', False)}[/]",
            )
            self._console.print(reward_table)

        # Timing
        timing: dict[str, float] = trace.get("timing", {})
        if timing:
            # Filter out internal keys (prefixed with '_')
            display_timing = {k: v for k, v in timing.items() if not k.startswith("_")}
            if display_timing:
                timing_text = "  ".join(f"{k}={v:.1f}" for k, v in display_timing.items())
                self._console.print(f"\n[dim]Timing: {timing_text}[/dim]")

        return self._console.export_text()

    # ------------------------------------------------------------------
    # compare_episodes
    # ------------------------------------------------------------------

    def compare_episodes(self, traces: list[dict[str, Any]]) -> str:
        """Compare multiple episode traces side-by-side.

        Displays a table with key metrics for each trace: env_id, difficulty,
        turns, reward components, and correctness.

        Args:
            traces: List of episode trace dicts.

        Returns:
            Rich-formatted comparison string.
        """
        self._console = Console(record=True, width=140)

        table = Table(title="Episode Comparison", show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Env", style="cyan", width=8)
        table.add_column("Diff", justify="right", width=5)
        table.add_column("Seed", justify="right", width=10)
        table.add_column("Turns", justify="right", width=6)
        table.add_column("r_task", justify="right", width=8)
        table.add_column("r_eff", justify="right", width=8)
        table.add_column("r_hall", justify="right", width=8)
        table.add_column("r_total", justify="right", width=8)
        table.add_column("Correct", justify="center", width=8)

        for i, trace in enumerate(traces):
            bd: dict[str, Any] = trace.get("reward_breakdown", {})
            is_correct: bool = bd.get("is_correct", False)
            correct_str = "[green]YES[/green]" if is_correct else "[red]NO[/red]"

            table.add_row(
                str(i),
                trace.get("env_id", "?"),
                str(trace.get("difficulty", 0)),
                str(trace.get("seed", 0)),
                str(trace.get("turn", 0)),
                f"{bd.get('r_task', 0.0):.4f}",
                f"{bd.get('r_eff', 0.0):.4f}",
                f"{bd.get('r_hall', 0.0):.4f}",
                f"{bd.get('r_total', 0.0):.4f}",
                correct_str,
            )

        self._console.print(table)
        return self._console.export_text()

    # ------------------------------------------------------------------
    # inspect_reward
    # ------------------------------------------------------------------

    def inspect_reward(self, trace: dict[str, Any]) -> str:
        """Detailed reward computation breakdown for an episode trace.

        Shows every intermediate value that went into the final reward,
        including weights, pre-clip totals, and per-component contributions.

        Args:
            trace: Episode trace dict.

        Returns:
            Rich-formatted reward breakdown string.
        """
        self._console = Console(record=True, width=120)
        breakdown: dict[str, Any] = trace.get("reward_breakdown", {})
        details: dict[str, Any] = breakdown.get("details", {})

        header = (
            f"[bold yellow]Reward Inspection[/bold yellow]  "
            f"env=[bold]{trace.get('env_id', '?')}[/bold]  "
            f"difficulty=[bold]{trace.get('difficulty', 0)}[/bold]"
        )
        self._console.print(Panel(header, border_style="yellow"))

        # Hard-fail check
        hard_fail: bool = details.get("hard_fail", False)
        if hard_fail:
            self._console.print(
                "[bold red]HARD FAIL[/bold red] -- reward forced to -1.0"
            )
            self._console.print(
                f"  format_valid={details.get('format_valid')}  "
                f"tool_valid={details.get('tool_valid')}  "
                f"safety_valid={details.get('safety_valid')}"
            )
            return self._console.export_text()

        # Component table
        comp_table = Table(title="Reward Components", show_header=True, header_style="bold")
        comp_table.add_column("Component", width=20)
        comp_table.add_column("Raw Value", justify="right", width=12)
        comp_table.add_column("Weight", justify="right", width=10)
        comp_table.add_column("Contribution", justify="right", width=12)

        r_task: float = details.get("r_task", breakdown.get("r_task", 0.0))
        r_eff: float = details.get("r_eff", breakdown.get("r_eff", 0.0))
        r_hall: float = details.get("r_hall", breakdown.get("r_hall", 0.0))
        w_task: float = details.get("w_task", 0.75)
        w_eff: float = details.get("w_eff", 0.15)
        w_hall: float = details.get("w_hall", 0.10)

        comp_table.add_row("r_task", f"{r_task:.4f}", f"{w_task:.2f}", f"{w_task * r_task:.4f}")
        comp_table.add_row("r_eff", f"{r_eff:.4f}", f"{w_eff:.2f}", f"{w_eff * r_eff:.4f}")
        comp_table.add_row("r_hall", f"{r_hall:.4f}", f"{w_hall:.2f}", f"{w_hall * r_hall:.4f}")
        comp_table.add_row(
            "[bold]r_total[/bold]",
            "",
            "",
            f"[bold]{breakdown.get('r_total', 0.0):.4f}[/bold]",
        )
        self._console.print(comp_table)

        # Extra details
        if "turns" in details:
            self._console.print(
                f"\n  Turns: {details['turns']} / T_max={details.get('t_max', '?')}"
            )
        if "hallucination_rate" in details:
            self._console.print(
                f"  Hallucination rate: {details['hallucination_rate']:.4f}"
            )
        if "seen_ids_count" in details:
            self._console.print(
                f"  Seen IDs count: {details['seen_ids_count']}"
            )
        if "output_ids" in details:
            self._console.print(
                f"  Output IDs: {details['output_ids']}"
            )

        return self._console.export_text()

    # ------------------------------------------------------------------
    # inspect_difficulty
    # ------------------------------------------------------------------

    def inspect_difficulty(self, engine: AdaptiveDifficultyEngine) -> str:
        """Show adaptive difficulty state for all tracked environments.

        Renders a table with per-env difficulty ranges, frontier trial counts,
        and accuracy at the frontier.

        Args:
            engine: AdaptiveDifficultyEngine instance.

        Returns:
            Rich-formatted difficulty state string.
        """
        self._console = Console(record=True, width=100)

        table = Table(
            title="Adaptive Difficulty State", show_header=True, header_style="bold magenta"
        )
        table.add_column("Env ID", style="cyan", width=10)
        table.add_column("Low", justify="right", width=6)
        table.add_column("High", justify="right", width=6)
        table.add_column("Trials", justify="right", width=8)
        table.add_column("Correct", justify="right", width=8)
        table.add_column("Accuracy", justify="right", width=10)

        for env_id in sorted(engine.env_ids):
            state = engine.get_state(env_id)
            acc_str = f"{state.accuracy():.3f}" if state.total_count > 0 else "N/A"
            table.add_row(
                env_id,
                str(state.low),
                str(state.high),
                str(state.total_count),
                str(state.correct_count),
                acc_str,
            )

        footer = (
            f"tau_acc={engine.tau_acc}  tau_num={engine.tau_num}  d_delta={engine.d_delta}"
        )
        self._console.print(table)
        self._console.print(f"[dim]{footer}[/dim]")
        return self._console.export_text()

    # ------------------------------------------------------------------
    # dump / load
    # ------------------------------------------------------------------

    def dump_episode(self, trace: dict[str, Any], path: str) -> None:
        """Save an episode trace as a JSON file.

        Args:
            trace: Episode trace dict.
            path:  File path to write to.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(trace, f, indent=2, default=str)
        logger.info("Episode trace saved to %s", p)

    def load_episode(self, path: str) -> dict[str, Any]:
        """Load an episode trace from a JSON file.

        Args:
            path: File path to read from.

        Returns:
            Episode trace dict.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        p = Path(path)
        with open(p) as f:
            trace: dict[str, Any] = json.load(f)
        logger.info("Episode trace loaded from %s", p)
        return trace
