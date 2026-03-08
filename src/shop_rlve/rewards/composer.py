"""Reward composition for ShopRLVE-GYM (Spec Section 5).

Combines task-specific reward (r_task), efficiency reward (r_eff), and
hallucination penalty (r_hall) into a single scalar reward in [-1, 1].

Composition rule:
    if format_invalid OR tool_invalid OR safety_violation:
        reward = -1.0
    else:
        reward = clip(w_task * r_task + w_eff * r_eff + w_hall * r_hall, -1, 1)

Default weights: w_task=0.75, w_eff=0.15, w_hall=0.10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shop_rlve.rewards.metrics import efficiency_reward, hallucination_reward


# ---------------------------------------------------------------------------
# Reward breakdown
# ---------------------------------------------------------------------------


@dataclass
class RewardBreakdown:
    """Complete reward breakdown for an episode.

    Provides both the final composite reward and all intermediate values
    for debugging and analysis.

    Attributes:
        r_task:       Task-specific reward in [-1, 1].
        r_eff:        Efficiency reward in [-1, 1].
        r_hall:       Hallucination penalty in [-1, 0].
        r_total:      Final composite reward in [-1, 1].
        format_valid: Whether the agent's output format was valid.
        tool_valid:   Whether all tool calls were valid.
        safety_valid: Whether no safety violations occurred.
        is_correct:   Whether the agent's answer meets the IsCorrect threshold.
        details:      Debug dictionary with all intermediate computation values.
    """

    r_task: float
    r_eff: float
    r_hall: float
    r_total: float
    format_valid: bool
    tool_valid: bool
    safety_valid: bool
    is_correct: bool
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward composition
# ---------------------------------------------------------------------------


def compose_reward(
    r_task: float,
    turns: int,
    t_max: int,
    output_ids: list[str],
    seen_ids: set[str],
    *,
    format_valid: bool = True,
    tool_valid: bool = True,
    safety_valid: bool = True,
    w_task: float = 0.75,
    w_eff: float = 0.15,
    w_hall: float = 0.10,
    is_correct: bool = False,
    debug: bool = False,
) -> RewardBreakdown:
    """Compose the final reward from task reward, efficiency, and hallucination penalty.

    Spec Section 5:
        if format_invalid OR tool_invalid OR safety_violation:
            reward = -1.0
        else:
            reward = clip(w_task * r_task + w_eff * r_eff + w_hall * r_hall, -1, 1)

    Default weights: w_task=0.75, w_eff=0.15, w_hall=0.10

    Args:
        r_task:       Task-specific reward in [-1, 1].
        turns:        Number of turns used in the episode (T >= 1).
        t_max:        Maximum allowed turns (T_max(d)).
        output_ids:   Product IDs in the agent's output.
        seen_ids:     Set of product IDs surfaced to the agent via tools.
        format_valid: Whether the agent's output JSON format was valid.
        tool_valid:   Whether all tool calls used valid names and arguments.
        safety_valid: Whether no safety violations occurred (e.g., denied categories).
        w_task:       Weight for task reward (default: 0.75).
        w_eff:        Weight for efficiency reward (default: 0.15).
        w_hall:       Weight for hallucination penalty (default: 0.10).
        is_correct:   Whether the agent's answer is correct (passed to breakdown).
        debug:        When True, populate the details dict with all intermediate values.

    Returns:
        RewardBreakdown with the composite reward and all components.
    """
    details: dict[str, Any] = {}

    # Hard fail checks
    hard_fail = not format_valid or not tool_valid or not safety_valid

    if hard_fail:
        r_eff_val = 0.0
        r_hall_val = 0.0
        r_total = -1.0

        if debug:
            details["hard_fail"] = True
            details["format_valid"] = format_valid
            details["tool_valid"] = tool_valid
            details["safety_valid"] = safety_valid
            details["r_task"] = r_task
            details["r_eff"] = r_eff_val
            details["r_hall"] = r_hall_val
            details["r_total_pre_clip"] = -1.0
            details["r_total"] = -1.0

        return RewardBreakdown(
            r_task=r_task,
            r_eff=r_eff_val,
            r_hall=r_hall_val,
            r_total=r_total,
            format_valid=format_valid,
            tool_valid=tool_valid,
            safety_valid=safety_valid,
            is_correct=False,  # Hard fail means never correct
            details=details,
        )

    # Compute component rewards
    r_eff_val = efficiency_reward(turns, t_max)
    r_hall_val = hallucination_reward(output_ids, seen_ids)

    # Weighted combination
    r_total_raw = w_task * r_task + w_eff * r_eff_val + w_hall * r_hall_val
    r_total = float(np.clip(r_total_raw, -1.0, 1.0))

    if debug:
        details["hard_fail"] = False
        details["format_valid"] = format_valid
        details["tool_valid"] = tool_valid
        details["safety_valid"] = safety_valid
        details["r_task"] = r_task
        details["r_eff"] = r_eff_val
        details["r_hall"] = r_hall_val
        details["w_task"] = w_task
        details["w_eff"] = w_eff
        details["w_hall"] = w_hall
        details["r_total_pre_clip"] = r_total_raw
        details["r_total"] = r_total
        details["turns"] = turns
        details["t_max"] = t_max
        details["output_ids"] = output_ids
        details["seen_ids_count"] = len(seen_ids)
        details["hallucination_rate"] = (
            sum(1 for pid in output_ids if pid not in seen_ids) / max(len(output_ids), 1)
            if output_ids
            else 0.0
        )

    return RewardBreakdown(
        r_task=r_task,
        r_eff=r_eff_val,
        r_hall=r_hall_val,
        r_total=r_total,
        format_valid=format_valid,
        tool_valid=tool_valid,
        safety_valid=safety_valid,
        is_correct=is_correct,
        details=details,
    )
