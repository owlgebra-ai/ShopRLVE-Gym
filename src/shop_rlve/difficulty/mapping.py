"""Difficulty integer to parameter vector mapping (Spec Section 4).

Maps a single integer difficulty level d to a 12-dimensional parameter
vector theta(d). Each axis has a standard formula that controls a
specific aspect of episode complexity.

12 difficulty axes:
    m(d)        = 2 + floor(d/2)           -- number of constraints
    k_rec(d)    = min(10, 3 + floor(d/3))  -- output items (k for ranking)
    T_max(d)    = 4 + floor(d/2)           -- max turns
    p_missing(d)= 0.8 * sigmoid((d-3)/1.5) -- slot omission probability
    p_noise(d)  = clip(0.02*d, 0, 0.25)    -- typo/noise rate
    p_switch(d) = 0.6 * sigmoid((d-5)/2)   -- context switch probability
    top_k(d)    = max(20, 200 - 10*d)      -- retrieval result count
    eps_rank(d) = min(0.4, 0.02*d)         -- retrieval degradation
    p_oos(d)    = min(0.5, 0.05*d)         -- out-of-stock probability
    H_orders(d) = 1 + floor(d/2)           -- order history depth
    B_branch(d) = 1 + floor(d/3)           -- policy branching complexity
    B_tool(d)   = 1 + floor(d/2)           -- tool call budget per step
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Sigmoid helper
# ---------------------------------------------------------------------------


def sigmoid(x: float) -> float:
    """Standard logistic sigmoid function.

    sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        x: Input value.

    Returns:
        Sigmoid output in (0, 1).
    """
    # Use numpy for numerical stability (handles large negative/positive x)
    return float(1.0 / (1.0 + np.exp(-x)))


# ---------------------------------------------------------------------------
# Individual axis functions
# ---------------------------------------------------------------------------


def m(d: int) -> int:
    """Number of constraints.

    Spec Section 4: m(d) = 2 + floor(d/2)

    Used by: PD, SUB, CART, BUNDLE.

    Args:
        d: Difficulty level (integer >= 0).

    Returns:
        Number of constraints (integer >= 2).
    """
    return 2 + math.floor(d / 2)


def k_rec(d: int) -> int:
    """Number of output items to recommend.

    Spec Section 4: k_rec(d) = min(10, 3 + floor(d/3))

    Used by: PD, SUB.

    Args:
        d: Difficulty level.

    Returns:
        Number of items to output (integer in [3, 10]).
    """
    return min(10, 3 + math.floor(d / 3))


def T_max(d: int) -> int:
    """Maximum allowed turns.

    Spec Section 4: T_max(d) = 4 + floor(d/2)

    Used by: All environments.

    Args:
        d: Difficulty level.

    Returns:
        Maximum turns (integer >= 4).
    """
    return 4 + math.floor(d / 2)


def p_missing(d: int) -> float:
    """Slot omission probability.

    Spec Section 4: p_missing(d) = 0.8 * sigmoid((d-3)/1.5)

    Controls how often non-critical slots are omitted from the user's
    initial message, forcing clarification.

    Used by: All environments.

    Args:
        d: Difficulty level.

    Returns:
        Probability in [0, 0.8).
    """
    return 0.8 * sigmoid((d - 3) / 1.5)


def p_noise(d: int) -> float:
    """Per-character typo/noise rate.

    Spec Section 4: p_noise(d) = clip(0.02*d, 0, 0.25)

    Used by: All environments.

    Args:
        d: Difficulty level.

    Returns:
        Noise probability in [0, 0.25].
    """
    return float(np.clip(0.02 * d, 0.0, 0.25))


def p_switch(d: int) -> float:
    """Context switch probability.

    Spec Section 4: p_switch(d) = 0.6 * sigmoid((d-5)/2)

    Probability of the user changing intent mid-conversation (JOURNEY env).

    Used by: JOURNEY.

    Args:
        d: Difficulty level.

    Returns:
        Probability in [0, 0.6).
    """
    return 0.6 * sigmoid((d - 5) / 2.0)


def top_k(d: int) -> int:
    """Number of retrieval results returned by catalog.search.

    Spec Section 4: top_k(d) = max(20, 200 - 10*d)

    Higher difficulty means fewer results, making discovery harder.

    Used by: PD, SUB.

    Args:
        d: Difficulty level.

    Returns:
        Top-k value (integer >= 20).
    """
    return max(20, 200 - 10 * d)


def eps_rank(d: int) -> float:
    """Retrieval degradation probability.

    Spec Section 4: eps_rank(d) = min(0.4, 0.02*d)

    With this probability, each retrieved result is replaced with a
    random distractor from the same category.

    Used by: PD, SUB.

    Args:
        d: Difficulty level.

    Returns:
        Degradation probability in [0, 0.4].
    """
    return min(0.4, 0.02 * d)


def p_oos(d: int) -> float:
    """Out-of-stock probability.

    Spec Section 4: p_oos(d) = min(0.5, 0.05*d)

    Probability that a given product in the evaluation pool is OOS.

    Used by: PD, SUB, CART.

    Args:
        d: Difficulty level.

    Returns:
        OOS probability in [0, 0.5].
    """
    return min(0.5, 0.05 * d)


def H_orders(d: int) -> int:
    """Order history depth.

    Spec Section 4: H_orders(d) = 1 + floor(d/2)

    Number of orders in the user's history. More orders make it harder
    to identify the correct one.

    Used by: RETURN, STATUS.

    Args:
        d: Difficulty level.

    Returns:
        Number of orders (integer >= 1).
    """
    return 1 + math.floor(d / 2)


def B_branch(d: int) -> int:
    """Policy branching complexity.

    Spec Section 4: B_branch(d) = 1 + floor(d/3)

    Number of condition clauses in the policy question. More branches
    require the agent to evaluate more complex rules.

    Used by: POLICY.

    Args:
        d: Difficulty level.

    Returns:
        Number of branches (integer >= 1).
    """
    return 1 + math.floor(d / 3)


def B_tool(d: int) -> int:
    """Tool call budget per step.

    Spec Section 4: B_tool(d) = 1 + floor(d/2)

    Maximum number of tool calls the agent can make per turn.
    Higher difficulty gives more budget for complex tool sequences.

    Used by: All environments.

    Args:
        d: Difficulty level.

    Returns:
        Tool call budget (integer >= 1).
    """
    return 1 + math.floor(d / 2)


# ---------------------------------------------------------------------------
# Difficulty parameter vector
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DifficultyParams:
    """Complete difficulty parameter vector theta(d).

    Spec Section 4: d -> theta(d) with 12 axes.

    All 12 difficulty axes computed from a single integer d, with
    optional per-axis overrides.

    Attributes:
        d:            Raw difficulty integer.
        m_val:        Number of constraints.
        k_rec_val:    Number of output items.
        T_max_val:    Maximum allowed turns.
        p_missing_val: Slot omission probability.
        p_noise_val:  Per-character noise rate.
        p_switch_val: Context switch probability.
        top_k_val:    Retrieval result count.
        eps_rank_val: Retrieval degradation probability.
        p_oos_val:    Out-of-stock probability.
        H_orders_val: Order history depth.
        B_branch_val: Policy branching complexity.
        B_tool_val:   Tool call budget per step.
    """

    d: int
    m_val: int
    k_rec_val: int
    T_max_val: int
    p_missing_val: float
    p_noise_val: float
    p_switch_val: float
    top_k_val: int
    eps_rank_val: float
    p_oos_val: float
    H_orders_val: int
    B_branch_val: int
    B_tool_val: int

    def as_dict(self) -> dict[str, Any]:
        """Return all parameters as a dictionary."""
        return {
            "d": self.d,
            "m": self.m_val,
            "k_rec": self.k_rec_val,
            "T_max": self.T_max_val,
            "p_missing": self.p_missing_val,
            "p_noise": self.p_noise_val,
            "p_switch": self.p_switch_val,
            "top_k": self.top_k_val,
            "eps_rank": self.eps_rank_val,
            "p_oos": self.p_oos_val,
            "H_orders": self.H_orders_val,
            "B_branch": self.B_branch_val,
            "B_tool": self.B_tool_val,
        }


def map_difficulty(d: int, overrides: dict[str, Any] | None = None) -> DifficultyParams:
    """Map a difficulty integer to the full parameter vector.

    Spec Section 4: d -> theta(d)

    Computes all 12 difficulty axes from d using their standard formulas,
    then applies any per-axis overrides.

    Args:
        d:         Difficulty level (integer >= 0).
        overrides: Optional dict of axis_name -> value to override defaults.
                   Valid keys: m, k_rec, T_max, p_missing, p_noise, p_switch,
                   top_k, eps_rank, p_oos, H_orders, B_branch, B_tool.

    Returns:
        DifficultyParams dataclass with all 12 axes.

    Raises:
        ValueError: If d is negative.
    """
    if d < 0:
        raise ValueError(f"Difficulty level must be >= 0, got {d}")

    # Compute standard values
    params: dict[str, Any] = {
        "d": d,
        "m_val": m(d),
        "k_rec_val": k_rec(d),
        "T_max_val": T_max(d),
        "p_missing_val": p_missing(d),
        "p_noise_val": p_noise(d),
        "p_switch_val": p_switch(d),
        "top_k_val": top_k(d),
        "eps_rank_val": eps_rank(d),
        "p_oos_val": p_oos(d),
        "H_orders_val": H_orders(d),
        "B_branch_val": B_branch(d),
        "B_tool_val": B_tool(d),
    }

    # Apply overrides
    if overrides:
        override_map: dict[str, str] = {
            "m": "m_val",
            "k_rec": "k_rec_val",
            "T_max": "T_max_val",
            "p_missing": "p_missing_val",
            "p_noise": "p_noise_val",
            "p_switch": "p_switch_val",
            "top_k": "top_k_val",
            "eps_rank": "eps_rank_val",
            "p_oos": "p_oos_val",
            "H_orders": "H_orders_val",
            "B_branch": "B_branch_val",
            "B_tool": "B_tool_val",
        }
        for key, value in overrides.items():
            param_key = override_map.get(key)
            if param_key is not None:
                params[param_key] = value

    return DifficultyParams(**params)


# ---------------------------------------------------------------------------
# Human-readable description
# ---------------------------------------------------------------------------


def describe_difficulty(d: int) -> str:
    """Generate a human-readable summary of difficulty parameters.

    Useful for debugging and logging.

    Args:
        d: Difficulty level.

    Returns:
        Multi-line string describing all 12 axes.
    """
    params = map_difficulty(d)
    lines = [
        f"Difficulty d={d} parameter vector theta(d):",
        f"  m (constraints)       = {params.m_val:>3}   [2 + floor({d}/2)]",
        f"  k_rec (output items)  = {params.k_rec_val:>3}   [min(10, 3 + floor({d}/3))]",
        f"  T_max (max turns)     = {params.T_max_val:>3}   [4 + floor({d}/2)]",
        f"  p_missing (omission)  = {params.p_missing_val:>6.3f} [0.8 * sigmoid(({d}-3)/1.5)]",
        f"  p_noise (typo rate)   = {params.p_noise_val:>6.3f} [clip(0.02*{d}, 0, 0.25)]",
        f"  p_switch (ctx switch) = {params.p_switch_val:>6.3f} [0.6 * sigmoid(({d}-5)/2)]",
        f"  top_k (retrieval)     = {params.top_k_val:>3}   [max(20, 200 - 10*{d})]",
        f"  eps_rank (degradation)= {params.eps_rank_val:>6.3f} [min(0.4, 0.02*{d})]",
        f"  p_oos (OOS rate)      = {params.p_oos_val:>6.3f} [min(0.5, 0.05*{d})]",
        f"  H_orders (history)    = {params.H_orders_val:>3}   [1 + floor({d}/2)]",
        f"  B_branch (branches)   = {params.B_branch_val:>3}   [1 + floor({d}/3)]",
        f"  B_tool (tool budget)  = {params.B_tool_val:>3}   [1 + floor({d}/2)]",
    ]
    return "\n".join(lines)
