"""Difficulty engine for EcomRLVE-GYM.

Provides the mapping from difficulty integer to parameter vector (12 axes)
and the RLVE-style adaptive difficulty state machine.

Components:
    - mapping:  d -> theta(d) with individual axis functions
    - adaptive: Per-env state tracking with promotion logic
"""

from ecom_rlve.difficulty.adaptive import AdaptiveDifficultyEngine, AdaptiveState
from ecom_rlve.difficulty.mapping import (
    B_branch,
    B_tool,
    DifficultyParams,
    H_orders,
    T_max,
    describe_difficulty,
    eps_rank,
    k_rec,
    m,
    map_difficulty,
    p_missing,
    p_noise,
    p_oos,
    p_switch,
    sigmoid,
    top_k,
)

__all__ = [
    # mapping
    "sigmoid",
    "m",
    "k_rec",
    "T_max",
    "p_missing",
    "p_noise",
    "p_switch",
    "top_k",
    "eps_rank",
    "p_oos",
    "H_orders",
    "B_branch",
    "B_tool",
    "DifficultyParams",
    "map_difficulty",
    "describe_difficulty",
    # adaptive
    "AdaptiveState",
    "AdaptiveDifficultyEngine",
]
