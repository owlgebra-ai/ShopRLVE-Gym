"""ShopRLVE-GYM debug and evaluation tools (Phase H).

Provides:
    - EpisodeInspector:  Pretty-print and compare episode traces.
    - replay_episode:    Re-run an episode from its trace for determinism checks.
    - probe_env:         Quick diagnostic: run N episodes and return stats.
    - validate_reward_bounds:    Assert reward in [-1, 1].
    - validate_episode_state:    Consistency checks on EpisodeState.
    - validate_env_solvability:  Check that an env can produce solvable problems.
    - validate_all_envs:         Solvability sweep across all 8 envs.
"""

from shop_rlve.debug.inspector import EpisodeInspector
from shop_rlve.debug.replay import probe_env, replay_episode
from shop_rlve.debug.validators import (
    validate_all_envs,
    validate_env_solvability,
    validate_episode_state,
    validate_reward_bounds,
)

__all__ = [
    "EpisodeInspector",
    "replay_episode",
    "probe_env",
    "validate_reward_bounds",
    "validate_episode_state",
    "validate_env_solvability",
    "validate_all_envs",
]
