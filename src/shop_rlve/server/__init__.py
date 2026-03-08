"""ShopRLVE-GYM OpenEnv server: episode management and reset/step loop.

Provides:
    - ShopRLVEEnv:   Core environment server with reset() / step() interface.
    - EpisodeState:  Internal mutable state per active episode.
    - Observation:   What the model sees at each step.
    - ActionSchema:  Expected LLM output format.
    - AnswerSchema:  Structured answer portion of the action.
    - parse_action:  Parse raw JSON into ActionSchema.

Usage:
    from shop_rlve.server import ShopRLVEEnv

    env = ShopRLVEEnv(collection="C1", seed=42)
    obs = env.reset()
    obs, reward, done, info = env.step(action_json)
"""

from shop_rlve.server.openenv import ShopRLVEEnv
from shop_rlve.server.state import (
    ActionSchema,
    AnswerSchema,
    EpisodeState,
    Observation,
    parse_action,
)

__all__ = [
    "ShopRLVEEnv",
    "EpisodeState",
    "Observation",
    "ActionSchema",
    "AnswerSchema",
    "parse_action",
]
