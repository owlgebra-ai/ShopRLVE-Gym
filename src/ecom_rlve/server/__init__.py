"""EcomRLVE-GYM OpenEnv server: episode management and reset/step loop.

Provides:
    - EcomRLVEEnv:   Core environment server with reset() / step() interface.
    - EpisodeState:  Internal mutable state per active episode.
    - Observation:   What the model sees at each step.
    - ActionSchema:  Expected LLM output format.
    - AnswerSchema:  Structured answer portion of the action.
    - parse_action:  Parse raw JSON into ActionSchema.

Usage:
    from ecom_rlve.server import EcomRLVEEnv

    env = EcomRLVEEnv(collection="C1", seed=42)
    obs = env.reset()
    obs, reward, done, info = env.step(action_json)
"""

from ecom_rlve.server.openenv import EcomRLVEEnv
from ecom_rlve.server.state import (
    ActionSchema,
    AnswerSchema,
    EpisodeState,
    Observation,
    parse_action,
)

__all__ = [
    "EcomRLVEEnv",
    "EpisodeState",
    "Observation",
    "ActionSchema",
    "AnswerSchema",
    "parse_action",
]
