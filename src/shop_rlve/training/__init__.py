"""ShopRLVE-GYM training harness: collections, rollouts, and GRPO integration.

Provides:
    - COLLECTIONS:              Canonical environment collection definitions.
    - get_collection:           Look up a collection by name.
    - validate_collection:      Check whether a collection name is valid.
    - RolloutResult:            Dataclass capturing a complete episode outcome.
    - run_rollout:              Run a single episode with a model function.
    - run_batch_rollouts:       Run multiple episodes.
    - DummyModelFn:             Debug model function for testing.
    - ShopRLVERewardFunction:   Reward function for TRL GRPOTrainer.
    - create_grpo_config:       Generate GRPOConfig-compatible dict.
    - ShopRLVEDataCollator:     Format episodes for TRL.

Usage:
    from shop_rlve.training import (
        COLLECTIONS, get_collection, run_rollout, DummyModelFn,
    )
    from shop_rlve.server import ShopRLVEEnv

    env = ShopRLVEEnv(collection="C1")
    dummy = DummyModelFn(env_id="PD", product_ids=["p1"])
    result = run_rollout(env, dummy)
"""

from shop_rlve.training.collections import (
    ALL_ENV_IDS,
    COLLECTIONS,
    collection_info,
    get_collection,
    validate_collection,
    validate_env_ids,
)
from shop_rlve.training.grpo import (
    ShopRLVEDataCollator,
    ShopRLVERewardFunction,
    create_grpo_config,
)
from shop_rlve.training.rollout import (
    DummyModelFn,
    RolloutResult,
    run_batch_rollouts,
    run_rollout,
)

__all__ = [
    # Collections
    "COLLECTIONS",
    "ALL_ENV_IDS",
    "get_collection",
    "validate_collection",
    "validate_env_ids",
    "collection_info",
    # Rollout
    "RolloutResult",
    "run_rollout",
    "run_batch_rollouts",
    "DummyModelFn",
    # GRPO
    "ShopRLVERewardFunction",
    "create_grpo_config",
    "ShopRLVEDataCollator",
]
