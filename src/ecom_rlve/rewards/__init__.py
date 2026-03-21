"""Rewards and verifiers for EcomRLVE-GYM.

Provides metric computations, reward composition, and per-environment
verification logic. All rewards are deterministic and in [-1, 1].

Components:
    - metrics:   Shared metric functions (nDCG, F1, constraint satisfaction, etc.)
    - composer:  Reward composition with hard-fail checks and weighted combination
    - verifiers: Per-environment verification logic (E_PD, E_SUB, E_CART, etc.)
"""

from ecom_rlve.rewards.composer import RewardBreakdown, compose_reward
from ecom_rlve.rewards.metrics import (
    constraint_satisfaction,
    dcg,
    efficiency_reward,
    f1_score,
    feasibility,
    hallucination_rate,
    hallucination_reward,
    ndcg,
    ndcg_reward,
    unit_f1,
)
from ecom_rlve.rewards.verifiers import (
    verify_bundle,
    verify_cart,
    verify_journey,
    verify_order_status,
    verify_policy,
    verify_product_discovery,
    verify_return,
    verify_substitution,
)

__all__ = [
    # metrics
    "dcg",
    "ndcg",
    "ndcg_reward",
    "constraint_satisfaction",
    "feasibility",
    "f1_score",
    "unit_f1",
    "hallucination_rate",
    "efficiency_reward",
    "hallucination_reward",
    # composer
    "RewardBreakdown",
    "compose_reward",
    # verifiers
    "verify_product_discovery",
    "verify_substitution",
    "verify_cart",
    "verify_return",
    "verify_order_status",
    "verify_policy",
    "verify_bundle",
    "verify_journey",
]
