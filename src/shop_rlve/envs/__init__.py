"""ShopRLVE-GYM: 8 atomic environments + base + registry.

Environment IDs:
    PD      -- Product Discovery
    SUB     -- OOS Substitution
    CART    -- Cart Building
    RETURN  -- Return + Replacement
    STATUS  -- Order Status
    POLICY  -- Policy QA
    BUNDLE  -- Bundle / Project Planning
    JOURNEY -- Multi-intent Composite

Usage:
    from shop_rlve.envs import get_env, ENV_REGISTRY

    env = get_env("PD")
    params = env.generate_problem(difficulty=3, catalog=catalog, seed=42)
    msg = env.generate_input(params)
    result = env.verify(answer, params, episode_state)
"""

# Base classes and registry
from shop_rlve.envs.base import (
    ENV_REGISTRY,
    BaseEnvironment,
    EpisodeResult,
    ProblemParams,
    get_env,
    register_env,
)

# Import all environments so they register themselves via @register_env
from shop_rlve.envs.product_discovery import ProductDiscoveryEnv
from shop_rlve.envs.substitution import SubstitutionEnv
from shop_rlve.envs.cart import CartBuildingEnv
from shop_rlve.envs.returns import ReturnEnv
from shop_rlve.envs.order_status import OrderStatusEnv
from shop_rlve.envs.policy_qa import PolicyQAEnv
from shop_rlve.envs.bundle import BundleEnv
from shop_rlve.envs.journey import JourneyEnv

__all__ = [
    # Registry
    "ENV_REGISTRY",
    "BaseEnvironment",
    "EpisodeResult",
    "ProblemParams",
    "get_env",
    "register_env",
    # Environments
    "ProductDiscoveryEnv",
    "SubstitutionEnv",
    "CartBuildingEnv",
    "ReturnEnv",
    "OrderStatusEnv",
    "PolicyQAEnv",
    "BundleEnv",
    "JourneyEnv",
]
