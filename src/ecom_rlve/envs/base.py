"""Abstract base environment and registry for EcomRLVE-GYM.

Every atomic environment inherits from BaseEnvironment and implements the
E = (I, P, R) abstraction:

    I  = generate_input(params)        -> initial user message
    P  = generate_problem(d, catalog)  -> ProblemParams
    R  = verify(answer, params, state) -> EpisodeResult

Shared helpers:
    generate_constraints   -- sample constraint predicates from a target product
    build_constraint_fn    -- convert {attr, op, value} to a callable
    build_evaluation_pool  -- top-K_eval products for nDCG computation
    ENV_REGISTRY / register_env / get_env -- env registration
"""

from __future__ import annotations

import math
import operator
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ecom_rlve.data.schema import ATTRIBUTE_ALLOWLIST, Product
from ecom_rlve.difficulty.mapping import DifficultyParams, map_difficulty
from ecom_rlve.rewards.metrics import constraint_satisfaction


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ProblemParams:
    """Parameters sampled by the problem generator."""

    env_id: str
    difficulty: int
    seed: int
    target_product_ids: list[str] = field(default_factory=list)
    constraints: list[dict[str, Any]] = field(default_factory=list)
    persona_weights: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Result of episode verification."""

    r_task: float
    is_correct: bool
    reward_breakdown: Any = None
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Operator lookup
# ---------------------------------------------------------------------------

_OP_MAP: dict[str, Callable[[Any, Any], bool]] = {
    "eq": operator.eq,
    "neq": operator.ne,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
}


# ---------------------------------------------------------------------------
# Base Environment ABC
# ---------------------------------------------------------------------------


class BaseEnvironment(ABC):
    """Abstract base for all atomic environments.  E = (I, P, R)."""

    ENV_ID: str = ""  # Override in subclass: "PD", "SUB", etc.

    # -----------------------------------------------------------------------
    # Abstract methods — must be implemented by each env
    # -----------------------------------------------------------------------

    @abstractmethod
    def generate_problem(
        self,
        difficulty: int,
        catalog: list[Product],
        seed: int,
        **kwargs: Any,
    ) -> ProblemParams:
        """P_d: Sample problem parameters at difficulty d."""

    @abstractmethod
    def generate_input(self, params: ProblemParams) -> str:
        """I: Generate the initial user message from problem params."""

    @abstractmethod
    def verify(
        self,
        answer: dict[str, Any],
        params: ProblemParams,
        episode_state: dict[str, Any],
    ) -> EpisodeResult:
        """R: Compute deterministic reward from episode output."""

    # -----------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def get_difficulty_params(difficulty: int) -> DifficultyParams:
        """Convenience: map difficulty integer to the full parameter vector."""
        return map_difficulty(difficulty)

    @staticmethod
    def generate_constraints(
        target_product: Product,
        n_constraints: int,
        difficulty: int,
        seed: int,
        exclude_attrs: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Sample *n_constraints* predicates that the target product satisfies.

        Discrete attributes use equality.  Numeric attributes use bounded
        predicates with slack controlled by difficulty:
            delta_max(d) = 0.5 * exp(-d / 5)

        Each predicate dict has keys: attr, op, value.

        Args:
            target_product: Product that must satisfy all generated constraints.
            n_constraints:  Number of constraints to sample.
            difficulty:     Difficulty level (controls numeric slack).
            seed:           Random seed for reproducibility.
            exclude_attrs:  Optional set of attribute names to exclude from
                            random sampling (e.g., {"price"} when the caller
                            handles price constraints separately).
        """
        rng = random.Random(seed)

        if exclude_attrs is None:
            exclude_attrs = set()

        # Separate candidate attributes into discrete and numeric
        discrete_attrs: list[str] = []
        numeric_attrs: list[str] = []

        # Top-level Product fields that live outside attrs dict
        _TOP_LEVEL_FIELDS = {"cat", "brand", "price", "rating", "ship_days",
                             "rating_count", "store"}

        # Gather attributes the target actually possesses
        product_attrs: dict[str, Any] = {}
        for attr_name, meta in ATTRIBUTE_ALLOWLIST.items():
            # Skip excluded attributes
            if attr_name in exclude_attrs:
                continue

            # Get value from product -- try top-level fields first, then attrs dict
            value: Any = None
            if attr_name in _TOP_LEVEL_FIELDS:
                value = getattr(target_product, attr_name, None)
            else:
                value = target_product.attrs.get(attr_name)

            # Skip empty / missing values
            if value is None or value == "":
                continue

            product_attrs[attr_name] = value
            if meta["type"] == "discrete":
                discrete_attrs.append(attr_name)
            else:
                numeric_attrs.append(attr_name)

        all_candidates = discrete_attrs + numeric_attrs
        if not all_candidates:
            return []

        n = min(n_constraints, len(all_candidates))
        chosen = rng.sample(all_candidates, n)

        delta_max = 0.5 * math.exp(-difficulty / 5.0)
        constraints: list[dict[str, Any]] = []

        for attr_name in chosen:
            meta = ATTRIBUTE_ALLOWLIST[attr_name]
            value = product_attrs[attr_name]

            if meta["type"] == "discrete":
                constraints.append({"attr": attr_name, "op": "eq", "value": value})
            else:
                # Numeric: randomly pick upper-bound or lower-bound predicate
                numeric_val = float(value)
                if rng.random() < 0.5:
                    # Upper bound: C(p) = 1[p.attr <= B] where B = val * (1+delta)
                    delta = rng.uniform(0, delta_max)
                    bound = numeric_val * (1.0 + delta) if numeric_val > 0 else numeric_val + delta
                    constraints.append({
                        "attr": attr_name,
                        "op": "lte",
                        "value": round(bound, 2),
                    })
                else:
                    # Lower bound: C(p) = 1[p.attr >= B] where B = val * (1-delta)
                    delta = rng.uniform(0, delta_max)
                    bound = numeric_val * (1.0 - delta) if numeric_val > 0 else numeric_val - delta
                    constraints.append({
                        "attr": attr_name,
                        "op": "gte",
                        "value": round(bound, 2),
                    })

        return constraints

    @staticmethod
    def build_constraint_fn(constraint: dict[str, Any]) -> Callable[..., float]:
        """Convert a {attr, op, value} dict to a callable predicate.

        Returns:
            A function ``(product) -> float`` returning 1.0 if the
            constraint is satisfied and 0.0 otherwise.
        """
        attr = constraint["attr"]
        op_name = constraint["op"]
        target_value = constraint["value"]
        op_fn = _OP_MAP.get(op_name, operator.eq)

        def _predicate(product: Any) -> float:
            # Extract attribute value from product
            if hasattr(product, attr):
                val = getattr(product, attr)
            elif hasattr(product, "attrs") and attr in product.attrs:
                val = product.attrs[attr]
            elif isinstance(product, dict):
                val = product.get(attr, product.get("attrs", {}).get(attr))
            else:
                return 0.0

            if val is None:
                return 0.0

            try:
                if isinstance(target_value, str):
                    return 1.0 if op_fn(str(val).lower().strip(), str(target_value).lower().strip()) else 0.0
                return 1.0 if op_fn(float(val), float(target_value)) else 0.0
            except (TypeError, ValueError):
                return 0.0

        return _predicate

    @staticmethod
    def build_constraint_fns(
        constraints: list[dict[str, Any]],
    ) -> list[Callable[..., float]]:
        """Convert a list of constraint dicts to callable predicates."""
        return [BaseEnvironment.build_constraint_fn(c) for c in constraints]

    @staticmethod
    def build_evaluation_pool(
        constraints: list[Callable[..., float]],
        catalog: list[Product],
        k_eval: int = 500,
    ) -> list[str]:
        """Find top *k_eval* products by constraint satisfaction for nDCG.

        Spec Section 6.1:
            P_eval = top_K_eval products scored by s(p|C).

        Returns:
            List of product IDs sorted by descending constraint satisfaction.
        """
        scored: list[tuple[str, float]] = []
        for product in catalog:
            s = constraint_satisfaction(product, constraints)
            scored.append((product.id, s))

        scored.sort(key=lambda t: t[1], reverse=True)
        return [pid for pid, _ in scored[:k_eval]]

    @staticmethod
    def products_by_id(catalog: list[Product]) -> dict[str, Product]:
        """Build an id -> Product lookup dict."""
        return {p.id: p for p in catalog}

    @staticmethod
    def price_range(catalog: list[Product]) -> tuple[float, float]:
        """Compute (p_low, p_high) from a catalog or pool of products."""
        if not catalog:
            return (0.0, 1.0)
        prices = [p.price for p in catalog]
        return (min(prices), max(prices))


# ---------------------------------------------------------------------------
# Environment registry
# ---------------------------------------------------------------------------

ENV_REGISTRY: dict[str, type[BaseEnvironment]] = {}


def register_env(cls: type[BaseEnvironment]) -> type[BaseEnvironment]:
    """Class decorator that registers an environment in ``ENV_REGISTRY``."""
    if not cls.ENV_ID:
        raise ValueError(f"Class {cls.__name__} must set ENV_ID before registration.")
    ENV_REGISTRY[cls.ENV_ID] = cls
    return cls


def get_env(env_id: str) -> BaseEnvironment:
    """Instantiate and return an environment by its ID.

    Raises:
        KeyError: If *env_id* is not in the registry.
    """
    cls = ENV_REGISTRY.get(env_id)
    if cls is None:
        available = sorted(ENV_REGISTRY.keys())
        raise KeyError(
            f"Unknown env_id '{env_id}'. Available: {available}"
        )
    return cls()
