"""Runtime validation checks for EcomRLVE-GYM (debug levers).

Provides validation functions that can be turned on/off to verify
internal consistency of rewards, episode state, and environment
solvability.

Usage:
    from ecom_rlve.debug.validators import (
        validate_reward_bounds,
        validate_episode_state,
        validate_env_solvability,
        validate_all_envs,
    )

    ok = validate_reward_bounds(reward, context="E_PD step 3")
    warnings = validate_episode_state(state)
    results = validate_env_solvability(env_instance, catalog, difficulty=5)
    full = validate_all_envs(catalog, max_difficulty=10)
"""

from __future__ import annotations

import logging
from typing import Any

from ecom_rlve.data.schema import Product
from ecom_rlve.envs.base import (
    ENV_REGISTRY,
    BaseEnvironment,
    EpisodeResult,
    ProblemParams,
    get_env,
)
from ecom_rlve.server.state import EpisodeState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# validate_reward_bounds
# ---------------------------------------------------------------------------


def validate_reward_bounds(reward: float, context: str = "") -> bool:
    """Assert that a reward value is within [-1, 1].

    Raises AssertionError if the reward is out of bounds.

    Args:
        reward:  Scalar reward to validate.
        context: Optional context string for the error message.

    Returns:
        True if the reward is within bounds.

    Raises:
        AssertionError: If reward is not in [-1, 1].
    """
    if -1.0 <= reward <= 1.0:
        return True

    ctx = f" ({context})" if context else ""
    msg = f"Reward out of bounds{ctx}: {reward:.6f} not in [-1, 1]"
    logger.warning(msg)
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# validate_episode_state
# ---------------------------------------------------------------------------


def validate_episode_state(state: EpisodeState) -> list[str]:
    """Validate internal consistency of an EpisodeState.

    Checks:
        - seen_product_ids non-empty if tools were called
        - turn count consistent with conversation length
        - cart state consistent (no negative quantities, etc.)

    Args:
        state: The EpisodeState to validate.

    Returns:
        List of warning/error strings (empty if all checks pass).
    """
    warnings: list[str] = []

    # Check 1: If tool results exist, seen_product_ids should likely be non-empty
    if state.tool_results_history and not state.seen_product_ids:
        # Only warn for envs that use product tools
        if state.env_id in ("PD", "SUB", "CART", "BUNDLE", "JOURNEY"):
            warnings.append(
                f"Tool calls were made ({len(state.tool_results_history)}) "
                f"but seen_product_ids is empty (env={state.env_id})"
            )

    # Check 2: Turn count consistency with conversation
    # Each turn consists of at least one user + one assistant message.
    # After reset: 1 user message, turn=0.
    # After first step: 1 user + 1 assistant (+ possibly another user), turn=1.
    n_assistant_msgs = sum(
        1 for msg in state.conversation if msg.get("role") == "assistant"
    )
    if state.turn != n_assistant_msgs:
        warnings.append(
            f"Turn count mismatch: state.turn={state.turn} but "
            f"found {n_assistant_msgs} assistant messages in conversation"
        )

    # Check 3: Cart state consistency
    for line in state.cart.lines:
        if line.qty < 1:
            warnings.append(
                f"Cart line {line.line_id} has qty={line.qty} (should be >= 1)"
            )
        if line.unit_price <= 0:
            warnings.append(
                f"Cart line {line.line_id} has unit_price={line.unit_price} (should be > 0)"
            )

    # Check 4: Reward bounds (if episode is done)
    if state.done and state.reward is not None:
        if not (-1.0 <= state.reward <= 1.0):
            warnings.append(
                f"Final reward out of bounds: {state.reward} not in [-1, 1]"
            )

    # Check 5: Env ID should be non-empty
    if not state.env_id:
        warnings.append("env_id is empty")

    # Check 6: Conversation should have at least one message
    if not state.conversation:
        warnings.append("Conversation is empty (expected at least the initial user message)")

    return warnings


# ---------------------------------------------------------------------------
# validate_env_solvability
# ---------------------------------------------------------------------------


def validate_env_solvability(
    env: BaseEnvironment,
    catalog: list[Product],
    difficulty: int,
    n_trials: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Check whether an environment can generate solvable problems.

    For each trial, generates a problem at the given difficulty and checks
    whether the target products actually satisfy the constraints (i.e., an
    oracle with perfect knowledge could solve it).

    Args:
        env:        BaseEnvironment instance.
        catalog:    Product catalog list.
        difficulty: Difficulty level to test.
        n_trials:   Number of problems to generate.
        seed:       Base random seed.

    Returns:
        Dict with:
            - env_id:        str
            - difficulty:    int
            - n_trials:      int
            - solvable_rate: float
            - failures:      list[dict] describing each unsolvable problem
    """
    products_by_id: dict[str, Product] = {p.id: p for p in catalog}
    solvable_count = 0
    failures: list[dict[str, Any]] = []

    for i in range(n_trials):
        trial_seed = seed + i
        try:
            params: ProblemParams = env.generate_problem(
                difficulty=difficulty,
                catalog=catalog,
                seed=trial_seed,
            )
        except Exception as exc:
            failures.append({
                "trial": i,
                "seed": trial_seed,
                "error": f"generate_problem failed: {type(exc).__name__}: {exc}",
            })
            continue

        # Check that target products exist in catalog and satisfy constraints
        if not params.target_product_ids:
            failures.append({
                "trial": i,
                "seed": trial_seed,
                "error": "No target_product_ids generated",
            })
            continue

        all_targets_feasible = True
        for pid in params.target_product_ids:
            product = products_by_id.get(pid)
            if product is None:
                all_targets_feasible = False
                failures.append({
                    "trial": i,
                    "seed": trial_seed,
                    "error": f"Target product {pid} not found in catalog",
                })
                break

            # Check constraints
            if params.constraints:
                constraint_fns = env.build_constraint_fns(params.constraints)
                for cfn in constraint_fns:
                    if cfn(product) < 0.5:
                        all_targets_feasible = False
                        failures.append({
                            "trial": i,
                            "seed": trial_seed,
                            "error": (
                                f"Target product {pid} does not satisfy "
                                f"constraint (score < 0.5)"
                            ),
                        })
                        break
                if not all_targets_feasible:
                    break

        if all_targets_feasible:
            solvable_count += 1

    return {
        "env_id": env.ENV_ID,
        "difficulty": difficulty,
        "n_trials": n_trials,
        "solvable_rate": solvable_count / max(n_trials, 1),
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# validate_all_envs
# ---------------------------------------------------------------------------


def validate_all_envs(
    catalog: list[Product],
    max_difficulty: int = 10,
    n_trials: int = 20,
) -> dict[str, Any]:
    """Run solvability checks for all 8 environments across difficulty levels.

    Args:
        catalog:        Product catalog list.
        max_difficulty: Maximum difficulty to test (0 through max_difficulty inclusive).
        n_trials:       Number of trials per (env, difficulty) pair.

    Returns:
        Dict mapping env_id -> list of per-difficulty result dicts.
        Each result dict has: difficulty, solvable_rate, n_trials, n_failures.
    """
    results: dict[str, Any] = {}

    for env_id, env_cls in sorted(ENV_REGISTRY.items()):
        env_instance = env_cls()
        env_results: list[dict[str, Any]] = []

        for d in range(max_difficulty + 1):
            try:
                result = validate_env_solvability(
                    env=env_instance,
                    catalog=catalog,
                    difficulty=d,
                    n_trials=n_trials,
                )
                env_results.append({
                    "difficulty": d,
                    "solvable_rate": result["solvable_rate"],
                    "n_trials": result["n_trials"],
                    "n_failures": len(result["failures"]),
                })
            except Exception as exc:
                logger.warning(
                    "validate_all_envs: %s at d=%d failed: %s", env_id, d, exc
                )
                env_results.append({
                    "difficulty": d,
                    "solvable_rate": 0.0,
                    "n_trials": n_trials,
                    "n_failures": n_trials,
                    "error": str(exc),
                })

        results[env_id] = env_results
        logger.info(
            "validate_all_envs: %s complete (d=0..%d)", env_id, max_difficulty
        )

    return results
