"""Multi-turn episode rollout for EcomRLVE-GYM RL training (Spec Section 9).

Rollout: run one complete episode through the OpenEnv server.

Each rollout:
    1. env.reset() -> initial observation
    2. Loop: model generates action -> env.step(action) -> next obs
    3. Until done
    4. Return scalar reward + optional decomposed components

Provides:
    - RolloutResult:       Dataclass capturing a complete episode outcome.
    - run_rollout:         Run a single episode with a model function.
    - run_batch_rollouts:  Run multiple episodes, optionally in parallel.
    - DummyModelFn:        Debug lever for testing without a real model.

Usage:
    from ecom_rlve.server import EcomRLVEEnv
    from ecom_rlve.training.rollout import run_rollout, DummyModelFn

    env = EcomRLVEEnv(collection="C1", seed=42)
    dummy = DummyModelFn(env_id="PD", product_ids=["p1", "p2"])
    result = run_rollout(env, dummy)
    print(result.reward, result.turns)
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from ecom_rlve.server.openenv import EcomRLVEEnv
from ecom_rlve.server.state import Observation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout result
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    """Result of a single episode rollout.

    Captures the scalar reward, correctness flag, turn count, and optionally
    the full conversation and episode trace for analysis.

    Attributes:
        reward:           Final scalar reward in [-1, 1].
        is_correct:       Whether the agent's answer met IsCorrect threshold.
        turns:            Number of turns in the episode.
        env_id:           Environment identifier used for this episode.
        difficulty:       Difficulty level used for this episode.
        reward_breakdown: Decomposed reward components as a dict.
        conversation:     Full conversation history [{role, content}].
        episode_trace:    Complete episode trace dict (if collected).
    """

    reward: float = 0.0
    is_correct: bool = False
    turns: int = 0
    env_id: str = ""
    difficulty: int = 0
    reward_breakdown: dict[str, Any] = field(default_factory=dict)
    conversation: list[dict[str, str]] = field(default_factory=list)
    episode_trace: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------


def run_rollout(
    env: EcomRLVEEnv,
    model_fn: Callable[[list[dict[str, str]]], str],
    env_id: str | None = None,
    difficulty: int | None = None,
    seed: int | None = None,
    collect_trace: bool = False,
    max_steps: int = 50,
) -> RolloutResult:
    """Run a single complete episode through the OpenEnv server.

    The model function is called at each turn with the full conversation
    history and must return a valid action JSON string.

    Args:
        env:           EcomRLVEEnv instance.
        model_fn:      Callable that takes conversation (list of dicts) and
                       returns an action JSON string.
        env_id:        Force a specific environment (None = sample from collection).
        difficulty:    Force a specific difficulty (None = adaptive).
        seed:          Episode seed (None = auto).
        collect_trace: If True, include the full episode trace in the result.
        max_steps:     Safety limit on the number of steps (prevents infinite loops).

    Returns:
        RolloutResult with the episode outcome.
    """
    t_start = time.monotonic()

    # 1. Reset
    obs = env.reset(env_id=env_id, difficulty=difficulty, seed=seed)
    ep_state = env.get_episode_state()
    actual_env_id = ep_state.env_id if ep_state else ""
    actual_difficulty = ep_state.difficulty if ep_state else 0

    # 2. Loop until done
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}
    step_count = 0

    while not done and step_count < max_steps:
        # Model generates action from conversation
        try:
            action_json = model_fn(obs.conversation)
        except Exception as exc:
            logger.warning(
                "model_fn raised %s: %s. Sending empty action.",
                type(exc).__name__,
                exc,
            )
            action_json = json.dumps({
                "assistant_message": "I'm sorry, I encountered an error.",
                "tool_calls": [],
                "answer": {"env": actual_env_id, "done": True},
            })

        # Step the environment
        obs, reward, done, info = env.step(action_json)
        total_reward = reward  # Only the terminal reward matters
        step_count += 1

    # 3. Collect results
    is_correct = info.get("is_correct", False)
    turns = info.get("turn", step_count)
    reward_breakdown = info.get("reward_breakdown", {})
    conversation = obs.conversation

    episode_trace: dict[str, Any] | None = None
    if collect_trace:
        episode_trace = env.get_episode_trace()

    t_elapsed = time.monotonic() - t_start

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "run_rollout: env=%s, d=%d, turns=%d, reward=%.4f, "
            "correct=%s (%.1fms)",
            actual_env_id,
            actual_difficulty,
            turns,
            total_reward,
            is_correct,
            t_elapsed * 1000,
        )

    return RolloutResult(
        reward=total_reward,
        is_correct=is_correct,
        turns=turns,
        env_id=actual_env_id,
        difficulty=actual_difficulty,
        reward_breakdown=reward_breakdown,
        conversation=conversation,
        episode_trace=episode_trace,
    )


# ---------------------------------------------------------------------------
# Batch rollouts
# ---------------------------------------------------------------------------


def run_batch_rollouts(
    env: EcomRLVEEnv,
    model_fn: Callable[[list[dict[str, str]]], str],
    n_rollouts: int,
    collection: str | None = None,
    seeds: list[int] | None = None,
    collect_traces: bool = False,
    parallel: bool = False,
    max_workers: int = 4,
) -> list[RolloutResult]:
    """Run multiple rollouts, optionally in parallel.

    When parallel=True, note that EcomRLVEEnv is NOT thread-safe.
    You must create a separate env instance per worker for true
    parallel execution. This implementation runs sequentially by
    default and provides a parallel mode that creates independent
    env copies for each worker.

    Args:
        env:             EcomRLVEEnv instance (used as template for parallel).
        model_fn:        Model function (see run_rollout).
        n_rollouts:      Number of episodes to run.
        collection:      Collection name to cycle through (None = use env's collection).
        seeds:           Optional list of seeds, one per rollout.
        collect_traces:  If True, collect full episode traces.
        parallel:        If True, run rollouts in parallel threads.
        max_workers:     Number of parallel workers (only if parallel=True).

    Returns:
        List of RolloutResult, one per rollout.
    """
    if seeds is not None and len(seeds) != n_rollouts:
        raise ValueError(
            f"seeds list length ({len(seeds)}) must match n_rollouts ({n_rollouts})"
        )

    # Determine env_ids to cycle through
    env_ids = env.collection_env_ids
    if collection is not None:
        from ecom_rlve.training.collections import get_collection
        env_ids = get_collection(collection)

    if not parallel:
        # Sequential execution
        results: list[RolloutResult] = []
        for i in range(n_rollouts):
            env_id = env_ids[i % len(env_ids)]
            seed = seeds[i] if seeds is not None else None

            result = run_rollout(
                env=env,
                model_fn=model_fn,
                env_id=env_id,
                seed=seed,
                collect_trace=collect_traces,
            )
            results.append(result)

        return results

    # Parallel execution (create independent envs per worker)
    results_by_idx: dict[int, RolloutResult] = {}

    def _run_one(idx: int) -> tuple[int, RolloutResult]:
        """Run a single rollout in a worker thread."""
        # Create a fresh env for this worker
        worker_env = EcomRLVEEnv(
            collection=env._collection_name,
            catalog=(env._products, env._variants),
            config=env.config,
            seed=env._seed + idx + 1,
        )
        eid = env_ids[idx % len(env_ids)]
        s = seeds[idx] if seeds is not None else None

        result = run_rollout(
            env=worker_env,
            model_fn=model_fn,
            env_id=eid,
            seed=s,
            collect_trace=collect_traces,
        )
        worker_env.close()
        return idx, result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_one, i): i for i in range(n_rollouts)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results_by_idx[idx] = result

    # Return in original order
    return [results_by_idx[i] for i in range(n_rollouts)]


# ---------------------------------------------------------------------------
# DummyModelFn -- DEBUG lever for testing without a real model
# ---------------------------------------------------------------------------


class DummyModelFn:
    """Generates random valid actions for testing the env without a real model.

    This is a DEBUG lever: it produces syntactically valid action JSON
    with random tool calls and answers, enabling full end-to-end testing
    of the OpenEnv pipeline without requiring an actual LLM.

    The dummy model:
        - On turn 0: issues a catalog.search tool call
        - On turn 1-2: issues catalog.get_product calls
        - After that: submits a done answer with random product recommendations

    Args:
        env_id:      Target environment identifier.
        product_ids: List of product IDs to sample recommendations from.
        seed:        Random seed for reproducibility.

    Example:
        >>> dummy = DummyModelFn(env_id="PD", product_ids=["p1", "p2", "p3"])
        >>> action_json = dummy([{"role": "user", "content": "Find me headphones"}])
        >>> import json; action = json.loads(action_json)
        >>> "assistant_message" in action
        True
    """

    def __init__(
        self,
        env_id: str = "PD",
        product_ids: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.env_id = env_id
        self.product_ids = product_ids or []
        self._rng = random.Random(seed)
        self._call_count = 0

    def __call__(self, conversation: list[dict[str, str]]) -> str:
        """Generate a random valid action JSON string.

        Args:
            conversation: Full conversation history (not used by dummy
                          except to count turns).

        Returns:
            Valid action JSON string.
        """
        self._call_count += 1
        turn = self._call_count

        # Decide whether to search, explore, or submit answer
        if turn <= 1:
            # Issue a catalog search
            action = {
                "assistant_message": "Let me search for products that match your needs.",
                "tool_calls": [
                    {
                        "name": "catalog.search",
                        "args": {
                            "query": self._generate_search_query(conversation),
                            "top_k": 10,
                        },
                    }
                ],
            }
        elif turn <= 2 and self.product_ids:
            # Get product details
            pid = self._rng.choice(self.product_ids) if self.product_ids else "p_000"
            action = {
                "assistant_message": f"Let me get more details on {pid}.",
                "tool_calls": [
                    {
                        "name": "catalog.get_product",
                        "args": {"product_id": pid},
                    }
                ],
            }
        else:
            # Submit final answer
            action = self._generate_answer(conversation)

        return json.dumps(action)

    def _generate_search_query(
        self, conversation: list[dict[str, str]]
    ) -> str:
        """Extract a plausible search query from the conversation.

        Args:
            conversation: Conversation history.

        Returns:
            A search query string.
        """
        # Use the last user message as a simple query
        for msg in reversed(conversation):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # Take the first 50 characters as query
                return content[:50].strip() or "products"
        return "products"

    def _generate_answer(
        self, conversation: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Generate a final answer action.

        Args:
            conversation: Conversation history.

        Returns:
            Action dict with assistant_message and answer.
        """
        # Select random product IDs for recommendations
        n_recs = min(5, len(self.product_ids)) if self.product_ids else 0
        rec_ids = (
            self._rng.sample(self.product_ids, n_recs)
            if self.product_ids and n_recs > 0
            else []
        )

        answer: dict[str, Any] = {
            "env": self.env_id,
            "recommended_product_ids": rec_ids,
            "done": True,
        }

        # Env-specific answer fields
        if self.env_id == "STATUS":
            answer["selected_order_id"] = "ord_001"
            answer["order_status"] = "DELIVERED"
        elif self.env_id == "RETURN":
            answer["selected_order_id"] = "ord_001"
            answer["selected_line_id"] = "ord_001_line_01"
        elif self.env_id == "POLICY":
            answer["policy_answer"] = 30
        elif self.env_id == "CART":
            # No special fields needed beyond recommended_product_ids
            pass
        elif self.env_id == "BUNDLE":
            pass

        return {
            "assistant_message": (
                "Based on my research, here are my recommendations. "
                "I believe these are the best options for your needs."
            ),
            "tool_calls": [],
            "answer": answer,
        }
