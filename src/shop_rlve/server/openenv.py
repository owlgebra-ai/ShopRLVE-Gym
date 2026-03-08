"""OpenEnv server implementation for ShopRLVE-GYM (Spec Section 8).

Provides the core reset() / step() loop that orchestrates:
    1. Environment selection from a collection
    2. Difficulty sampling via adaptive engine
    3. Problem generation (ProblemParams)
    4. Tool execution with seen-set tracking
    5. User simulator dialogue
    6. Termination checking + deterministic reward computation

reset():
    1. Choose env from collection (uniform)
    2. Choose difficulty from adaptive [l_i, h_i] or forced
    3. Sample problem params via P_d
    4. Initialize state (T=0, Seen=empty, cart empty)
    5. Generate first user message
    6. Return Observation

step(action_json):
    1. Parse check: invalid JSON -> done=True, reward=-1
    2. Tool execution: validate + execute + track Seen
    3. Update conversation
    4. User simulator step
    5. Increment T
    6. Termination check: answer.done OR T==T_max OR user quit
    7. If terminal: compute reward via verifier + composer
    8. Else: reward=0
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from shop_rlve.data.catalog_loader import generate_synthetic_catalog
from shop_rlve.data.embeddings import EmbeddingEngine
from shop_rlve.data.index import MockVectorIndex
from shop_rlve.data.schema import DENIED_CATEGORIES, Product, Variant
from shop_rlve.difficulty.adaptive import AdaptiveDifficultyEngine
from shop_rlve.difficulty.mapping import DifficultyParams, map_difficulty
from shop_rlve.envs.base import ENV_REGISTRY, BaseEnvironment, EpisodeResult, ProblemParams, get_env
from shop_rlve.rewards.composer import RewardBreakdown, compose_reward
from shop_rlve.simulator.dialogue import UserSimulator
from shop_rlve.simulator.persona import PersonaWeights, sample_persona_weights
from shop_rlve.tools.cart import CartState, register_cart_tools
from shop_rlve.tools.catalog import CatalogState, register_catalog_tools
from shop_rlve.tools.orders import Order, generate_order_history, register_order_tools
from shop_rlve.tools.policy import PolicyKB, build_default_policy_kb, register_policy_tools
from shop_rlve.tools.registry import ToolCall, ToolRegistry, ToolResult
from shop_rlve.tools.returns import register_return_tools

from shop_rlve.server.state import (
    ActionSchema,
    AnswerSchema,
    EpisodeState,
    Observation,
    parse_action,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collection definitions (duplicated from training.collections for
# server self-containment; the canonical source is training.collections)
# ---------------------------------------------------------------------------

_COLLECTIONS: dict[str, list[str]] = {
    "C1": ["PD"],
    "C2": ["PD", "SUB"],
    "C4": ["PD", "SUB", "CART", "RETURN"],
    "C8": ["PD", "SUB", "CART", "RETURN", "STATUS", "POLICY", "BUNDLE", "JOURNEY"],
}

# Maps env_id short names to the registered ENV_REGISTRY keys
_ENV_ID_MAP: dict[str, str] = {
    "PD": "PD",
    "SUB": "SUB",
    "CART": "CART",
    "RETURN": "RETURN",
    "STATUS": "STATUS",
    "POLICY": "POLICY",
    "BUNDLE": "BUNDLE",
    "JOURNEY": "JOURNEY",
}


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------


def _default_config() -> dict[str, Any]:
    """Return the default configuration dict for ShopRLVEEnv.

    This mirrors what would be loaded from configs/default.yaml.
    """
    return {
        # Reward weights
        "w_task": 0.75,
        "w_eff": 0.15,
        "w_hall": 0.10,
        # Adaptive difficulty
        "tau_acc": 0.9,
        "tau_num": 32,
        "d_delta": 4,
        # Evaluation
        "K_eval": 500,
        "S_max": 14,
        # Catalog
        "n_synthetic_products": 1000,
        # Observation disclosure
        "disclose_env_id": True,
        "disclose_difficulty": False,
    }


def _load_config(path: str) -> dict[str, Any]:
    """Load configuration from a YAML file, merged with defaults.

    Args:
        path: Path to the YAML config file.

    Returns:
        Merged configuration dict.
    """
    import yaml

    config = _default_config()
    try:
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        config.update(user_config)
    except FileNotFoundError:
        logger.warning("Config file not found: %s. Using defaults.", path)
    return config


# ---------------------------------------------------------------------------
# ShopRLVEEnv -- the main OpenEnv server
# ---------------------------------------------------------------------------


class ShopRLVEEnv:
    """ShopRLVE-GYM OpenEnv server.

    Implements the reset()/step() contract for running RL training
    episodes over the 8 atomic e-commerce conversation environments.

    Attributes:
        collection:     Name of the environment collection (C1, C2, C4, C8).
        config:         Configuration dict with reward weights, adaptive params, etc.

    Debug levers:
        ShopRLVEEnv.validate_rewards = True
            Assert reward in [-1, 1] at every step.

        ShopRLVEEnv.trace_episodes = True
            Log full episode traces to the logger.

        ShopRLVEEnv.dump_dir = "debug_dumps"
            Dump episode JSON to disk on termination.

    Example:
        >>> env = ShopRLVEEnv(collection="C1", seed=42)
        >>> obs = env.reset()
        >>> obs.turn
        0
        >>> obs, reward, done, info = env.step('{"assistant_message": "Hi", "tool_calls": []}')
        >>> done
        False
    """

    # Class-level debug levers
    validate_rewards: bool = True
    trace_episodes: bool = False
    dump_dir: str = "debug_dumps"
    fixed_persona: PersonaWeights | None = None  # DEBUG lever: override Dirichlet sampling

    def __init__(
        self,
        collection: str = "C1",
        catalog: tuple[list[Product], list[Variant]] | None = None,
        config: dict[str, Any] | str | None = None,
        seed: int = 42,
        persona_alpha: np.ndarray | None = None,
    ) -> None:
        """Initialize the ShopRLVE-GYM environment server.

        Args:
            collection:    Environment collection name (C1, C2, C4, C8) or a
                           list of env_id strings.
            catalog:       Optional pre-built (products, variants) tuple.
                           If None, generates a synthetic catalog.
            config:        Configuration dict, path to YAML file, or None for defaults.
            seed:          Master random seed for reproducibility.
            persona_alpha: Optional Dirichlet alpha override (shape (5,)).
                           None = default [2, 2, 1, 1, 1].

        Raises:
            ValueError: If collection name is unknown.
        """
        # Resolve collection
        if isinstance(collection, str) and collection in _COLLECTIONS:
            self._collection_name = collection
            self._env_ids = list(_COLLECTIONS[collection])
        elif isinstance(collection, str):
            raise ValueError(
                f"Unknown collection '{collection}'. "
                f"Available: {sorted(_COLLECTIONS.keys())}"
            )
        else:
            self._collection_name = "custom"
            self._env_ids = list(collection)

        # Load config
        if isinstance(config, str):
            self.config = _load_config(config)
        elif isinstance(config, dict):
            merged = _default_config()
            merged.update(config)
            self.config = merged
        else:
            self.config = _default_config()

        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)
        self._episode_counter: int = 0
        self.persona_alpha = persona_alpha  # None = default [2,2,1,1,1]

        # Build catalog
        if catalog is not None:
            self._products, self._variants = catalog
        else:
            logger.info("ShopRLVEEnv: generating synthetic catalog...")
            self._products, self._variants = generate_synthetic_catalog(
                n_products=self.config.get("n_synthetic_products", 1000),
                seed=seed,
            )

        self._products_by_id: dict[str, Product] = {p.id: p for p in self._products}
        self._variants_by_product: dict[str, list[Variant]] = {}
        for v in self._variants:
            self._variants_by_product.setdefault(v.product_id, []).append(v)

        # Build embedding engine (debug mode for synthetic catalog)
        self._embedding_engine = EmbeddingEngine(debug_mode=True)

        # Build vector index (MockVectorIndex for synthetic/debug)
        self._vector_index = MockVectorIndex(dim=self._embedding_engine.dim)
        self._build_vector_index()

        # Build policy KB
        self._policy_kb = build_default_policy_kb()

        # Build tool registry
        self._tool_registry = ToolRegistry()
        register_catalog_tools(self._tool_registry)
        register_cart_tools(self._tool_registry)
        register_order_tools(self._tool_registry)
        register_return_tools(self._tool_registry)
        register_policy_tools(self._tool_registry)

        # Initialize adaptive difficulty engine
        self._adaptive_engine = AdaptiveDifficultyEngine(
            env_ids=self._env_ids,
            tau_acc=self.config.get("tau_acc", 0.9),
            tau_num=self.config.get("tau_num", 32),
            d_delta=self.config.get("d_delta", 4),
        )

        # Episode state (None when no episode is active)
        self._state: EpisodeState | None = None
        self._user_sim: UserSimulator | None = None

        logger.info(
            "ShopRLVEEnv initialized: collection=%s (%d envs), "
            "catalog=%d products, seed=%d",
            self._collection_name,
            len(self._env_ids),
            len(self._products),
            seed,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def collection_env_ids(self) -> list[str]:
        """Return the list of environment IDs in the current collection."""
        return list(self._env_ids)

    @property
    def adaptive_engine(self) -> AdaptiveDifficultyEngine:
        """Return the adaptive difficulty engine."""
        return self._adaptive_engine

    # ------------------------------------------------------------------
    # Internal: build vector index
    # ------------------------------------------------------------------

    def _build_vector_index(self) -> None:
        """Embed all products and build the vector index."""
        if not self._products:
            return

        texts = [f"{p.title} {p.desc}" for p in self._products]
        ids = [p.id for p in self._products]
        embeddings = self._embedding_engine.encode(texts, normalize=True)
        self._vector_index.build(embeddings, ids)

        logger.info(
            "ShopRLVEEnv: vector index built with %d products", len(self._products)
        )

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        env_id: str | None = None,
        difficulty: int | None = None,
        seed: int | None = None,
    ) -> Observation:
        """Start a new episode.

        Spec Section 8 -- reset():
            1. Choose env from collection (uniform) or use forced env_id
            2. Choose difficulty from adaptive [l_i, h_i] or forced
            3. Sample problem params via P_d
            4. Initialize state (T=0, Seen=empty, cart empty)
            5. Generate first user message
            6. Return Observation

        Args:
            env_id:     Force a specific environment (None = sample from collection).
            difficulty: Force a specific difficulty level (None = adaptive sampling).
            seed:       Episode seed (None = auto-increment from master seed).

        Returns:
            Observation with the first user message.

        Raises:
            ValueError: If forced env_id is not in the collection.
        """
        t_start = time.monotonic()
        self._episode_counter += 1

        # 1. Choose environment
        if env_id is not None:
            if env_id not in self._env_ids:
                raise ValueError(
                    f"env_id '{env_id}' is not in collection "
                    f"{self._collection_name}: {self._env_ids}"
                )
            chosen_env_id = env_id
        else:
            chosen_env_id = self._py_rng.choice(self._env_ids)

        # 2. Choose difficulty
        if difficulty is not None:
            chosen_difficulty = difficulty
        else:
            chosen_difficulty = self._adaptive_engine.sample_difficulty(
                chosen_env_id, rng=self._rng
            )

        # Episode seed
        if seed is not None:
            ep_seed = seed
        else:
            ep_seed = int(self._rng.integers(0, 2**31))

        ep_rng = random.Random(ep_seed)

        # 3. Sample problem params
        env_instance = get_env(chosen_env_id)
        diff_params = map_difficulty(chosen_difficulty)

        problem_params = env_instance.generate_problem(
            difficulty=chosen_difficulty,
            catalog=self._products,
            seed=ep_seed,
        )

        # 4. Initialize episode state
        # Build catalog state for this episode
        catalog_state = CatalogState(
            products=self._products,
            variants=self._variants,
            vector_index=self._vector_index,
            embedding_engine=self._embedding_engine,
            eps_rank=diff_params.eps_rank_val,
            seed=ep_seed,
        )

        # Generate order history for STATUS / RETURN envs
        orders: list[Order] = []
        if chosen_env_id in ("STATUS", "RETURN"):
            n_orders = diff_params.H_orders_val
            orders = generate_order_history(
                products=self._products,
                n_orders=n_orders,
                seed=ep_seed,
            )

        # Sample persona weights (with debug lever overrides)
        if self.fixed_persona is not None:
            persona_weights = self.fixed_persona
        elif self.persona_alpha is not None:
            persona_weights = sample_persona_weights(seed=ep_seed, alpha=self.persona_alpha)
        else:
            persona_weights = sample_persona_weights(seed=ep_seed)

        state = EpisodeState(
            env_id=chosen_env_id,
            difficulty=chosen_difficulty,
            hidden_goal=problem_params,
            persona_weights=persona_weights,
            products_by_id=dict(self._products_by_id),
            variants_by_product=dict(self._variants_by_product),
            orders=orders,
            cart=CartState(),
            catalog_state=catalog_state,
            policy_kb=self._policy_kb,
            seen_product_ids=set(),
            conversation=[],
            tool_results_history=[],
            turn=0,
            done=False,
            reward=None,
            initiated_returns=set(),
            seed=ep_seed,
            reward_breakdown=None,
            timing={},
            today=date.today().isoformat(),
        )

        # 5. Generate first user message
        goal_params = self._extract_goal_params(problem_params, chosen_env_id)

        self._user_sim = UserSimulator(
            persona_weights=persona_weights,
            goal_params=goal_params,
            env_id=chosen_env_id,
            p_missing=diff_params.p_missing_val,
            p_noise=diff_params.p_noise_val,
            T_patience=diff_params.T_max_val,
            seed=ep_seed,
        )

        initial_message = self._user_sim.generate_initial_message()
        state.conversation.append({"role": "user", "content": initial_message})

        self._state = state

        # Build observation
        obs = Observation(
            conversation=list(state.conversation),
            tool_results=[],
            turn=0,
            env_id=chosen_env_id if self.config.get("disclose_env_id", True) else None,
            difficulty=chosen_difficulty if self.config.get("disclose_difficulty", False) else None,
            done=False,
            reward=None,
        )

        t_elapsed = time.monotonic() - t_start
        state.timing["reset_ms"] = t_elapsed * 1000

        if self.trace_episodes:
            logger.info(
                "ShopRLVEEnv.reset(): env=%s, difficulty=%d, seed=%d, "
                "initial_message='%s' (%.1fms)",
                chosen_env_id,
                chosen_difficulty,
                ep_seed,
                initial_message[:80],
                t_elapsed * 1000,
            )

        return obs

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self,
        action_json: str,
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Process one assistant turn.

        Spec Section 8 -- step():
            1. Parse check: invalid JSON -> done=True, reward=-1
            2. Tool execution: validate + execute + track Seen
            3. Update conversation
            4. User simulator step
            5. Increment T
            6. Termination check: answer.done OR T==T_max OR user quit
            7. If terminal: compute reward via verifier + composer
            8. Else: reward=0

        Args:
            action_json: Raw JSON string from the LLM.

        Returns:
            Tuple of (observation, reward, done, info):
                - observation: Next Observation for the model.
                - reward: Scalar reward (0.0 during episode, final on done).
                - done: Whether the episode has terminated.
                - info: Debug dict with reward_breakdown, env_id, etc.

        Raises:
            RuntimeError: If called without a prior reset().
        """
        if self._state is None:
            raise RuntimeError(
                "step() called without a prior reset(). Call reset() first."
            )

        state = self._state
        if state.done:
            raise RuntimeError(
                "step() called on a terminated episode. Call reset() to start a new one."
            )

        t_start = time.monotonic()
        state.timing["_step_start"] = t_start
        diff_params = map_difficulty(state.difficulty)
        info: dict[str, Any] = {
            "env_id": state.env_id,
            "difficulty": state.difficulty,
            "turn": state.turn,
        }

        # ---------------------------------------------------------------
        # 1. Parse check
        # ---------------------------------------------------------------
        action, format_valid = parse_action(action_json)

        if not format_valid or action is None:
            return self._terminate_episode(
                reward=-1.0,
                format_valid=False,
                tool_valid=True,
                safety_valid=True,
                info=info,
                reason="format_invalid",
            )

        # ---------------------------------------------------------------
        # 2. Tool execution
        # ---------------------------------------------------------------
        tool_valid = True
        step_tool_results: list[dict[str, Any]] = []

        if action.tool_calls:
            tool_results = self._tool_registry.execute_batch(
                tool_calls=action.tool_calls,
                state=state,
                budget=diff_params.B_tool_val,
            )

            for tc, tr in zip(action.tool_calls, tool_results):
                result_entry = {
                    "name": tc.name,
                    "args": tc.args,
                    "result": tr.result,
                    "error": tr.error,
                    "duration_ms": tr.duration_ms,
                }
                step_tool_results.append(result_entry)
                state.tool_results_history.append(result_entry)

                if tr.error is not None:
                    tool_valid = False

                # Track seen product IDs from tool results
                self._extract_seen_ids(tr.result, state.seen_product_ids)

        # ---------------------------------------------------------------
        # 3. Safety check on recommended products
        # ---------------------------------------------------------------
        safety_valid = True
        if action.answer and action.answer.recommended_product_ids:
            for pid in action.answer.recommended_product_ids:
                product = state.products_by_id.get(pid)
                if product is not None:
                    cat_parts = product.cat.lower().split("/")
                    for part in cat_parts:
                        if part in DENIED_CATEGORIES:
                            safety_valid = False
                            break

        # ---------------------------------------------------------------
        # 4. Update conversation with assistant message
        # ---------------------------------------------------------------
        state.conversation.append({
            "role": "assistant",
            "content": action.assistant_message,
        })

        # ---------------------------------------------------------------
        # 5. Termination check
        # ---------------------------------------------------------------
        agent_done = action.answer is not None and action.answer.done
        t_max = diff_params.T_max_val
        state.turn += 1

        # Check for user ragequit or continuation
        user_quit = False
        user_message = ""

        if not agent_done:
            if self._user_sim is not None:
                user_message, user_quit = self._user_sim.generate_response(
                    assistant_message=action.assistant_message,
                    tool_results=step_tool_results,
                    progress_info=None,
                )
                state.conversation.append({"role": "user", "content": user_message})

        terminal = agent_done or state.turn >= t_max or user_quit

        # ---------------------------------------------------------------
        # 6. Compute reward if terminal
        # ---------------------------------------------------------------
        if terminal:
            if not format_valid:
                return self._terminate_episode(
                    reward=-1.0,
                    format_valid=False,
                    tool_valid=tool_valid,
                    safety_valid=safety_valid,
                    info=info,
                    reason="format_invalid",
                )

            if not tool_valid:
                return self._terminate_episode(
                    reward=-1.0,
                    format_valid=format_valid,
                    tool_valid=False,
                    safety_valid=safety_valid,
                    info=info,
                    reason="tool_invalid",
                )

            if not safety_valid:
                return self._terminate_episode(
                    reward=-1.0,
                    format_valid=format_valid,
                    tool_valid=tool_valid,
                    safety_valid=False,
                    info=info,
                    reason="safety_violation",
                )

            # Compute task reward via verifier
            r_task, is_correct = self._compute_task_reward(state, action)

            # Compose final reward
            output_ids = (
                action.answer.recommended_product_ids
                if action.answer is not None
                else []
            )

            breakdown = compose_reward(
                r_task=r_task,
                turns=state.turn,
                t_max=t_max,
                output_ids=output_ids,
                seen_ids=state.seen_product_ids,
                format_valid=format_valid,
                tool_valid=tool_valid,
                safety_valid=safety_valid,
                w_task=self.config.get("w_task", 0.75),
                w_eff=self.config.get("w_eff", 0.15),
                w_hall=self.config.get("w_hall", 0.10),
                is_correct=is_correct,
                debug=True,
            )

            return self._terminate_episode(
                reward=breakdown.r_total,
                format_valid=format_valid,
                tool_valid=tool_valid,
                safety_valid=safety_valid,
                info=info,
                reason="agent_done" if agent_done else ("t_max" if state.turn >= t_max else "user_quit"),
                breakdown=breakdown,
                is_correct=is_correct,
            )

        # ---------------------------------------------------------------
        # 7. Non-terminal: return observation with reward=0
        # ---------------------------------------------------------------
        obs = Observation(
            conversation=list(state.conversation),
            tool_results=step_tool_results,
            turn=state.turn,
            env_id=state.env_id if self.config.get("disclose_env_id", True) else None,
            difficulty=state.difficulty if self.config.get("disclose_difficulty", False) else None,
            done=False,
            reward=None,
        )

        t_elapsed = time.monotonic() - t_start
        info["step_ms"] = t_elapsed * 1000

        if self.trace_episodes:
            logger.info(
                "ShopRLVEEnv.step(): env=%s, turn=%d/%d, tools=%d, done=False (%.1fms)",
                state.env_id,
                state.turn,
                t_max,
                len(action.tool_calls),
                t_elapsed * 1000,
            )

        return obs, 0.0, False, info

    # ------------------------------------------------------------------
    # State access (debug)
    # ------------------------------------------------------------------

    def get_episode_state(self) -> EpisodeState | None:
        """Return the current episode state for debugging.

        Returns:
            The EpisodeState if an episode is active, None otherwise.
        """
        return self._state

    def get_episode_trace(self) -> dict[str, Any]:
        """Return a complete episode trace for replay and debugging.

        Returns:
            Dict containing all episode data: env_id, difficulty, seed,
            conversation, tool_results_history, reward, reward_breakdown,
            timing, and hidden_goal.
        """
        if self._state is None:
            return {}

        state = self._state
        breakdown_dict: dict[str, Any] = {}
        if state.reward_breakdown is not None:
            breakdown_dict = {
                "r_task": state.reward_breakdown.r_task,
                "r_eff": state.reward_breakdown.r_eff,
                "r_hall": state.reward_breakdown.r_hall,
                "r_total": state.reward_breakdown.r_total,
                "format_valid": state.reward_breakdown.format_valid,
                "tool_valid": state.reward_breakdown.tool_valid,
                "safety_valid": state.reward_breakdown.safety_valid,
                "is_correct": state.reward_breakdown.is_correct,
                "details": state.reward_breakdown.details,
            }

        return {
            "env_id": state.env_id,
            "difficulty": state.difficulty,
            "seed": state.seed,
            "turn": state.turn,
            "done": state.done,
            "reward": state.reward,
            "reward_breakdown": breakdown_dict,
            "conversation": list(state.conversation),
            "tool_results_history": state.tool_results_history,
            "seen_product_ids": sorted(state.seen_product_ids),
            "timing": dict(state.timing),
            "hidden_goal": _serialize_problem_params(state.hidden_goal),
        }

    def close(self) -> None:
        """Clean up resources.

        Resets internal state. Safe to call multiple times.
        """
        self._state = None
        self._user_sim = None
        logger.debug("ShopRLVEEnv.close() called")

    # ------------------------------------------------------------------
    # Internal: terminate episode
    # ------------------------------------------------------------------

    def _terminate_episode(
        self,
        reward: float,
        format_valid: bool,
        tool_valid: bool,
        safety_valid: bool,
        info: dict[str, Any],
        reason: str,
        breakdown: RewardBreakdown | None = None,
        is_correct: bool = False,
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Finalize the episode and return terminal observation.

        Args:
            reward:       Final scalar reward.
            format_valid: Whether the action format was valid.
            tool_valid:   Whether tool calls were valid.
            safety_valid: Whether safety checks passed.
            info:         Info dict to augment.
            reason:       Termination reason string.
            breakdown:    Optional full reward breakdown.
            is_correct:   Whether the answer is correct.

        Returns:
            (observation, reward, done=True, info) tuple.
        """
        state = self._state
        assert state is not None

        # Validate reward bounds
        if self.validate_rewards:
            assert -1.0 <= reward <= 1.0, (
                f"Reward {reward} out of bounds [-1, 1]"
            )

        # Update state
        state.done = True
        state.reward = reward

        if breakdown is None:
            breakdown = RewardBreakdown(
                r_task=0.0,
                r_eff=0.0,
                r_hall=0.0,
                r_total=reward,
                format_valid=format_valid,
                tool_valid=tool_valid,
                safety_valid=safety_valid,
                is_correct=is_correct,
            )
        state.reward_breakdown = breakdown

        # Update adaptive difficulty
        self._adaptive_engine.update(
            env_id=state.env_id,
            difficulty=state.difficulty,
            is_correct=is_correct,
        )

        # Build info
        info["reward_breakdown"] = {
            "r_task": breakdown.r_task,
            "r_eff": breakdown.r_eff,
            "r_hall": breakdown.r_hall,
            "r_total": breakdown.r_total,
            "format_valid": breakdown.format_valid,
            "tool_valid": breakdown.tool_valid,
            "safety_valid": breakdown.safety_valid,
            "is_correct": breakdown.is_correct,
            "details": breakdown.details,
        }
        info["is_correct"] = is_correct
        info["termination_reason"] = reason
        info["turn"] = state.turn

        # Build observation
        obs = Observation(
            conversation=list(state.conversation),
            tool_results=[],
            turn=state.turn,
            env_id=state.env_id if self.config.get("disclose_env_id", True) else None,
            difficulty=state.difficulty if self.config.get("disclose_difficulty", False) else None,
            done=True,
            reward=reward,
        )

        t_elapsed = time.monotonic() - state.timing.get("_step_start", time.monotonic())
        state.timing["total_ms"] = t_elapsed * 1000

        if self.trace_episodes:
            logger.info(
                "ShopRLVEEnv TERMINAL: env=%s, difficulty=%d, turn=%d, "
                "reward=%.4f, is_correct=%s, reason=%s",
                state.env_id,
                state.difficulty,
                state.turn,
                reward,
                is_correct,
                reason,
            )

        # Dump trace to disk if configured
        if self.dump_dir:
            self._maybe_dump_trace(state)

        return obs, reward, True, info

    # ------------------------------------------------------------------
    # Internal: compute task reward via per-env verifiers
    # ------------------------------------------------------------------

    def _compute_task_reward(
        self,
        state: EpisodeState,
        action: ActionSchema,
    ) -> tuple[float, bool]:
        """Compute the task-specific reward r_task and is_correct.

        Dispatches to the appropriate per-environment verifier based
        on state.env_id.

        Args:
            state:  Current episode state.
            action: Parsed action from the LLM.

        Returns:
            Tuple of (r_task, is_correct).
        """
        env_id = state.env_id
        answer = action.answer
        problem = state.hidden_goal

        if answer is None:
            # No answer submitted (e.g., T_max reached without done)
            return -1.0, False

        try:
            env_instance = get_env(env_id)
            answer_dict = answer.model_dump()

            episode_state_dict: dict[str, Any] = {
                "seen_product_ids": state.seen_product_ids,
                "products_by_id": state.products_by_id,
                "cart": state.cart,
                "orders": state.orders,
                "initiated_returns": state.initiated_returns,
                "conversation": state.conversation,
                "turn": state.turn,
                "difficulty": state.difficulty,
            }

            result: EpisodeResult = env_instance.verify(
                answer=answer_dict,
                params=problem,
                episode_state=episode_state_dict,
            )
            return result.r_task, result.is_correct

        except Exception as exc:
            logger.warning(
                "Verifier error for env=%s: %s: %s",
                env_id,
                type(exc).__name__,
                exc,
            )
            return -1.0, False

    # ------------------------------------------------------------------
    # Internal: extract seen product IDs from tool results
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_seen_ids(result: Any, seen: set[str]) -> None:
        """Extract product IDs from a tool result and add to the seen set.

        Handles common result shapes:
            - list of dicts with 'product_id' or 'id' keys (catalog_search)
            - single dict with 'id' key (catalog_get_product)
            - list of dicts with 'id' key (catalog_get_variants)

        Args:
            result: The tool result payload (can be list, dict, or other).
            seen:   Mutable set to add found product IDs into.
        """
        if result is None:
            return

        if isinstance(result, dict):
            # Single product result (e.g., catalog_get_product)
            for key in ("product_id", "id"):
                if key in result and isinstance(result[key], str):
                    seen.add(result[key])
                    break
        elif isinstance(result, list):
            # List of results (e.g., catalog_search, catalog_get_variants)
            for item in result:
                if isinstance(item, dict):
                    for key in ("product_id", "id"):
                        if key in item and isinstance(item[key], str):
                            seen.add(item[key])
                            break

    # ------------------------------------------------------------------
    # Internal: extract goal params for user simulator
    # ------------------------------------------------------------------

    def _extract_goal_params(
        self,
        problem: ProblemParams,
        env_id: str,
    ) -> dict[str, Any]:
        """Extract goal parameters from ProblemParams for user simulator templates.

        The user simulator needs slot values to fill templates. We extract
        relevant info from the problem params and target products.

        Args:
            problem: ProblemParams from the environment generator.
            env_id:  Environment identifier.

        Returns:
            Dict of slot_name -> value for template rendering.
        """
        goal_params: dict[str, Any] = {}

        # Common: extract from target products
        if problem.target_product_ids:
            target_id = problem.target_product_ids[0]
            target = self._products_by_id.get(target_id)
            if target is not None:
                goal_params["category"] = target.cat
                goal_params["product_type"] = target.cat.split("/")[-1] if "/" in target.cat else target.cat
                # Derive price_max from target if not set by a constraint later
                goal_params["price_max"] = f"{target.price * 1.2:.0f}"

        # Extract constraints as user-facing descriptions
        for constraint in problem.constraints:
            attr = constraint.get("attr", "")
            value = constraint.get("value")
            if attr == "price" and constraint.get("op") == "lte":
                goal_params["price_max"] = str(value)
            elif attr == "price" and constraint.get("op") == "gte":
                goal_params["price_min"] = str(value)
            elif attr == "brand" and constraint.get("op") == "eq":
                goal_params["brand_pref"] = str(value)
            elif attr == "rating" and constraint.get("op") == "gte":
                goal_params["rating_req"] = str(value)
            elif attr == "ship_days" and constraint.get("op") == "lte":
                goal_params["ship_req"] = str(value)
            elif attr == "color" and constraint.get("op") == "eq":
                goal_params["color_pref"] = str(value)
            elif attr == "size" and constraint.get("op") == "eq":
                goal_params["size_pref"] = str(value)
            elif attr == "material" and constraint.get("op") == "eq":
                goal_params["material_pref"] = str(value)

        # Env-specific extra params
        extra = problem.extra or {}
        if env_id == "RETURN":
            goal_params["reason"] = extra.get("reason", "defective")
            goal_params["replacement_req"] = str(extra.get("replacement_required", False))
        elif env_id == "STATUS":
            goal_params["order_query"] = extra.get("order_query", "my recent order")
        elif env_id == "POLICY":
            goal_params["policy_question"] = extra.get("question_text", "")
        elif env_id == "BUNDLE":
            goal_params["budget"] = str(extra.get("budget", ""))
            categories = extra.get("required_categories", [])
            goal_params["project_categories"] = ", ".join(categories)

        return goal_params

    # ------------------------------------------------------------------
    # Internal: dump trace to disk
    # ------------------------------------------------------------------

    def _maybe_dump_trace(self, state: EpisodeState) -> None:
        """Conditionally dump the episode trace to JSON on disk.

        Only writes if ``self.dump_dir`` is set and non-empty.

        Args:
            state: The terminated episode state.
        """
        if not self.dump_dir:
            return

        try:
            dump_path = Path(self.dump_dir)
            dump_path.mkdir(parents=True, exist_ok=True)

            filename = (
                f"episode_{self._episode_counter:06d}_"
                f"{state.env_id}_d{state.difficulty}_s{state.seed}.json"
            )
            filepath = dump_path / filename

            trace = self.get_episode_trace()
            with open(filepath, "w") as f:
                json.dump(trace, f, indent=2, default=str)

            if self.trace_episodes:
                logger.info("Episode trace dumped to: %s", filepath)

        except Exception as exc:
            logger.warning("Failed to dump episode trace: %s", exc)


# ---------------------------------------------------------------------------
# Utility: serialize ProblemParams for trace
# ---------------------------------------------------------------------------


def _serialize_problem_params(params: Any) -> dict[str, Any]:
    """Best-effort serialization of ProblemParams to a JSON-safe dict.

    Args:
        params: ProblemParams or any object with common fields.

    Returns:
        Serializable dict representation.
    """
    if params is None:
        return {}

    if hasattr(params, "__dict__"):
        result: dict[str, Any] = {}
        for key, value in vars(params).items():
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                result[key] = str(value)
        return result

    return {"raw": str(params)}
