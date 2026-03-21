"""OpenEnv server implementation for EcomRLVE-GYM (Spec Section 8).

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

from ecom_rlve.data.catalog_loader import generate_synthetic_catalog
from ecom_rlve.data.embeddings import EmbeddingEngine
from ecom_rlve.data.index import MockVectorIndex, VectorIndex
from ecom_rlve.data.schema import DENIED_CATEGORIES, Product, Variant
from ecom_rlve.difficulty.adaptive import AdaptiveDifficultyEngine
from ecom_rlve.difficulty.mapping import DifficultyParams, map_difficulty
from ecom_rlve.envs.base import ENV_REGISTRY, BaseEnvironment, EpisodeResult, ProblemParams, get_env
from ecom_rlve.rewards.composer import RewardBreakdown, compose_reward
from ecom_rlve.simulator.dialogue import UserAct, UserSimulator
from ecom_rlve.simulator.persona import PersonaWeights, sample_persona_weights
from ecom_rlve.tools.cart import CartState, register_cart_tools
from ecom_rlve.tools.catalog import CatalogState, register_catalog_tools
from ecom_rlve.tools.orders import Order, generate_order_history, register_order_tools
from ecom_rlve.tools.policy import PolicyKB, build_default_policy_kb, register_policy_tools
from ecom_rlve.tools.registry import ToolCall, ToolRegistry, ToolResult
from ecom_rlve.tools.datetime_tool import register_datetime_tools
from ecom_rlve.tools.returns import register_return_tools
from ecom_rlve.tools.user import register_user_tools

from ecom_rlve.server.state import (
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
    """Return the default configuration dict for EcomRLVEEnv.

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
# EcomRLVEEnv -- the main OpenEnv server
# ---------------------------------------------------------------------------


class EcomRLVEEnv:
    """EcomRLVE-GYM OpenEnv server.

    Implements the reset()/step() contract for running RL training
    episodes over the 8 atomic e-commerce conversation environments.

    Attributes:
        collection:     Name of the environment collection (C1, C2, C4, C8).
        config:         Configuration dict with reward weights, adaptive params, etc.

    Debug levers:
        EcomRLVEEnv.validate_rewards = True
            Assert reward in [-1, 1] at every step.

        EcomRLVEEnv.trace_episodes = True
            Log full episode traces to the logger.

        EcomRLVEEnv.dump_dir = "debug_dumps"
            Dump episode JSON to disk on termination.

    Example:
        >>> env = EcomRLVEEnv(collection="C1", seed=42)
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
        """Initialize the EcomRLVE-GYM environment server.

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
            logger.info("EcomRLVEEnv: generating synthetic catalog...")
            self._products, self._variants = generate_synthetic_catalog(
                n_products=self.config.get("n_synthetic_products", 1000),
                seed=seed,
            )

        self._products_by_id: dict[str, Product] = {p.id: p for p in self._products}
        self._variants_by_product: dict[str, list[Variant]] = {}
        for v in self._variants:
            self._variants_by_product.setdefault(v.product_id, []).append(v)

        # Build embedding engine
        # Config keys: embedding_model (str), embedding_debug (bool)
        emb_model = self.config.get("embedding_model", "thenlper/gte-small")
        emb_debug = self.config.get("embedding_debug", catalog is None)  # auto: debug for synthetic, real for real catalog
        emb_device = self.config.get("embedding_device", None)
        self._embedding_engine = EmbeddingEngine(
            model_name=emb_model,
            device=emb_device,
            debug_mode=emb_debug,
        )

        # Build vector index
        # Check if a pre-built FAISS index should be loaded from disk
        faiss_index_path = self.config.get("faiss_index_path", None)

        if faiss_index_path is not None:
            # Load pre-built index from disk (skips expensive re-encoding)
            try:
                self._vector_index = VectorIndex(
                    dim=self._embedding_engine.dim,
                )
                self._vector_index.load_from_dir(faiss_index_path)
                logger.info(
                    "EcomRLVEEnv: loaded pre-built FAISS index from %s (%d vectors)",
                    faiss_index_path, len(self._vector_index),
                )

                # If the index covers more products than loaded, restrict
                # FAISS search to only the loaded product IDs so that
                # results are guaranteed to exist in products_by_id.
                if len(self._vector_index) > len(self._products):
                    loaded_ids = {p.id for p in self._products}
                    self._vector_index.set_allowed_ids(loaded_ids)
                    logger.info(
                        "EcomRLVEEnv: restricted FAISS search to %d loaded products",
                        len(loaded_ids),
                    )
            except (ImportError, FileNotFoundError) as exc:
                logger.error("Failed to load FAISS index from %s: %s", faiss_index_path, exc)
                raise
        elif not emb_debug:
            # Build new index at runtime (encodes all products)
            try:
                index_factory = self.config.get("faiss_index_factory", "Flat")
                self._vector_index = VectorIndex(
                    dim=self._embedding_engine.dim,
                    index_factory=index_factory,
                )
                logger.info(
                    "EcomRLVEEnv: using FAISS VectorIndex (factory='%s')",
                    index_factory,
                )
            except ImportError:
                logger.warning(
                    "faiss-cpu not installed, falling back to MockVectorIndex"
                )
                self._vector_index = MockVectorIndex(dim=self._embedding_engine.dim)
            self._build_vector_index()
        else:
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
        register_datetime_tools(self._tool_registry)
        register_user_tools(self._tool_registry)

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
            "EcomRLVEEnv initialized: collection=%s (%d envs), "
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
            "EcomRLVEEnv: vector index built with %d products", len(self._products)
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

        # Inject synthetic variants for CART env so catalog.get_variants sees them
        ep_variants_by_product = dict(self._variants_by_product)
        if chosen_env_id == "CART":
            synth_data = problem_params.extra.get("synthetic_variants", [])
            for vd in synth_data:
                v = Variant(**vd)
                catalog_state.variants_by_product.setdefault(v.product_id, []).append(v)
                ep_variants_by_product.setdefault(v.product_id, []).append(v)

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
            variants_by_product=ep_variants_by_product,
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
            visit_history=(
                problem_params.extra.get("visit_history", [])
                if chosen_env_id == "CART" else []
            ),
        )

        # 5. Generate first user message
        goal_params = self._extract_goal_params(problem_params, chosen_env_id)

        # Build human-readable summaries for LLM user simulation context
        goal_summary = self._build_goal_summary(problem_params, chosen_env_id)
        persona_summary = self._build_persona_summary(persona_weights)

        self._user_sim = UserSimulator(
            persona_weights=persona_weights,
            goal_params=goal_params,
            env_id=chosen_env_id,
            p_missing=diff_params.p_missing_val,
            p_noise=diff_params.p_noise_val,
            T_patience=diff_params.T_max_val,
            seed=ep_seed,
            goal_summary=goal_summary,
            persona_summary=persona_summary,
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
                "EcomRLVEEnv.reset(): env=%s, difficulty=%d, seed=%d, "
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
                # Compute progress_info so the user sim can give
                # ground-truth-aware feedback (especially for CART)
                progress_info = self._compute_progress_info(state, step_tool_results)

                user_message, user_quit, user_act = self._user_sim.generate_response(
                    assistant_message=action.assistant_message,
                    tool_results=step_tool_results,
                    progress_info=progress_info,
                )
                state.user_act_history.append(user_act.value)
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

            # Count user-driven turns that should not penalise efficiency.
            # Only CONFIRM and CLARIFY are clearly user-initiated turns
            # where the agent was responding to user questions/confirmations.
            # CONTINUE is the fallback and often masks user elaboration, so
            # it is NOT discounted.
            _NON_PENALTY_ACTS = {"confirm", "clarify"}
            user_clarification_turns = sum(
                1 for act in state.user_act_history if act in _NON_PENALTY_ACTS
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
                user_clarification_turns=user_clarification_turns,
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
                "EcomRLVEEnv.step(): env=%s, turn=%d/%d, tools=%d, done=False (%.1fms)",
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
        logger.debug("EcomRLVEEnv.close() called")

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
        info["user_act_history"] = list(state.user_act_history)

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
                "EcomRLVEEnv TERMINAL: env=%s, difficulty=%d, turn=%d, "
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
                "cart_lines": [
                    {
                        "product_id": line.product_id,
                        "variant_id": line.variant_id,
                        "qty": line.qty,
                    }
                    for line in state.cart.lines
                ],
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
    # Internal: detect presented product candidates for CART confirmation
    # ------------------------------------------------------------------

    def _extract_presented_product_ids(
        self,
        state: EpisodeState,
        step_tool_results: list[dict[str, Any]],
    ) -> set[str]:
        """Extract product IDs that the agent may be presenting to the user.

        Checks if the agent called catalog.search, catalog.rerank, or
        user.get_visit_history this turn, and if the assistant's message
        indicates they're presenting options to choose from.

        Args:
            state:             Current episode state.
            step_tool_results: Tool results from the current step.

        Returns:
            Set of product IDs from presentation tools, or empty set.
        """
        # Only trigger if a search/browse tool was used this turn
        presentation_tools = {"catalog.search", "catalog.rerank", "user.get_visit_history"}
        used_presentation = False
        presented_ids: set[str] = set()

        for tr in step_tool_results:
            tool_name = tr.get("name", "")
            if tool_name in presentation_tools and tr.get("error") is None:
                used_presentation = True
                result = tr.get("result")
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict):
                            for key in ("product_id", "id"):
                                if key in item and isinstance(item[key], str):
                                    presented_ids.add(item[key])
                                    break

        if not used_presentation or not presented_ids:
            return set()

        # Check if the assistant's message is presenting options
        # (asking user to choose, listing items, etc.)
        if state.conversation:
            last_msg = state.conversation[-1]
            if last_msg.get("role") == "assistant":
                msg_lower = last_msg.get("content", "").lower()
                presentation_cues = [
                    "which", "would you like", "here are", "i found",
                    "options", "choose", "select", "these", "results",
                    "do you want", "let me know", "prefer",
                    "following", "listed", "available",
                ]
                if any(cue in msg_lower for cue in presentation_cues):
                    return presented_ids

        return set()

    # ------------------------------------------------------------------
    # Internal: build goal and persona summaries for user simulator
    # ------------------------------------------------------------------

    def _build_goal_summary(
        self,
        problem: ProblemParams,
        env_id: str,
    ) -> str:
        """Build a human-readable goal summary for the user simulator LLM.

        This tells the LLM what the user is actually trying to accomplish,
        so it can stay on-topic and evaluate the assistant's actions.

        Args:
            problem: ProblemParams from the environment generator.
            env_id:  Environment identifier.

        Returns:
            Goal summary string.
        """
        extra = problem.extra or {}

        if env_id == "CART":
            item_details = extra.get("item_details", [])
            parts: list[str] = []
            for d in item_details:
                part = d.get("title", "unknown item")
                if d.get("qty", 1) > 1:
                    part += f" (qty: {d['qty']})"
                if d.get("variant_desc"):
                    part += f" [{d['variant_desc']}]"
                parts.append(part)
            return "Add these items to your cart: " + ", ".join(parts)

        elif env_id == "PD":
            constraints = problem.constraints or []
            constraint_parts = []
            for c in constraints:
                attr = c.get("attr", "")
                op = c.get("op", "")
                value = c.get("value", "")
                if attr == "price" and op == "lte":
                    constraint_parts.append(f"under ${value}")
                elif attr == "brand" and op == "eq":
                    constraint_parts.append(f"brand: {value}")
                elif attr == "rating" and op == "gte":
                    constraint_parts.append(f"rated {value}+")
                else:
                    constraint_parts.append(f"{attr}: {value}")
            if problem.target_product_ids:
                target = self._products_by_id.get(problem.target_product_ids[0])
                cat = target.cat if target else "products"
                return f"Find {cat} matching: " + ", ".join(constraint_parts)
            return "Find products matching: " + ", ".join(constraint_parts)

        elif env_id == "SUB":
            if problem.target_product_ids:
                target = self._products_by_id.get(problem.target_product_ids[0])
                name = target.title if target else "your desired product"
                return f"Find a substitute for '{name}' which is out of stock."
            return "Find a substitute for an out-of-stock product."

        elif env_id == "RETURN":
            product_title = extra.get("target_product_title", "an item")
            target_order_id = extra.get("target_order_id", "unknown")
            target_line_id = extra.get("target_line_id", "unknown")
            window_days = extra.get("window_days", 30)
            t_days = extra.get("t_days", 0)
            replacement_required = extra.get("replacement_required", False)
            reason = extra.get("reason", "defective")

            parts = [
                f"Return '{product_title}' from order {target_order_id}",
                f"(line {target_line_id}).",
                f"Purchased {t_days} days ago.",
                f"Return window: {window_days} days.",
                f"Reason: {reason}.",
            ]
            if replacement_required:
                constraints = problem.constraints or []
                if constraints:
                    from ecom_rlve.simulator.llm_backend import _format_constraints_list
                    constraint_desc = _format_constraints_list(constraints)
                    parts.append(f"Replacement needed matching: {constraint_desc}.")
                else:
                    parts.append("Replacement needed.")
            return " ".join(parts)

        elif env_id == "STATUS":
            return "Find out the status of a specific order."

        elif env_id == "POLICY":
            question = extra.get("question_text", "a policy question")
            return f"Get an answer to: {question}"

        elif env_id == "BUNDLE":
            categories = extra.get("required_categories", [])
            budget = extra.get("budget", "")
            summary = "Buy one item from each category: " + ", ".join(categories)
            if budget:
                summary += f". Budget: ${budget}"
            return summary

        elif env_id == "JOURNEY":
            return "Complete multiple shopping tasks in one conversation."

        return "Complete your shopping task."

    def _build_persona_summary(self, persona: PersonaWeights) -> str:
        """Build a human-readable persona summary for the user simulator LLM.

        Args:
            persona: Persona preference weights.

        Returns:
            Persona summary string.
        """
        parts: list[str] = []
        w = persona.as_dict()
        # Only mention dimensions the persona cares about (weight > 0.25)
        if w.get("w_price", 0) > 0.25:
            parts.append("price-sensitive")
        if w.get("w_rating", 0) > 0.25:
            parts.append("quality-focused (rating matters)")
        if w.get("w_ship", 0) > 0.25:
            parts.append("wants fast shipping")
        if w.get("w_brand", 0) > 0.25:
            parts.append("brand-loyal")
        if w.get("w_similarity", 0) > 0.25:
            parts.append("wants something similar to original")
        return ", ".join(parts) if parts else "no strong preferences"

    # ------------------------------------------------------------------
    # Internal: compute progress info for user simulator feedback
    # ------------------------------------------------------------------

    def _compute_progress_info(
        self,
        state: EpisodeState,
        step_tool_results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Compute progress info so the user simulator can give ground-truth feedback.

        For CART: compares the current cart state against the required items
        (including variant requirements) and returns specific issues.

        For other envs: returns None (no mid-dialogue feedback yet).

        Args:
            state:             Current episode state.
            step_tool_results: Tool results from the current step.

        Returns:
            Dict with 'satisfaction' and optionally 'cart_issues', or None.
        """
        if state.env_id == "RETURN":
            return self._compute_return_progress_info(state, step_tool_results)

        if state.env_id != "CART":
            return None

        if state.hidden_goal is None:
            return None

        required_items: dict[str, int] = state.hidden_goal.extra.get("required_items", {})
        variant_reqs: dict[str, str | None] = state.hidden_goal.extra.get("variant_reqs", {})
        if not required_items:
            return None

        # ---------------------------------------------------------
        # Candidate confirmation: detect when the agent presents
        # product options (from search/visit history) and the user
        # needs to confirm which item(s) to add.
        # ---------------------------------------------------------
        # Check if search or visit history tools were called this step
        # AND the agent's message is presenting options to the user.
        presented_ids = self._extract_presented_product_ids(
            state, step_tool_results
        )
        if presented_ids:
            # Find which presented IDs match targets
            target_ids = set(required_items.keys())
            matching = presented_ids & target_ids
            non_matching = presented_ids - target_ids

            if matching:
                # Build confirmation info for the user sim
                confirmations: list[dict[str, str]] = []
                for pid in matching:
                    product = state.products_by_id.get(pid)
                    if product:
                        confirmations.append({
                            "product_id": pid,
                            "title": product.title,
                            "qty": required_items.get(pid, 1),
                        })
                return {
                    "satisfaction": len(matching) / len(target_ids),
                    "cart_candidates": confirmations,
                }

        # Only provide cart feedback if the agent actually touched the cart
        cart_touched = any(
            tr.get("name", "").startswith("cart.")
            for tr in step_tool_results
        )
        if not cart_touched:
            return None

        # Build cart state: (product_id, variant_id) -> qty
        cart_pv: dict[tuple[str, str | None], int] = {}
        for line in state.cart.lines:
            key = (line.product_id, line.variant_id)
            cart_pv[key] = cart_pv.get(key, 0) + line.qty

        # Also build product-level totals for simpler checks
        cart_by_pid: dict[str, int] = {}
        cart_variants_by_pid: dict[str, list[str | None]] = {}
        for (pid, vid), qty in cart_pv.items():
            cart_by_pid[pid] = cart_by_pid.get(pid, 0) + qty
            cart_variants_by_pid.setdefault(pid, []).append(vid)

        correct = 0
        total_required = len(required_items)
        issues: list[str] = []

        for pid, req_qty in required_items.items():
            product = state.products_by_id.get(pid)
            title = product.title if product else pid
            req_vid = variant_reqs.get(pid)

            if pid not in cart_by_pid:
                issues.append(f"'{title}' hasn't been added yet")
                continue

            # Check variant match
            if req_vid is not None:
                # Variant is required — check (pid, vid) pair
                matched_qty = cart_pv.get((pid, req_vid), 0)
                wrong_variants = [
                    vid for vid in cart_variants_by_pid.get(pid, [])
                    if vid != req_vid and vid is not None
                ]

                if matched_qty == 0:
                    # Product is in cart but with wrong variant
                    # Look up the variant descriptions for helpful feedback
                    wrong_descs = []
                    for wv in wrong_variants:
                        # Find variant in episode state
                        for v_list in state.variants_by_product.get(pid, []):
                            if hasattr(v_list, 'variant_id') and v_list.variant_id == wv:
                                parts = []
                                if v_list.color:
                                    parts.append(v_list.color)
                                if v_list.size:
                                    parts.append(v_list.size)
                                for k, val in v_list.attrs.items():
                                    parts.append(f"{k}: {val}")
                                wrong_descs.append(", ".join(parts) or wv)
                    if wrong_descs:
                        issues.append(
                            f"'{title}' is in the cart but with the wrong variant "
                            f"({', '.join(wrong_descs)}) — I need a different one"
                        )
                    else:
                        issues.append(
                            f"'{title}' is in the cart but with the wrong variant"
                        )
                elif matched_qty < req_qty:
                    issues.append(
                        f"'{title}' has quantity {matched_qty} but I need {req_qty}"
                    )
                elif matched_qty > req_qty:
                    issues.append(
                        f"'{title}' has quantity {matched_qty} but I only need {req_qty}"
                    )
                else:
                    correct += 1

                # Also flag wrong-variant lines as extra
                if wrong_variants:
                    for wv in wrong_variants:
                        wv_qty = cart_pv.get((pid, wv), 0)
                        if wv_qty > 0 and matched_qty > 0:
                            # Already reported the variant mismatch above
                            pass
            else:
                # No variant required — just check product-level qty
                cart_qty = cart_by_pid.get(pid, 0)
                if cart_qty == req_qty:
                    correct += 1
                elif cart_qty < req_qty:
                    issues.append(
                        f"'{title}' has quantity {cart_qty} but I need {req_qty}"
                    )
                else:
                    issues.append(
                        f"'{title}' has quantity {cart_qty} but I only need {req_qty}"
                    )

        # Check for extra items not in the required list
        for pid in cart_by_pid:
            if pid not in required_items:
                product = state.products_by_id.get(pid)
                title = product.title if product else pid
                issues.append(f"'{title}' shouldn't be in the cart")

        satisfaction = correct / total_required if total_required > 0 else 0.0

        return {
            "satisfaction": satisfaction,
            "cart_issues": issues,
        }

    # ------------------------------------------------------------------
    # Internal: compute progress info for RETURN env (disambiguation)
    # ------------------------------------------------------------------

    def _compute_return_progress_info(
        self,
        state: EpisodeState,
        step_tool_results: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Compute progress info for RETURN env so the user sim can disambiguate.

        When the agent calls order.list and presents order options, the user
        sim needs to know which order/line is the target so it can respond
        correctly during disambiguation.

        When the agent shows order items asking "which order?", we detect this
        and provide return_disambiguation info so the LLM user sim can pick
        the correct order.

        Args:
            state:             Current episode state.
            step_tool_results: Tool results from the current step.

        Returns:
            Dict with disambiguation hints, or None if no order tools were called.
        """
        if state.hidden_goal is None:
            return None

        extra = state.hidden_goal.extra or {}
        target_order_id = extra.get("target_order_id", "")
        target_line_id = extra.get("target_line_id", "")
        target_title = extra.get("target_product_title", "")

        # Check if the agent called any order tools this step
        order_touched = any(
            tr.get("name", "").startswith("order.")
            for tr in step_tool_results
        )

        # Check if agent already initiated a return
        return_initiated = bool(state.initiated_returns)

        if not order_touched and not return_initiated:
            return None

        # Build disambiguation info for the user sim
        return {
            "return_disambiguation": {
                "target_order_id": target_order_id,
                "target_line_id": target_line_id,
                "target_product_title": target_title,
            },
            "return_initiated": return_initiated,
            "satisfaction": 0.5 if not return_initiated else 1.0,
        }

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
        if env_id == "CART":
            # Build item_list using short product-type from title,
            # NOT full titles or raw categories.
            # Real customers say "phone case by Anker" not "Nurbo Cute Dragonfly..."
            from ecom_rlve.simulator.llm_backend import extract_product_type

            item_details = extra.get("item_details", [])
            item_parts: list[str] = []
            variant_parts: list[str] = []
            qty_parts: list[str] = []
            for d in item_details:
                # Short product-type derived from title, plus brand
                brand = d.get("brand", "")
                product_type = extract_product_type(d.get("title", "an item"), brand)
                if brand and brand.lower() not in ("unknown", "generic", "unbranded", ""):
                    short_desc = f"{product_type} by {brand}"
                else:
                    short_desc = product_type
                part = short_desc
                if d.get("qty", 1) > 1:
                    part += f" (x{d['qty']})"
                item_parts.append(part)
                if d.get("variant_desc"):
                    variant_parts.append(f"{short_desc}: {d['variant_desc']}")
                if d.get("qty", 1) > 1:
                    qty_parts.append(f"{short_desc}: qty {d['qty']}")
            goal_params["item_list"] = ", ".join(item_parts)
            if variant_parts:
                goal_params["variant_details"] = "; ".join(variant_parts)
            if qty_parts:
                goal_params["quantity_details"] = "; ".join(qty_parts)
        elif env_id == "RETURN":
            goal_params["product_desc"] = extra.get("target_product_title", "an item")
            goal_params["reason"] = extra.get("reason", "defective")
            # order_ref is available for clarification but NOT in initial message
            goal_params["order_ref"] = extra.get("target_order_id", "")
            if extra.get("replacement_required"):
                goal_params["replacement_req"] = "I'd also like a replacement."
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
