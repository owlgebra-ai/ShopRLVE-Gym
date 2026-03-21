"""E_JOURNEY -- Multi-intent Composite environment.

Spec Section 5 / E8:
    Difficulty:
        L_int(d) = min(5, 2 + floor(d/4))   subgoals
        p_switch(d) = 0.6 * sigmoid((d-5)/2) context switch probability

    Generator:
        1. Sample L_int(d) subgoals, each from one of the atomic envs.
        2. Generate shared world state (catalog, orders, cart) once.
        3. Create user utterances introducing subgoals sequentially.

    Reward:
        Subtask rewards r_1..r_L from the same verifiers as atomic envs.
        r_task = clip(mean(subtask_rewards), -1, 1)

    IsCorrect:
        all r_j >= 0.95
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from ecom_rlve.data.schema import Product
from ecom_rlve.difficulty.mapping import map_difficulty
from ecom_rlve.rewards.verifiers import verify_journey
from ecom_rlve.simulator.templates import render_template

from ecom_rlve.envs.base import (
    ENV_REGISTRY,
    BaseEnvironment,
    EpisodeResult,
    ProblemParams,
    register_env,
)


# Env IDs eligible for subgoals (exclude JOURNEY itself to avoid recursion)
_SUBGOAL_ENV_IDS: list[str] = ["PD", "SUB", "CART", "RETURN", "STATUS", "POLICY", "BUNDLE"]


@register_env
class JourneyEnv(BaseEnvironment):
    """E_JOURNEY: Multi-intent Composite -- chained tasks in one conversation."""

    ENV_ID = "JOURNEY"

    # ------------------------------------------------------------------
    # P_d : Problem generator
    # ------------------------------------------------------------------

    def generate_problem(
        self,
        difficulty: int,
        catalog: list[Product],
        seed: int,
        **kwargs: Any,
    ) -> ProblemParams:
        """Sample L_int(d) subgoals from atomic environments.

        Spec Section 5 / E8:
            L_int(d) = min(5, 2 + floor(d/4))
        """
        dp = map_difficulty(difficulty)
        rng = random.Random(seed)

        n_intents = min(5, 2 + math.floor(difficulty / 4))

        # Sample subgoal env IDs
        available = [eid for eid in _SUBGOAL_ENV_IDS if eid in ENV_REGISTRY]
        if not available:
            available = _SUBGOAL_ENV_IDS  # fallback; envs may be registered later

        subgoal_env_ids = rng.choices(available, k=n_intents)

        # Generate subproblems
        subgoal_params: list[ProblemParams] = []
        for i, env_id in enumerate(subgoal_env_ids):
            env_cls = ENV_REGISTRY.get(env_id)
            if env_cls is not None:
                env_instance = env_cls()
                sub_seed = seed + 1000 * (i + 1)
                sub_params = env_instance.generate_problem(
                    difficulty=difficulty,
                    catalog=catalog,
                    seed=sub_seed,
                    **kwargs,
                )
                subgoal_params.append(sub_params)

        # Build first task description for the initial message
        first_task_desc = "I need help with something"
        if subgoal_params:
            first = subgoal_params[0]
            first_task_desc = _describe_subgoal(first)

        second_hint = ""
        if len(subgoal_params) > 1:
            second_hint = "I also have another request after this."

        return ProblemParams(
            env_id=self.ENV_ID,
            difficulty=difficulty,
            seed=seed,
            target_product_ids=[],
            constraints=[],
            extra={
                "T_max": dp.T_max_val,
                "p_missing": dp.p_missing_val,
                "p_noise": dp.p_noise_val,
                "p_switch": dp.p_switch_val,
                "n_intents": n_intents,
                "subgoal_env_ids": subgoal_env_ids,
                "subgoal_params": subgoal_params,
                "first_task_desc": first_task_desc,
                "second_hint": second_hint,
            },
        )

    # ------------------------------------------------------------------
    # I : Input generator
    # ------------------------------------------------------------------

    def generate_input(self, params: ProblemParams) -> str:
        """Render journey initial message introducing first subgoal."""
        template_params: dict[str, Any] = {
            "first_task": params.extra.get("first_task_desc", "I need help"),
        }
        if params.extra.get("second_hint"):
            template_params["second_hint"] = params.extra["second_hint"]

        return render_template(
            env_id=self.ENV_ID,
            params=template_params,
            p_missing=params.extra.get("p_missing", 0.0),
            p_noise=params.extra.get("p_noise", 0.0),
            seed=params.seed + 1,
        )

    # ------------------------------------------------------------------
    # R : Verifier
    # ------------------------------------------------------------------

    def verify(
        self,
        answer: dict[str, Any],
        params: ProblemParams,
        episode_state: dict[str, Any],
    ) -> EpisodeResult:
        """Compute r_task per Spec Section 5 / E8.

        r_task = clip(mean(subtask_rewards), -1, 1)
        IsCorrect = all r_j >= 0.95
        """
        subtask_rewards: list[float] = episode_state.get("subtask_rewards", [])

        # If subtask_rewards are not pre-computed in episode_state,
        # compute them from subgoal_params and per-subgoal answers
        if not subtask_rewards:
            subgoal_params: list[ProblemParams] = params.extra.get("subgoal_params", [])
            subgoal_answers: list[dict[str, Any]] = answer.get("subgoal_answers", [])
            subgoal_states: list[dict[str, Any]] = episode_state.get("subgoal_states", [])

            for i, sub_params in enumerate(subgoal_params):
                env_cls = ENV_REGISTRY.get(sub_params.env_id)
                if env_cls is None:
                    subtask_rewards.append(-1.0)
                    continue

                env_instance = env_cls()
                sub_answer = subgoal_answers[i] if i < len(subgoal_answers) else {}
                sub_state = subgoal_states[i] if i < len(subgoal_states) else episode_state

                result = env_instance.verify(sub_answer, sub_params, sub_state)
                subtask_rewards.append(result.r_task)

        r_task, is_correct = verify_journey(subtask_rewards)

        return EpisodeResult(
            r_task=r_task,
            is_correct=is_correct,
            details={
                "n_intents": params.extra.get("n_intents", 0),
                "subtask_rewards": subtask_rewards,
                "subgoal_env_ids": params.extra.get("subgoal_env_ids", []),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _describe_subgoal(params: ProblemParams) -> str:
    """Build a short description of a subgoal for the initial message."""
    env_id = params.env_id
    if env_id == "PD":
        target = params.extra.get("target_product", None)
        cat = target.cat if target else "something"
        return f"I'm looking for a {cat} product"
    elif env_id == "SUB":
        orig = params.extra.get("original_product", None)
        title = orig.title if orig else "a product"
        return f"{title} is out of stock, I need an alternative"
    elif env_id == "CART":
        details = params.extra.get("item_details", [])
        if details:
            titles = [d["title"] for d in details[:2]]
            return f"I want to add {', '.join(titles)} to my cart"
        return "I need to add some items to my cart"
    elif env_id == "RETURN":
        title = params.extra.get("target_product_title", "an item")
        return f"I want to return {title}"
    elif env_id == "STATUS":
        oid = params.extra.get("target_order_id", "my order")
        return f"I want to check the status of {oid}"
    elif env_id == "POLICY":
        q = params.extra.get("question_text", "a policy question")
        return q
    elif env_id == "BUNDLE":
        name = params.extra.get("project_name", "a project")
        return f"I need items for {name}"
    return "I need help with something"
