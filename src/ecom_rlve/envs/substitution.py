"""E_SUB -- OOS Substitution environment.

Spec Section 5 / E2:
    Generator:
        Sample original product, set OOS, add compatibility constraint.

    Difficulty:
        lambda_sim(d) = clip(0.4 + 0.05*d, 0.4, 0.8)

    Reward:
        For each recommended p_i:
            rel_i = lambda_sim(d) * sim_i + (1 - lambda_sim(d)) * s_i
        r_rank = 2*nDCG(rel) - 1
        r_oos  = -clip(oos_rate, 0, 1)
        r_task = clip(0.80*r_rank + 0.20*r_oos, -1, 1)

    IsCorrect:
        r_task >= 0.95
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from ecom_rlve.data.schema import DENIED_CATEGORIES, Product, avail
from ecom_rlve.difficulty.mapping import map_difficulty
from ecom_rlve.rewards.verifiers import verify_substitution
from ecom_rlve.simulator.persona import sample_persona_weights
from ecom_rlve.simulator.templates import render_template

from ecom_rlve.envs.base import (
    BaseEnvironment,
    EpisodeResult,
    ProblemParams,
    register_env,
)


@register_env
class SubstitutionEnv(BaseEnvironment):
    """E_SUB: OOS Substitution -- find alternative for out-of-stock product."""

    ENV_ID = "SUB"

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
        """Sample original product, set OOS, generate constraints.

        Spec Section 5 / E2 generator:
            1. Sample p0, set stock_qty(p0) = 0 for this episode.
            2. Generate constraints C including compatibility.
            3. User message: "{p0} is out of stock -- find alternative".
        """
        dp = map_difficulty(difficulty)
        rng = random.Random(seed)

        eligible = [p for p in catalog if p.cat not in DENIED_CATEGORIES]
        if not eligible:
            eligible = catalog

        original = rng.choice(eligible)

        # Force original OOS for this episode (shallow copy)
        original_oos = original.model_copy(update={"stock_qty": 0})

        # Generate constraints from original product
        constraints = self.generate_constraints(original, dp.m_val, difficulty, seed)

        # Ensure at least one compatibility constraint (connector_type, size, etc.)
        compat_attrs = ["connector_type", "size", "material"]
        has_compat = any(c["attr"] in compat_attrs for c in constraints)
        if not has_compat:
            for attr in compat_attrs:
                val = original.attrs.get(attr)
                if val is not None:
                    constraints.append({"attr": attr, "op": "eq", "value": val})
                    break

        persona = sample_persona_weights(seed)

        return ProblemParams(
            env_id=self.ENV_ID,
            difficulty=difficulty,
            seed=seed,
            target_product_ids=[original.id],
            constraints=constraints,
            persona_weights=persona,
            extra={
                "original_product": original_oos,
                "k": dp.k_rec_val,
                "T_max": dp.T_max_val,
                "top_k": dp.top_k_val,
                "eps_rank": dp.eps_rank_val,
                "p_missing": dp.p_missing_val,
                "p_noise": dp.p_noise_val,
                "lambda_sim": float(np.clip(0.4 + 0.05 * difficulty, 0.4, 0.8)),
            },
        )

    # ------------------------------------------------------------------
    # I : Input generator
    # ------------------------------------------------------------------

    def generate_input(self, params: ProblemParams) -> str:
        """Render substitution request from problem parameters."""
        original: Product = params.extra["original_product"]
        template_params: dict[str, Any] = {
            "original_product": original.title,
        }
        # Optional constraint slots
        for c in params.constraints:
            if c["attr"] == "brand":
                template_params["brand_pref"] = f"Preferably {c['value']}."
            elif c["attr"] == "price" and c["op"] == "lte":
                template_params["price_range"] = f"Under ${c['value']}."
            elif c["attr"] == "color":
                template_params["color_pref"] = f"In {c['value']}."
            elif c["attr"] == "ship_days":
                template_params["ship_req"] = f"{c['value']}"

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
        """Compute r_task per Spec Section 5 / E2.

        r_task = clip(0.80*r_rank + 0.20*r_oos, -1, 1)
        IsCorrect = r_task >= 0.95
        """
        recommended_ids: list[str] = answer.get("recommended_product_ids", [])
        products_by_id: dict[str, Product] = episode_state.get("products_by_id", {})
        eval_pool: list[str] = episode_state.get("eval_pool", [])
        embeddings_fn = episode_state.get("embeddings_fn", lambda pid: None)
        k: int = params.extra.get("k", 3)
        original_id: str = params.target_product_ids[0]

        constraint_fns = self.build_constraint_fns(params.constraints)

        r_task, is_correct = verify_substitution(
            recommended_ids=recommended_ids,
            constraints=constraint_fns,
            original_product_id=original_id,
            products_by_id=products_by_id,
            eval_pool=eval_pool,
            k=k,
            difficulty=params.difficulty,
            embeddings_fn=embeddings_fn,
        )

        return EpisodeResult(
            r_task=r_task,
            is_correct=is_correct,
            details={
                "k": k,
                "lambda_sim": params.extra.get("lambda_sim"),
                "original_id": original_id,
                "n_recommended": len(recommended_ids),
            },
        )
