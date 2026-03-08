"""E_PD -- Product Discovery (recommendation) environment.

Spec Section 5 / E1:
    Generator:
        Sample target product, m constraints, build user message
        with p_missing slot omission.

    Difficulty overrides:
        m = m(d),  k = k_rec(d),  T_max = T_max(d)
        top_k = top_k(d),  eps_rank = eps_rank(d)

    Reward:
        r_rank = 2*nDCG(rel=u(p)) - 1
        alpha(d) = 4 + floor(d/4)
        r_cons = 2*(s_best)^alpha(d) - 1
        r_oos  = -clip(oos_rate, 0, 1)
        r_task = clip(0.55*r_rank + 0.35*r_cons + 0.10*r_oos, -1, 1)

    IsCorrect:
        r_task >= 0.95
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from shop_rlve.data.schema import DENIED_CATEGORIES, Product, avail
from shop_rlve.difficulty.mapping import map_difficulty
from shop_rlve.rewards.verifiers import verify_product_discovery
from shop_rlve.simulator.persona import PersonaWeights, sample_persona_weights
from shop_rlve.simulator.templates import render_template

from shop_rlve.envs.base import (
    BaseEnvironment,
    EpisodeResult,
    ProblemParams,
    register_env,
)


@register_env
class ProductDiscoveryEnv(BaseEnvironment):
    """E_PD: Product Discovery -- recommend products matching constraints."""

    ENV_ID = "PD"

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
        """Sample a target product and *m(d)* constraints at difficulty *d*.

        Spec Section 5 / E1 generator:
            1. Sample target product p* ~ Uniform(catalog_filtered).
            2. Sample m constraint attributes.
            3. Generate predicates so that C_j(p*) = 1.
            4. User utterance includes each constraint with prob (1 - p_missing).
        """
        dp = map_difficulty(difficulty)
        rng = random.Random(seed)

        # Filter out denied categories
        eligible = [p for p in catalog if p.cat not in DENIED_CATEGORIES and avail(p)]
        if not eligible:
            eligible = [p for p in catalog if p.cat not in DENIED_CATEGORIES]
        if not eligible:
            eligible = catalog

        target = rng.choice(eligible)
        constraints = self.generate_constraints(target, dp.m_val, difficulty, seed)
        persona = sample_persona_weights(seed)

        return ProblemParams(
            env_id=self.ENV_ID,
            difficulty=difficulty,
            seed=seed,
            target_product_ids=[target.id],
            constraints=constraints,
            persona_weights=persona,
            extra={
                "k": dp.k_rec_val,
                "T_max": dp.T_max_val,
                "top_k": dp.top_k_val,
                "eps_rank": dp.eps_rank_val,
                "p_missing": dp.p_missing_val,
                "p_noise": dp.p_noise_val,
                "target_product": target,
            },
        )

    # ------------------------------------------------------------------
    # I : Input generator
    # ------------------------------------------------------------------

    def generate_input(self, params: ProblemParams) -> str:
        """Render the initial user utterance from problem parameters."""
        target: Product = params.extra["target_product"]
        template_params: dict[str, Any] = {
            "category": target.cat,
            "price_max": f"{target.price * 1.2:.0f}",
        }
        # Optional slots derived from constraints
        for c in params.constraints:
            attr = c["attr"]
            value = c["value"]
            if attr == "brand":
                template_params["brand_pref"] = f"I prefer {value}."
            elif attr == "color":
                template_params["color_pref"] = f"I want it in {value}."
            elif attr == "rating":
                template_params["rating_req"] = f"Rated at least {value} stars."
            elif attr == "ship_days":
                template_params["ship_req"] = f"{value}"

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
        """Compute r_task per Spec Section 5 / E1.

        r_task = clip(0.55*r_rank + 0.35*r_cons + 0.10*r_oos, -1, 1)
        IsCorrect = r_task >= 0.95
        """
        recommended_ids: list[str] = answer.get("recommended_product_ids", [])
        products_by_id: dict[str, Product] = episode_state.get("products_by_id", {})
        eval_pool: list[str] = episode_state.get("eval_pool", [])
        k: int = params.extra.get("k", 3)

        constraint_fns = self.build_constraint_fns(params.constraints)
        persona: PersonaWeights = params.persona_weights

        pool_products = [products_by_id[pid] for pid in eval_pool if pid in products_by_id]
        p_low, p_high = self.price_range(pool_products) if pool_products else (0.0, 1.0)

        # Determine brand_pref from constraints
        brand_pref: str | None = None
        for c in params.constraints:
            if c["attr"] == "brand":
                brand_pref = c["value"]
                break

        r_task, is_correct = verify_product_discovery(
            recommended_ids=recommended_ids,
            constraints=constraint_fns,
            persona_weights=persona,
            products_by_id=products_by_id,
            eval_pool=eval_pool,
            k=k,
            difficulty=params.difficulty,
            p_low=p_low,
            p_high=p_high,
            brand_pref=brand_pref,
        )

        return EpisodeResult(
            r_task=r_task,
            is_correct=is_correct,
            details={
                "k": k,
                "n_recommended": len(recommended_ids),
                "n_constraints": len(params.constraints),
            },
        )
