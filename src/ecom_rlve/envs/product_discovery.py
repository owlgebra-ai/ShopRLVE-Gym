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

from ecom_rlve.data.schema import DENIED_CATEGORIES, Product, avail
from ecom_rlve.difficulty.mapping import map_difficulty
from ecom_rlve.rewards.verifiers import verify_product_discovery
from ecom_rlve.simulator.persona import (
    PersonaWeights,
    sample_aligned_persona,
    sample_persona_weights,
)
from ecom_rlve.simulator.llm_backend import (
    verbalize_constraints,
    verbalize_with_strategic_omission,
)
from ecom_rlve.simulator.templates import render_template

from ecom_rlve.envs.base import (
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

        Fix 4 (price reconciliation):
            Price is ALWAYS included as constraint #0, derived from the
            target product's actual price with difficulty-controlled slack.
            The remaining m(d)-1 constraints are sampled from non-price
            attributes via generate_constraints(exclude_attrs={"price"}).

        Fix 2 (aligned persona):
            Persona weights are sampled with Dirichlet alpha boosted
            for dimensions that appear in the constraint set.
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

        # Fix 4: Always include price as constraint #0
        delta_max = 0.5 * math.exp(-difficulty / 5.0)
        delta = rng.uniform(0, delta_max)
        price_bound = round(target.price * (1.0 + delta), 2)
        price_constraint = {"attr": "price", "op": "lte", "value": price_bound}

        # Sample remaining m(d)-1 constraints, excluding price
        remaining = max(0, dp.m_val - 1)
        other_constraints = self.generate_constraints(
            target, remaining, difficulty, seed,
            exclude_attrs={"price"},
        )
        constraints = [price_constraint] + other_constraints

        # Fix 2: Sample persona aligned with constrained dimensions
        persona = sample_aligned_persona(constraints, seed)

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
                "price_bound": price_bound,
            },
        )

    # ------------------------------------------------------------------
    # I : Input generator
    # ------------------------------------------------------------------

    # -- Expanded attribute-to-slot mapping (Fix 3) --------------------------
    # Covers ALL attributes in the allowlist so the verifier never penalizes
    # the agent for constraints that were not communicated to it.
    _ATTR_TO_SLOT: dict[str, tuple[str, str]] = {
        "brand":              ("brand_pref",     "I prefer {value}."),
        "color":              ("color_pref",     "I want it in {value}."),
        "rating":             ("rating_req",     "Rated at least {value} stars."),
        "ship_days":          ("ship_req",       "Ships within {value} days."),
        "material":           ("material_pref",  "Made of {value}."),
        "connector_type":     ("connector_pref", "Must have {value} connectivity."),
        "size":               ("size_pref",      "Size {value}."),
        "wattage":            ("wattage_req",    "At least {value} watts."),
        "weight_lbs":         ("weight_req",     "Under {value} pounds."),
        "item_form":          ("form_pref",      "In {value} form."),
        "skin_type":          ("skin_pref",      "For {value} skin."),
        "finish_type":        ("finish_pref",    "With {value} finish."),
        "age_range":          ("age_pref",       "For {value}."),
        "screen_size_inches": ("screen_req",     "{value}-inch screen."),
        "rating_count":       ("popularity_req", "At least {value} reviews."),
        "store":              ("store_pref",     "From {value} store."),
        "cat":                ("category",       "{value}"),
    }

    def generate_input(self, params: ProblemParams) -> str:
        """Render the initial user utterance from problem parameters.

        Fix 1 (LLM-verbalized constraints):
            First attempts to use the Ollama LLM backend to generate a
            natural, varied user message covering ALL constraint types.
            Falls back to expanded template-based generation if the LLM
            is unavailable.

        Fix 4 (price reconciliation):
            price_max in the template is derived from the actual formal
            price constraint bound, not a separate hardcoded 1.2x markup.
        """
        target: Product = params.extra["target_product"]
        p_missing = params.extra.get("p_missing", 0.0)

        # --- Fix 1 + Fix 5: Try LLM-based verbalization first ---------------
        if p_missing > 0.05:
            # Use strategic omission (LLM decides what to mention)
            text, mentioned, omitted = verbalize_with_strategic_omission(
                category=target.cat,
                constraints=params.constraints,
                seed=params.seed + 1,
            )
            if text is not None:
                # Track what was mentioned/omitted for the dialogue simulator
                params.extra["mentioned_attrs"] = mentioned
                params.extra["omitted_attrs"] = omitted
                return text
        else:
            # No omission needed — verbalize all constraints
            text = verbalize_constraints(
                category=target.cat,
                constraints=params.constraints,
                seed=params.seed + 1,
            )
            if text is not None:
                params.extra["mentioned_attrs"] = {
                    c["attr"] for c in params.constraints
                }
                params.extra["omitted_attrs"] = set()
                return text

        # --- Fallback: expanded template-based generation -------------------
        # Fix 4: Use the formal price bound, not hardcoded 1.2x
        price_bound = params.extra.get("price_bound", target.price * 1.2)
        template_params: dict[str, Any] = {
            "category": target.cat,
            "price_max": f"{price_bound:.0f}",
        }

        # Fix 3: Map ALL constraint attributes to template slots
        for c in params.constraints:
            attr = c["attr"]
            value = c["value"]
            if attr == "price":
                continue  # already handled via price_max
            slot_info = self._ATTR_TO_SLOT.get(attr)
            if slot_info is not None:
                slot_name, slot_template = slot_info
                template_params[slot_name] = slot_template.format(value=value)

        params.extra["mentioned_attrs"] = set(template_params.keys())
        params.extra["omitted_attrs"] = set()

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
