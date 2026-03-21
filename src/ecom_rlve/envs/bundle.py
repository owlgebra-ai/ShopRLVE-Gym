"""E_BUNDLE -- Bundle / Project Planning environment.

Spec Section 5 / E7:
    Generator:
        |S_req| = 3 + floor(d/2)   (cap at 12) required categories
        Optionally add budget constraints using m(d).

    Reward:
        Category-level F1:
            S_hat = {cat(p) : p in L}
            TP = |S_hat intersect S_req|
            prec = TP / (|S_hat| + 1e-9)
            rec  = TP / (|S_req| + 1e-9)
            F1 = 2*prec*rec / (prec + rec + 1e-9)
        Budget penalty:
            cost = sum price(p) for p in L
            viol = max(0, (cost - B) / B)
            pen  = clip(viol, 0, 1)
        r_task = clip(2*F1 - 1 - pen, -1, 1)

    IsCorrect:
        F1 == 1.0 AND pen == 0
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from ecom_rlve.data.schema import DENIED_CATEGORIES, Product, avail
from ecom_rlve.difficulty.mapping import map_difficulty
from ecom_rlve.rewards.verifiers import verify_bundle
from ecom_rlve.simulator.templates import render_template

from ecom_rlve.envs.base import (
    BaseEnvironment,
    EpisodeResult,
    ProblemParams,
    register_env,
)


# ---------------------------------------------------------------------------
# Project templates for bundle generation
# ---------------------------------------------------------------------------

_PROJECT_TEMPLATES: list[dict[str, Any]] = [
    {
        "name": "Home baking starter kit",
        "categories": [
            "baking_supplies", "kitchen_utensils", "mixing_bowls", "measuring_cups",
            "baking_pans", "oven_mitts", "apron", "recipe_book", "flour",
            "sugar", "vanilla_extract", "baking_soda",
        ],
    },
    {
        "name": "Home office setup",
        "categories": [
            "desk", "chair", "monitor", "keyboard", "mouse", "headphones",
            "webcam", "desk_lamp", "cable_management", "mousepad",
            "monitor_stand", "surge_protector",
        ],
    },
    {
        "name": "Camping trip essentials",
        "categories": [
            "tent", "sleeping_bag", "backpack", "flashlight", "first_aid",
            "water_bottle", "camp_stove", "cooler", "camp_chair",
            "insect_repellent", "sunscreen", "fire_starter",
        ],
    },
    {
        "name": "DIY painting project",
        "categories": [
            "paint", "brushes", "rollers", "drop_cloth", "painters_tape",
            "primer", "sandpaper", "paint_tray", "ladder",
            "cleaning_supplies", "stir_sticks", "respirator",
        ],
    },
    {
        "name": "Fitness starter pack",
        "categories": [
            "yoga_mat", "dumbbells", "resistance_bands", "jump_rope",
            "water_bottle", "workout_shoes", "gym_bag", "towel",
            "protein_powder", "shaker_bottle", "fitness_tracker", "headphones",
        ],
    },
]


@register_env
class BundleEnv(BaseEnvironment):
    """E_BUNDLE: Bundle / Project Planning -- shopping list recommendation."""

    ENV_ID = "BUNDLE"

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
        """Sample a project template and required categories.

        Spec Section 5 / E7:
            |S_req| = min(12, 3 + floor(d/2))
        """
        dp = map_difficulty(difficulty)
        rng = random.Random(seed)

        n_categories = min(12, 3 + math.floor(difficulty / 2))

        # Select a project template
        template = rng.choice(_PROJECT_TEMPLATES)
        all_cats = template["categories"]
        n_cats = min(n_categories, len(all_cats))
        required_categories = rng.sample(all_cats, n_cats)

        # Optionally set a budget
        has_budget = difficulty >= 2 and rng.random() < 0.6
        budget: float | None = None
        if has_budget:
            eligible = [p for p in catalog if avail(p) and p.cat not in DENIED_CATEGORIES]
            if eligible:
                avg_price = sum(p.price for p in eligible) / len(eligible)
                budget = round(avg_price * n_cats * rng.uniform(0.8, 1.5), 2)

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
                "project_name": template["name"],
                "required_categories": required_categories,
                "n_categories": n_cats,
                "budget": budget,
            },
        )

    # ------------------------------------------------------------------
    # I : Input generator
    # ------------------------------------------------------------------

    def generate_input(self, params: ProblemParams) -> str:
        """Render bundle planning request."""
        required_cats: list[str] = params.extra["required_categories"]
        category_list = ", ".join(required_cats)
        budget = params.extra.get("budget")

        template_params: dict[str, Any] = {
            "category_list": category_list,
        }
        if budget is not None:
            template_params["budget"] = f"${budget:.2f}"

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
        """Compute r_task per Spec Section 5 / E7.

        r_task = clip(2*F1 - 1 - budget_penalty, -1, 1)
        IsCorrect = F1 == 1.0 AND penalty == 0
        """
        recommended_ids: list[str] = answer.get("recommended_product_ids", [])
        required_categories: list[str] = params.extra["required_categories"]
        products_by_id: dict[str, Product] = episode_state.get("products_by_id", {})
        budget: float | None = params.extra.get("budget")

        r_task, is_correct = verify_bundle(
            recommended_ids=recommended_ids,
            required_categories=required_categories,
            products_by_id=products_by_id,
            budget=budget,
        )

        return EpisodeResult(
            r_task=r_task,
            is_correct=is_correct,
            details={
                "required_categories": required_categories,
                "budget": budget,
                "n_recommended": len(recommended_ids),
                "n_categories_required": len(required_categories),
            },
        )
