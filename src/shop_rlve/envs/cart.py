"""E_CART -- Cart Building environment.

Spec Section 5 / E3:
    Difficulty:
        n_items(d) = 1 + floor(d/3)  (cap at 5)
        p_var(d)   = sigmoid((d-2)/1.5)
        p_qty(d)   = min(0.5, 0.1*d)

    Generator:
        1. Sample n_items target products.
        2. For each, optionally require specific variant with prob p_var(d).
        3. For each, set required quantity:
            q_j = 1 with prob (1 - p_qty(d)), else q_j ~ U{2,4}.

    Reward:
        Unit-level F1:
            prec = U_match / (U_cart + 1e-9)
            rec  = U_match / (U_req  + 1e-9)
            F1   = 2*prec*rec / (prec + rec + 1e-9)
        r_task = 2*F1 - 1

    IsCorrect:
        F1 == 1.0
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from shop_rlve.data.schema import DENIED_CATEGORIES, Product, avail
from shop_rlve.difficulty.mapping import map_difficulty, sigmoid
from shop_rlve.rewards.verifiers import verify_cart
from shop_rlve.simulator.templates import render_template

from shop_rlve.envs.base import (
    BaseEnvironment,
    EpisodeResult,
    ProblemParams,
    register_env,
)


@register_env
class CartBuildingEnv(BaseEnvironment):
    """E_CART: Cart Building -- add correct items/variants/qty to cart."""

    ENV_ID = "CART"

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
        """Sample n_items target products with optional variants and quantities.

        Spec Section 5 / E3 generator:
            n_items(d) = min(5, 1 + floor(d/3))
            p_var(d) = sigmoid((d-2)/1.5)
            p_qty(d) = min(0.5, 0.1*d)
        """
        dp = map_difficulty(difficulty)
        rng = random.Random(seed)

        n_items = min(5, 1 + math.floor(difficulty / 3))
        p_var = sigmoid((difficulty - 2) / 1.5)
        p_qty = min(0.5, 0.1 * difficulty)

        eligible = [p for p in catalog if p.cat not in DENIED_CATEGORIES and avail(p)]
        if len(eligible) < n_items:
            eligible = [p for p in catalog if p.cat not in DENIED_CATEGORIES]
        if len(eligible) < n_items:
            eligible = catalog

        targets = rng.sample(eligible, min(n_items, len(eligible)))

        # Build required items: {product_id -> qty}
        # Also track variant requirements
        required_items: dict[str, int] = {}
        variant_reqs: dict[str, str | None] = {}
        item_details: list[dict[str, Any]] = []

        for product in targets:
            # Quantity
            if rng.random() < p_qty:
                qty = rng.randint(2, 4)
            else:
                qty = 1

            # Variant
            variant_id: str | None = None
            variant_desc: str | None = None
            if rng.random() < p_var and product.attrs:
                # Pick a variant-like attribute
                for attr_name in ("color", "size"):
                    val = product.attrs.get(attr_name)
                    if val is not None:
                        variant_id = f"{product.id}_v_{val}"
                        variant_desc = f"{attr_name}: {val}"
                        break

            required_items[product.id] = qty
            variant_reqs[product.id] = variant_id
            item_details.append({
                "product_id": product.id,
                "title": product.title,
                "qty": qty,
                "variant_id": variant_id,
                "variant_desc": variant_desc,
            })

        return ProblemParams(
            env_id=self.ENV_ID,
            difficulty=difficulty,
            seed=seed,
            target_product_ids=[p.id for p in targets],
            constraints=[],
            extra={
                "n_items": n_items,
                "p_var": p_var,
                "p_qty": p_qty,
                "T_max": dp.T_max_val,
                "p_missing": dp.p_missing_val,
                "p_noise": dp.p_noise_val,
                "required_items": required_items,
                "variant_reqs": variant_reqs,
                "item_details": item_details,
            },
        )

    # ------------------------------------------------------------------
    # I : Input generator
    # ------------------------------------------------------------------

    def generate_input(self, params: ProblemParams) -> str:
        """Render cart building request."""
        details: list[dict[str, Any]] = params.extra["item_details"]

        # Build item list string
        item_parts: list[str] = []
        for d in details:
            part = d["title"]
            if d["qty"] > 1:
                part += f" (x{d['qty']})"
            item_parts.append(part)
        item_list = ", ".join(item_parts)

        # Build variant and quantity detail strings
        variant_parts: list[str] = []
        qty_parts: list[str] = []
        for d in details:
            if d["variant_desc"]:
                variant_parts.append(f"{d['title']}: {d['variant_desc']}")
            if d["qty"] > 1:
                qty_parts.append(f"{d['title']}: qty {d['qty']}")

        template_params: dict[str, Any] = {"item_list": item_list}
        if variant_parts:
            template_params["variant_details"] = "; ".join(variant_parts)
        if qty_parts:
            template_params["quantity_details"] = "; ".join(qty_parts)

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
        """Compute r_task per Spec Section 5 / E3.

        r_task = 2*F1 - 1 (unit-level)
        IsCorrect = F1 == 1.0
        """
        required_items: dict[str, int] = params.extra["required_items"]
        cart_lines: list[dict[str, Any]] = episode_state.get("cart_lines", [])
        products_by_id: dict[str, Product] = episode_state.get("products_by_id", {})

        r_task, is_correct = verify_cart(
            required_items=required_items,
            cart_lines=cart_lines,
            products_by_id=products_by_id,
        )

        return EpisodeResult(
            r_task=r_task,
            is_correct=is_correct,
            details={
                "n_items": params.extra.get("n_items"),
                "required_items": required_items,
                "cart_lines_count": len(cart_lines),
            },
        )
