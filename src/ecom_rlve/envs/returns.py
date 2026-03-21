"""E_RETURN -- Return + Replacement environment.

Spec Section 5 / E4:
    Difficulty:
        H_orders(d) orders, edge-case return window
        p_edge(d) = 0.7 * sigmoid((d-4)/1.5)

    Generator:
        1. Generate H_orders(d) orders with 1-5 line items each.
        2. Select target line (o*, l*) to be returned.
        3. Set purchase age near boundary with prob p_edge(d).
        4. Generate replacement constraints from l*.

    Reward:
        r_sel = 2 * 1[correct order AND line] - 1
        r_ret = 2 * 1[return.initiate called correctly] - 1
        r_rep = replacement quality (from E_SUB relevance), or 1 if not required
        r_task = clip(0.45*r_sel + 0.45*r_ret + 0.10*r_rep, -1, 1)

    IsCorrect:
        r_sel == 1 AND r_ret == 1 AND (r_rep >= 0.95 OR replacement not required)
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from ecom_rlve.data.schema import DENIED_CATEGORIES, Product, avail
from ecom_rlve.difficulty.mapping import map_difficulty, sigmoid
from ecom_rlve.rewards.verifiers import verify_return
from ecom_rlve.simulator.templates import render_template
from ecom_rlve.tools.orders import Order, generate_order_history
from ecom_rlve.tools.returns import RETURN_WINDOWS

from ecom_rlve.envs.base import (
    BaseEnvironment,
    EpisodeResult,
    ProblemParams,
    register_env,
)


@register_env
class ReturnEnv(BaseEnvironment):
    """E_RETURN: Return + Replacement."""

    ENV_ID = "RETURN"

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
        """Generate return episode: orders, target line, edge-case window.

        Spec Section 5 / E4:
            p_edge(d) = 0.7 * sigmoid((d-4)/1.5)
        """
        dp = map_difficulty(difficulty)
        rng = random.Random(seed)

        n_orders = dp.H_orders_val
        p_edge = 0.7 * sigmoid((difficulty - 4) / 1.5)

        # Generate order history
        eligible = [p for p in catalog if p.cat not in DENIED_CATEGORIES and avail(p)]
        if len(eligible) < 5:
            eligible = [p for p in catalog if p.cat not in DENIED_CATEGORIES]
        if not eligible:
            eligible = catalog

        orders = generate_order_history(
            products=eligible,
            n_orders=n_orders,
            seed=seed,
        )

        # Pick a target order that is DELIVERED (returnable)
        delivered = [o for o in orders if o.status == "DELIVERED" and o.lines]
        if not delivered:
            # Force at least one delivered order
            if orders and orders[0].lines:
                orders[0].status = "DELIVERED"
                delivered = [orders[0]]
            else:
                # Fallback: generate a minimal order
                p = rng.choice(eligible)
                from ecom_rlve.tools.orders import OrderLine

                fallback_order = Order(
                    order_id="ord_001",
                    order_date="2026-02-01",
                    status="DELIVERED",
                    lines=[
                        OrderLine(
                            line_id="ord_001_line_01",
                            product_id=p.id,
                            product_title=p.title,
                            qty=1,
                            unit_price=p.price,
                        )
                    ],
                )
                orders = [fallback_order]
                delivered = [fallback_order]

        target_order = rng.choice(delivered)
        target_line = rng.choice(target_order.lines)

        # Determine return window for the product category
        product_cat = "general"
        for p in eligible:
            if p.id == target_line.product_id:
                product_cat = p.cat
                break
        window_days = RETURN_WINDOWS.get(product_cat, RETURN_WINDOWS["default"])

        # Set purchase age (edge case or normal)
        if rng.random() < p_edge and window_days > 1:
            # Edge: near boundary — can be exactly at, one day within,
            # or one day past the return window
            t_days = window_days + rng.choice([-1, 0, 1])
        else:
            t_days = rng.randint(1, max(1, window_days - 5)) if window_days > 5 else 1

        # Decide whether replacement is required
        replacement_required = rng.random() < 0.5

        # Generate replacement constraints if needed
        replacement_constraints: list[dict[str, Any]] = []
        if replacement_required:
            for p in eligible:
                if p.id == target_line.product_id:
                    replacement_constraints = self.generate_constraints(
                        p, max(1, dp.m_val - 1), difficulty, seed + 100
                    )
                    break

        return ProblemParams(
            env_id=self.ENV_ID,
            difficulty=difficulty,
            seed=seed,
            target_product_ids=[target_line.product_id],
            constraints=replacement_constraints,
            extra={
                "T_max": dp.T_max_val,
                "p_missing": dp.p_missing_val,
                "p_noise": dp.p_noise_val,
                "orders": orders,
                "target_order_id": target_order.order_id,
                "target_line_id": target_line.line_id,
                "target_product_title": target_line.product_title,
                "window_days": window_days,
                "t_days": t_days,
                "p_edge": p_edge,
                "replacement_required": replacement_required,
            },
        )

    # ------------------------------------------------------------------
    # I : Input generator
    # ------------------------------------------------------------------

    def generate_input(self, params: ProblemParams) -> str:
        """Render return request message.

        Uses LLM verbalization when available (with p_noise applied after).
        The user describes the product but does NOT mention the order ID —
        the agent must discover the order by scanning order history.
        Falls back to template rendering if LLM is unavailable.
        """
        from ecom_rlve.simulator.llm_backend import verbalize_return_request
        from ecom_rlve.simulator.templates import apply_noise
        import random as _rand

        p_missing = params.extra.get("p_missing", 0.0)
        p_noise = params.extra.get("p_noise", 0.0)
        product_title = params.extra["target_product_title"]
        reason = "It doesn't meet my expectations."
        replacement_required = params.extra.get("replacement_required", False)
        replacement_constraints = params.constraints if replacement_required else None

        # Try LLM verbalization first
        llm_text, mentioned, omitted = verbalize_return_request(
            product_title=product_title,
            reason=reason,
            replacement_required=replacement_required,
            replacement_constraints=replacement_constraints,
            p_missing=p_missing,
            seed=params.seed + 1,
        )

        if llm_text is not None:
            # Apply p_noise to LLM-generated text
            if p_noise > 0.0:
                noise_rng = _rand.Random(params.seed + 2)
                llm_text = apply_noise(llm_text, p_noise, noise_rng)
            return llm_text

        # Fallback: template-based (no order_ref — user must describe product)
        template_params: dict[str, Any] = {
            "product_desc": product_title,
        }
        if replacement_required:
            template_params["replacement_req"] = "I'd also like a replacement."
        template_params["reason"] = reason
        # order_ref is available for clarification but NOT in the initial message
        template_params["order_ref"] = params.extra["target_order_id"]

        return render_template(
            env_id=self.ENV_ID,
            params=template_params,
            p_missing=p_missing,
            p_noise=p_noise,
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
        """Compute r_task per Spec Section 5 / E4.

        r_task = clip(0.45*r_sel + 0.45*r_ret + 0.10*r_rep, -1, 1)
        """
        target_order_id: str = params.extra["target_order_id"]
        target_line_id: str = params.extra["target_line_id"]
        replacement_required: bool = params.extra["replacement_required"]

        initiated_returns: set[str] = episode_state.get("initiated_returns", set())

        # Replacement reward (if required, look for it in episode_state)
        replacement_reward: float | None = None
        if replacement_required:
            replacement_reward = episode_state.get("replacement_reward")

        r_task, is_correct = verify_return(
            answer=answer,
            target_order_id=target_order_id,
            target_line_id=target_line_id,
            initiated_returns=initiated_returns,
            replacement_required=replacement_required,
            replacement_reward=replacement_reward,
        )

        return EpisodeResult(
            r_task=r_task,
            is_correct=is_correct,
            details={
                "target_order_id": target_order_id,
                "target_line_id": target_line_id,
                "replacement_required": replacement_required,
                "n_orders": len(params.extra.get("orders", [])),
            },
        )
