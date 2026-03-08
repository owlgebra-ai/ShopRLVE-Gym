"""E_STATUS -- Order Status / Tracking environment.

Spec Section 5 / E5:
    Difficulty:
        H_orders(d) orders
        p_ref(d) = sigmoid((d-2)/1.5) -- indirect reference probability

    Generator:
        1. Generate H_orders(d) orders with product titles.
        2. Pick target order o*.
        3. User asks either directly (order_id) or indirectly
           ("the charger I bought last week") with prob p_ref(d).

    Reward:
        r_oid  = 1[answer.selected_order_id == o*]
        r_stat = 1[status matches ground truth]
        r_task = 2*(0.5*r_oid + 0.5*r_stat) - 1

    IsCorrect:
        r_oid == 1 AND r_stat == 1
"""

from __future__ import annotations

import random
from typing import Any

from shop_rlve.data.schema import DENIED_CATEGORIES, Product, avail
from shop_rlve.difficulty.mapping import map_difficulty, sigmoid
from shop_rlve.rewards.verifiers import verify_order_status
from shop_rlve.simulator.templates import render_template
from shop_rlve.tools.orders import generate_order_history

from shop_rlve.envs.base import (
    BaseEnvironment,
    EpisodeResult,
    ProblemParams,
    register_env,
)


@register_env
class OrderStatusEnv(BaseEnvironment):
    """E_STATUS: Order Status -- identify and report order status."""

    ENV_ID = "STATUS"

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
        """Generate order status query episode.

        Spec Section 5 / E5:
            p_ref(d) = sigmoid((d-2)/1.5)
            H_orders = H_orders(d)
        """
        dp = map_difficulty(difficulty)
        rng = random.Random(seed)

        p_ref = sigmoid((difficulty - 2) / 1.5)

        # Generate orders
        eligible = [p for p in catalog if p.cat not in DENIED_CATEGORIES]
        if not eligible:
            eligible = catalog

        orders = generate_order_history(
            products=eligible,
            n_orders=dp.H_orders_val,
            seed=seed,
        )
        if not orders:
            raise ValueError("Could not generate order history: empty catalog or zero orders")

        # Pick target order
        target_order = rng.choice(orders)

        # Determine whether the user references indirectly
        use_indirect = rng.random() < p_ref

        # Build indirect reference from product title
        indirect_ref = ""
        if use_indirect and target_order.lines:
            line = target_order.lines[0]
            indirect_ref = f"the {line.product_title} I ordered recently"

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
                "orders": orders,
                "target_order_id": target_order.order_id,
                "target_status": target_order.status,
                "target_eta": target_order.eta_date,
                "use_indirect": use_indirect,
                "indirect_ref": indirect_ref,
                "p_ref": p_ref,
            },
        )

    # ------------------------------------------------------------------
    # I : Input generator
    # ------------------------------------------------------------------

    def generate_input(self, params: ProblemParams) -> str:
        """Render order status query."""
        use_indirect: bool = params.extra.get("use_indirect", False)

        if use_indirect and params.extra.get("indirect_ref"):
            order_ref = params.extra["indirect_ref"]
        else:
            order_ref = params.extra["target_order_id"]

        template_params: dict[str, Any] = {"order_ref": order_ref}
        template_params["eta_req"] = "When will it arrive?"

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
        """Compute r_task per Spec Section 5 / E5.

        r_task = 2*(0.5*r_oid + 0.5*r_stat) - 1
        IsCorrect = r_oid == 1 AND r_stat == 1
        """
        target_order_id: str = params.extra["target_order_id"]
        target_status: str = params.extra["target_status"]
        target_eta: str | None = params.extra.get("target_eta")

        r_task, is_correct = verify_order_status(
            answer=answer,
            target_order_id=target_order_id,
            target_status=target_status,
            target_eta=target_eta,
        )

        return EpisodeResult(
            r_task=r_task,
            is_correct=is_correct,
            details={
                "target_order_id": target_order_id,
                "target_status": target_status,
                "use_indirect": params.extra.get("use_indirect", False),
                "n_orders": len(params.extra.get("orders", [])),
            },
        )
