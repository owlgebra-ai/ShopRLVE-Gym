"""Per-environment verification logic for EcomRLVE-GYM (Spec Section 5).

Each verifier computes r_task (float in [-1, 1]) and is_correct (bool)
for a specific environment type. These are deterministic, pure functions
that operate on structured episode outputs.

Environments:
    E_PD:      Product Discovery
    E_SUB:     Substitution
    E_CART:    Cart Building
    E_RETURN:  Returns
    E_STATUS:  Order Status
    E_POLICY:  Policy Q&A
    E_BUNDLE:  Bundle Planning
    E_JOURNEY: Multi-Intent Journey
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from ecom_rlve.rewards.metrics import (
    constraint_satisfaction,
    dcg,
    ndcg,
    ndcg_reward,
    unit_f1,
)
from ecom_rlve.simulator.persona import PersonaWeights, compute_utility


# ---------------------------------------------------------------------------
# E_PD: Product Discovery
# ---------------------------------------------------------------------------


def verify_product_discovery(
    recommended_ids: list[str],
    constraints: list[Callable[..., float]],
    persona_weights: PersonaWeights,
    products_by_id: dict[str, Any],
    eval_pool: list[str],
    k: int,
    difficulty: int,
    *,
    p_low: float,
    p_high: float,
    brand_pref: str | None = None,
    ref_embedding: np.ndarray | None = None,
    s_max: int = 14,
) -> tuple[float, bool]:
    """Verify product discovery recommendations.

    Spec Section 5.1 (E_PD):
        r_task = clip(0.55*r_rank + 0.35*r_cons + 0.10*r_oos, -1, 1)
        where:
            r_rank = 2*nDCG(rel=u(p)) - 1
            r_cons = 2*(s_best)^alpha(d) - 1, alpha(d) = 4 + floor(d/4)
            r_oos  = -clip(oos_rate, 0, 1)
        IsCorrect = 1[r_task >= 0.95]

    Args:
        recommended_ids:  List of product IDs recommended by the agent.
        constraints:      List of constraint functions (each: product -> [0,1]).
        persona_weights:  Persona preference weights.
        products_by_id:   Dict mapping product_id -> Product.
        eval_pool:        List of product IDs in the evaluation pool.
        k:                Number of items expected (k_rec(d)).
        difficulty:       Current difficulty level.
        p_low:            Price range lower bound.
        p_high:           Price range upper bound.
        brand_pref:       Preferred brand (optional).
        ref_embedding:    Reference embedding for similarity (optional).
        s_max:            Max shipping days for normalization.

    Returns:
        Tuple of (r_task, is_correct).
    """
    # Compute relevance scores (utility) for recommended products
    recommended_ids_k = recommended_ids[:k]
    relevances: list[float] = []
    for pid in recommended_ids_k:
        product = products_by_id.get(pid)
        if product is not None:
            u = compute_utility(
                product, persona_weights,
                p_low=p_low, p_high=p_high,
                brand_pref=brand_pref, ref_embedding=ref_embedding,
                s_max=s_max,
            )
            relevances.append(u)
        else:
            relevances.append(0.0)

    # Compute ideal relevances from eval pool
    pool_utilities: list[float] = []
    for pid in eval_pool:
        product = products_by_id.get(pid)
        if product is not None:
            u = compute_utility(
                product, persona_weights,
                p_low=p_low, p_high=p_high,
                brand_pref=brand_pref, ref_embedding=ref_embedding,
                s_max=s_max,
            )
            pool_utilities.append(u)
    pool_utilities.sort(reverse=True)
    ideal_relevances = pool_utilities[:k]

    # r_rank = 2*nDCG - 1
    r_rank = ndcg_reward(relevances, ideal_relevances, k=k)

    # r_cons = 2*(s_best)^alpha(d) - 1
    # s_best = best constraint satisfaction among recommended products
    alpha_d = 4 + math.floor(difficulty / 4)
    s_best = 0.0
    for pid in recommended_ids_k:
        product = products_by_id.get(pid)
        if product is not None and constraints:
            s = constraint_satisfaction(product, constraints)
            s_best = max(s_best, s)
    if not constraints:
        s_best = 1.0  # No constraints means trivially satisfied
    r_cons = 2.0 * (s_best ** alpha_d) - 1.0

    # r_oos = -clip(oos_rate, 0, 1)
    # OOS rate: fraction of recommended products that are out of stock
    oos_count = 0
    for pid in recommended_ids_k:
        product = products_by_id.get(pid)
        if product is not None:
            stock = product.stock_qty if hasattr(product, "stock_qty") else product.get("stock_qty", 0)
            if stock <= 0:
                oos_count += 1
        else:
            oos_count += 1  # Unknown product counts as OOS
    oos_rate = oos_count / max(len(recommended_ids_k), 1)
    r_oos = -float(np.clip(oos_rate, 0.0, 1.0))

    # Combine
    r_task = 0.55 * r_rank + 0.35 * r_cons + 0.10 * r_oos
    r_task = float(np.clip(r_task, -1.0, 1.0))

    is_correct = r_task >= 0.95

    return r_task, is_correct


# ---------------------------------------------------------------------------
# E_SUB: Substitution
# ---------------------------------------------------------------------------


def verify_substitution(
    recommended_ids: list[str],
    constraints: list[Callable[..., float]],
    original_product_id: str,
    products_by_id: dict[str, Any],
    eval_pool: list[str],
    k: int,
    difficulty: int,
    embeddings_fn: Callable[[str], np.ndarray | None],
) -> tuple[float, bool]:
    """Verify substitution recommendations.

    Spec Section 5.2 (E_SUB):
        r_task = clip(0.80*r_rank + 0.20*r_oos, -1, 1)
        where:
            rel_i = lambda_sim(d)*sim_i + (1-lambda_sim(d))*s_i
            lambda_sim(d) = clip(0.4 + 0.05*d, 0.4, 0.8)
            r_rank = 2*nDCG(rel) - 1
            r_oos = -clip(oos_rate, 0, 1)
        IsCorrect = 1[r_task >= 0.95]

    Args:
        recommended_ids:     List of recommended substitute product IDs.
        constraints:         List of constraint functions (product -> [0,1]).
        original_product_id: ID of the original (OOS) product.
        products_by_id:      Dict mapping product_id -> Product.
        eval_pool:           List of product IDs in the evaluation pool.
        k:                   Number of items expected.
        difficulty:          Current difficulty level.
        embeddings_fn:       Function mapping product_id -> embedding vector (or None).

    Returns:
        Tuple of (r_task, is_correct).
    """
    # lambda_sim(d) = clip(0.4 + 0.05*d, 0.4, 0.8)
    lambda_sim = float(np.clip(0.4 + 0.05 * difficulty, 0.4, 0.8))

    # Get original product embedding
    original_embedding = embeddings_fn(original_product_id)

    def _compute_rel(pid: str) -> float:
        """Compute composite relevance for a substitute product."""
        product = products_by_id.get(pid)
        if product is None:
            return 0.0

        # Similarity component
        sim_i = 0.0
        if original_embedding is not None:
            prod_embedding = embeddings_fn(pid)
            if prod_embedding is not None:
                cos_sim = float(np.dot(original_embedding, prod_embedding))
                sim_i = (cos_sim + 1.0) / 2.0  # scale to [0, 1]

        # Constraint satisfaction component
        s_i = constraint_satisfaction(product, constraints) if constraints else 1.0

        # Composite: rel_i = lambda_sim*sim_i + (1-lambda_sim)*s_i
        return lambda_sim * sim_i + (1.0 - lambda_sim) * s_i

    # Compute relevances for recommended products
    recommended_ids_k = recommended_ids[:k]
    relevances = [_compute_rel(pid) for pid in recommended_ids_k]

    # Compute ideal relevances from eval pool
    pool_rels = [_compute_rel(pid) for pid in eval_pool]
    pool_rels.sort(reverse=True)
    ideal_relevances = pool_rels[:k]

    # r_rank = 2*nDCG - 1
    r_rank = ndcg_reward(relevances, ideal_relevances, k=k)

    # r_oos = -clip(oos_rate, 0, 1)
    oos_count = 0
    for pid in recommended_ids_k:
        product = products_by_id.get(pid)
        if product is not None:
            stock = product.stock_qty if hasattr(product, "stock_qty") else product.get("stock_qty", 0)
            if stock <= 0:
                oos_count += 1
        else:
            oos_count += 1
    oos_rate = oos_count / max(len(recommended_ids_k), 1)
    r_oos = -float(np.clip(oos_rate, 0.0, 1.0))

    # Combine
    r_task = 0.80 * r_rank + 0.20 * r_oos
    r_task = float(np.clip(r_task, -1.0, 1.0))

    is_correct = r_task >= 0.95

    return r_task, is_correct


# ---------------------------------------------------------------------------
# E_CART: Cart Building
# ---------------------------------------------------------------------------


def verify_cart(
    required_items: dict[str, int],
    cart_lines: list[dict[str, Any]],
    products_by_id: dict[str, Any],
    variant_reqs: dict[str, str | None] | None = None,
) -> tuple[float, bool]:
    """Verify cart building accuracy (variant-aware).

    Spec Section 5.3 (E_CART):
        r_task = 2*F1 - 1
        IsCorrect = F1 == 1.0

    where F1 is computed at the unit level over composite keys:
        - If a variant is required for product_id p: key = "p::variant_id"
        - If no variant is required: key = "p" (any variant or none is fine)

    This ensures the agent is rewarded only when it picks both the correct
    product AND the correct variant (when one is specified).

    Args:
        required_items: Dict mapping product_id -> required quantity.
        cart_lines:     List of cart line dicts with 'product_id', 'variant_id', 'qty'.
        products_by_id: Dict mapping product_id -> Product (for validation).
        variant_reqs:   Dict mapping product_id -> required variant_id (or None).
                        Defaults to {} for backward compatibility.

    Returns:
        Tuple of (r_task, is_correct).
    """
    if variant_reqs is None:
        variant_reqs = {}

    # Build effective required dict with composite keys
    effective_required: dict[str, int] = {}
    for pid, qty in required_items.items():
        vid = variant_reqs.get(pid)
        key = f"{pid}::{vid}" if vid else pid
        effective_required[key] = qty

    # Build effective cart dict with matching composite keys
    effective_cart: dict[str, int] = {}
    for line in cart_lines:
        pid = line.get("product_id", "")
        cart_vid = line.get("variant_id")
        qty = line.get("qty", 0)

        # Use composite key if this product requires a specific variant
        req_vid = variant_reqs.get(pid)
        if req_vid is not None:
            # Product requires a variant — key by (pid, cart_variant_id)
            key = f"{pid}::{cart_vid}" if cart_vid else f"{pid}::None"
        else:
            # No variant required — key by pid only (ignore cart variant)
            key = pid

        effective_cart[key] = effective_cart.get(key, 0) + qty

    f1 = unit_f1(effective_required, effective_cart)
    r_task = 2.0 * f1 - 1.0
    r_task = float(np.clip(r_task, -1.0, 1.0))

    is_correct = abs(f1 - 1.0) < 1e-9

    return r_task, is_correct


# ---------------------------------------------------------------------------
# E_RETURN: Returns
# ---------------------------------------------------------------------------


def verify_return(
    answer: dict[str, Any] | None,
    target_order_id: str,
    target_line_id: str,
    initiated_returns: set[str],
    replacement_required: bool,
    replacement_reward: float | None = None,
) -> tuple[float, bool]:
    """Verify return handling.

    Spec Section 5.4 (E_RETURN):
        r_task = clip(0.45*r_sel + 0.45*r_ret + 0.10*r_rep, -1, 1)

    Components:
        r_sel: Did the agent identify the correct order and line?
            - 1.0 if correct order_id AND line_id selected
            - -1.0 otherwise
        r_ret: Did the agent successfully initiate the return?
            - 1.0 if a return was initiated (return_id in initiated_returns)
            - -1.0 otherwise
        r_rep: Did the agent find a replacement (if required)?
            - replacement_reward if provided (in [-1, 1])
            - 1.0 if not required
            - -1.0 if required but not provided

    IsCorrect = r_sel==1 AND r_ret==1 AND r_rep>=0.95

    Args:
        answer:               Agent's structured answer dict with 'selected_order_id',
                              'selected_line_id', and optionally 'return_id'.
        target_order_id:      Expected order ID.
        target_line_id:       Expected line ID.
        initiated_returns:    Set of return IDs that were successfully initiated.
        replacement_required: Whether a replacement product was part of the goal.
        replacement_reward:   Reward for the replacement quality (if applicable).

    Returns:
        Tuple of (r_task, is_correct).
    """
    if answer is None:
        return -1.0, False

    # r_sel: correct order and line selection
    selected_order = answer.get("selected_order_id", "")
    selected_line = answer.get("selected_line_id", "")
    r_sel = 1.0 if (selected_order == target_order_id and selected_line == target_line_id) else -1.0

    # r_ret: return initiated successfully
    r_ret = -1.0
    if initiated_returns:
        # Check if any return was initiated (the set is populated by the tools)
        r_ret = 1.0

    # r_rep: replacement handling
    if not replacement_required:
        r_rep = 1.0  # Not required, full credit
    elif replacement_reward is not None:
        r_rep = float(np.clip(replacement_reward, -1.0, 1.0))
    else:
        r_rep = -1.0  # Required but no replacement provided

    # Combine
    r_task = 0.45 * r_sel + 0.45 * r_ret + 0.10 * r_rep
    r_task = float(np.clip(r_task, -1.0, 1.0))

    is_correct = (
        r_sel == 1.0
        and r_ret == 1.0
        and r_rep >= 0.95
    )

    return r_task, is_correct


# ---------------------------------------------------------------------------
# E_STATUS: Order Status
# ---------------------------------------------------------------------------


def verify_order_status(
    answer: dict[str, Any] | None,
    target_order_id: str,
    target_status: str,
    target_eta: str | None,
) -> tuple[float, bool]:
    """Verify order status query.

    Spec Section 5.5 (E_STATUS):
        r_task = 2*(0.5*r_oid + 0.5*r_stat) - 1

    Components:
        r_oid:  1.0 if correct order_id identified, 0.0 otherwise.
        r_stat: 1.0 if correct status reported, 0.0 otherwise.

    IsCorrect = r_oid==1 AND r_stat==1

    Args:
        answer:          Agent's structured answer dict with 'selected_order_id',
                         'order_status', and optionally 'eta'.
        target_order_id: Expected order ID.
        target_status:   Expected order status string.
        target_eta:      Expected ETA string (not used in reward but for validation).

    Returns:
        Tuple of (r_task, is_correct).
    """
    if answer is None:
        return -1.0, False

    # r_oid: correct order ID
    selected_order = answer.get("selected_order_id", "")
    r_oid = 1.0 if selected_order == target_order_id else 0.0

    # r_stat: correct status
    reported_status = answer.get("order_status", "")
    r_stat = 1.0 if reported_status.upper().strip() == target_status.upper().strip() else 0.0

    # Combine: r_task = 2*(0.5*r_oid + 0.5*r_stat) - 1
    r_task = 2.0 * (0.5 * r_oid + 0.5 * r_stat) - 1.0
    r_task = float(np.clip(r_task, -1.0, 1.0))

    is_correct = r_oid == 1.0 and r_stat == 1.0

    return r_task, is_correct


# ---------------------------------------------------------------------------
# E_POLICY: Policy Q&A
# ---------------------------------------------------------------------------


def verify_policy(
    answer_value: Any,
    expected_value: Any,
    answer_type: str = "numeric",
) -> tuple[float, bool]:
    """Verify policy question answer.

    Spec Section 5.6 (E_POLICY):
        Numeric:     rho = (min(x,y)/max(x,y))^4, r_task = 2*rho - 1
        Categorical: exact match = 1, else -1

    IsCorrect = r_task >= 0.95

    Args:
        answer_value:   The agent's answer (number or string).
        expected_value: The expected correct answer.
        answer_type:    "numeric" or "categorical".

    Returns:
        Tuple of (r_task, is_correct).
    """
    if answer_type == "numeric":
        try:
            x = float(answer_value)
            y = float(expected_value)
        except (TypeError, ValueError):
            return -1.0, False

        # Handle zero cases
        if abs(y) < 1e-12 and abs(x) < 1e-12:
            # Both zero: perfect match
            rho = 1.0
        elif abs(y) < 1e-12 or abs(x) < 1e-12:
            # One is zero, other isn't: poor match
            rho = 0.0
        else:
            # rho = (min(x,y) / max(x,y))^4
            x_abs = abs(x)
            y_abs = abs(y)
            ratio = min(x_abs, y_abs) / max(x_abs, y_abs)
            rho = ratio ** 4

        r_task = 2.0 * rho - 1.0
        r_task = float(np.clip(r_task, -1.0, 1.0))

    elif answer_type == "categorical":
        # Exact string match (case-insensitive, stripped)
        if answer_value is None or expected_value is None:
            r_task = -1.0
        elif str(answer_value).lower().strip() == str(expected_value).lower().strip():
            r_task = 1.0
        else:
            r_task = -1.0
    else:
        raise ValueError(f"Unknown answer_type: '{answer_type}'. Must be 'numeric' or 'categorical'.")

    is_correct = r_task >= 0.95

    return r_task, is_correct


# ---------------------------------------------------------------------------
# E_BUNDLE: Bundle Planning
# ---------------------------------------------------------------------------


def verify_bundle(
    recommended_ids: list[str],
    required_categories: list[str],
    products_by_id: dict[str, Any],
    budget: float | None = None,
) -> tuple[float, bool]:
    """Verify bundle recommendations.

    Spec Section 5.7 (E_BUNDLE):
        r_task = clip(2*F1 - 1 - budget_penalty, -1, 1)

    where F1 is computed on category coverage:
        - For each required category, check if any recommended product is in that category
        - matched = number of required categories covered
        - predicted = number of distinct categories in recommendations
        - actual = number of required categories

    budget_penalty:
        - 0 if budget is None or total_price <= budget
        - (total_price - budget) / budget if over budget

    IsCorrect = F1 == 1.0 AND budget_penalty == 0

    Args:
        recommended_ids:     List of recommended product IDs (one per category).
        required_categories: List of required category strings.
        products_by_id:      Dict mapping product_id -> Product.
        budget:              Optional total budget constraint.

    Returns:
        Tuple of (r_task, is_correct).
    """
    # Determine which categories are covered by recommendations
    rec_categories: list[str] = []
    total_price = 0.0
    for pid in recommended_ids:
        product = products_by_id.get(pid)
        if product is not None:
            cat = product.cat if hasattr(product, "cat") else product.get("cat", "unknown")
            rec_categories.append(cat)
            price = product.price if hasattr(product, "price") else product.get("price", 0.0)
            total_price += price
        else:
            rec_categories.append("_unknown_")

    # Compute category-level F1
    required_set = set(required_categories)
    covered = set(rec_categories) & required_set
    matched = len(covered)
    predicted = len(set(rec_categories))
    actual = len(required_set)

    # F1
    precision = matched / (predicted + 1e-9)
    recall = matched / (actual + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)

    # Budget penalty
    budget_penalty = 0.0
    if budget is not None and budget > 0 and total_price > budget:
        budget_penalty = (total_price - budget) / budget

    r_task = 2.0 * f1 - 1.0 - budget_penalty
    r_task = float(np.clip(r_task, -1.0, 1.0))

    is_correct = abs(f1 - 1.0) < 1e-9 and budget_penalty < 1e-9

    return r_task, is_correct


# ---------------------------------------------------------------------------
# E_JOURNEY: Multi-Intent Journey
# ---------------------------------------------------------------------------


def verify_journey(subtask_rewards: list[float]) -> tuple[float, bool]:
    """Verify multi-intent journey completion.

    Spec Section 5.8 (E_JOURNEY):
        r_task = clip(mean(subtask_rewards), -1, 1)
        IsCorrect = all(r_j >= 0.95 for r_j in subtask_rewards)

    Args:
        subtask_rewards: List of r_task values from each sub-environment.

    Returns:
        Tuple of (r_task, is_correct).
    """
    if not subtask_rewards:
        return -1.0, False

    mean_reward = float(np.mean(subtask_rewards))
    r_task = float(np.clip(mean_reward, -1.0, 1.0))

    is_correct = all(r >= 0.95 for r in subtask_rewards)

    return r_task, is_correct
