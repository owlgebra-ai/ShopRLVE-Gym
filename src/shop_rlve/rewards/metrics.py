"""Shared metric computations for ShopRLVE-GYM reward system (Spec Section 5).

All metrics are implemented as pure functions with numpy. Every function
accepts simple inputs (lists, floats) and returns float.

Formulas implemented:
    nDCG:
        DCG(L) = sum_{i=1..k} rel(p_i) / log2(i+1)
        nDCG = DCG(L) / (IDCG + 1e-9)
        r_ndcg = 2*nDCG - 1  (mapped to [-1, 1])

    Constraint satisfaction:
        s(p|C) = (1/m) * sum_j C_j(p)
        F(p|C) = prod_j C_j(p)  (feasibility)

    F1:
        prec = U_match / (U_cart + 1e-9)
        rec  = U_match / (U_req + 1e-9)
        F1 = 2*prec*rec / (prec+rec+1e-9)

    Hallucination rate:
        hall_rate = |{p in L : p not in Seen}| / max(|L|, 1)

    Efficiency reward:
        r_eff = 1 - 2*(T-1)/(T_max-1)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# nDCG (Normalized Discounted Cumulative Gain)
# ---------------------------------------------------------------------------


def dcg(relevances: list[float], k: int | None = None) -> float:
    """Compute Discounted Cumulative Gain.

    Spec Section 5:
        DCG(L) = sum_{i=1..k} rel(p_i) / log2(i+1)

    Args:
        relevances: List of relevance scores in rank order.
        k:          Cutoff rank. If None, uses all items.

    Returns:
        DCG score as a float (>= 0).
    """
    if not relevances:
        return 0.0
    rels = np.array(relevances, dtype=np.float64)
    if k is not None:
        rels = rels[:k]
    positions = np.arange(1, len(rels) + 1, dtype=np.float64)
    discounts = np.log2(positions + 1.0)
    return float(np.sum(rels / discounts))


def ndcg(
    relevances: list[float],
    ideal_relevances: list[float],
    k: int | None = None,
) -> float:
    """Compute Normalized Discounted Cumulative Gain.

    Spec Section 5:
        nDCG = DCG(L) / (IDCG + 1e-9)

    The ideal relevances should be sorted in descending order (best ranking).

    Args:
        relevances:       Relevance scores in the order produced by the system.
        ideal_relevances: Relevance scores in ideal (descending) order.
        k:                Cutoff rank. If None, uses all items.

    Returns:
        nDCG score in [0, 1].
    """
    actual_dcg = dcg(relevances, k=k)
    ideal_dcg = dcg(ideal_relevances, k=k)
    return float(actual_dcg / (ideal_dcg + 1e-9))


def ndcg_reward(
    relevances: list[float],
    ideal_relevances: list[float],
    k: int | None = None,
) -> float:
    """Compute nDCG mapped to [-1, 1] for use as a reward signal.

    Spec Section 5:
        r_ndcg = 2*nDCG - 1

    Args:
        relevances:       Relevance scores in system order.
        ideal_relevances: Relevance scores in ideal order.
        k:                Cutoff rank.

    Returns:
        Reward in [-1, 1].
    """
    score = ndcg(relevances, ideal_relevances, k=k)
    return float(2.0 * score - 1.0)


# ---------------------------------------------------------------------------
# Constraint satisfaction
# ---------------------------------------------------------------------------


def constraint_satisfaction(product: Any, constraints: list[Callable[..., float]]) -> float:
    """Compute average constraint satisfaction score.

    Spec Section 5:
        s(p|C) = (1/m) * sum_j C_j(p)

    Each constraint function should return a value in [0, 1], where 1 means
    fully satisfied and 0 means fully violated.

    Args:
        product:     Product object passed to each constraint function.
        constraints: List of constraint functions, each taking a product and
                     returning a float in [0, 1].

    Returns:
        Average satisfaction score in [0, 1]. Returns 1.0 if constraints is empty.
    """
    if not constraints:
        return 1.0
    m = len(constraints)
    total = sum(float(c(product)) for c in constraints)
    return float(total / m)


def feasibility(product: Any, constraints: list[Callable[..., float]]) -> bool:
    """Check whether a product satisfies ALL constraints (binary feasibility).

    Spec Section 5:
        F(p|C) = prod_j C_j(p)

    A product is feasible if every constraint returns > 0. Since constraints
    return values in [0, 1], feasibility requires all constraints to be
    strictly positive.

    Args:
        product:     Product object passed to each constraint function.
        constraints: List of constraint functions.

    Returns:
        True if product satisfies all constraints (all return > 0).
    """
    if not constraints:
        return True
    return all(float(c(product)) > 0.0 for c in constraints)


# ---------------------------------------------------------------------------
# F1 Score
# ---------------------------------------------------------------------------


def f1_score(matched: int, predicted: int, actual: int) -> float:
    """Compute F1 score from matched, predicted, and actual counts.

    Spec Section 5:
        prec = U_match / (U_cart + 1e-9)
        rec  = U_match / (U_req + 1e-9)
        F1 = 2*prec*rec / (prec+rec+1e-9)

    Args:
        matched:   Number of correctly matched items (U_match).
        predicted: Number of items in the prediction (U_cart).
        actual:    Number of items in the ground truth (U_req).

    Returns:
        F1 score in [0, 1].
    """
    precision = matched / (predicted + 1e-9)
    recall = matched / (actual + 1e-9)
    return float(2.0 * precision * recall / (precision + recall + 1e-9))


def unit_f1(required_items: dict[str, int], cart_items: dict[str, int]) -> float:
    """Compute unit-level F1 for cart building (E_CART).

    Spec Section 5 (E_CART):
        prec = U_match / (U_cart + 1e-9)
        rec  = U_match / (U_req + 1e-9)
        F1 = 2*prec*rec / (prec+rec+1e-9)

    where U_match = sum over all product_ids of min(required_qty, cart_qty),
    U_cart = sum of all cart quantities, U_req = sum of all required quantities.

    Args:
        required_items: Dict mapping product_id -> required quantity.
        cart_items:     Dict mapping product_id -> quantity in cart.

    Returns:
        F1 score in [0, 1].
    """
    # Total required units
    u_req = sum(required_items.values())
    # Total cart units
    u_cart = sum(cart_items.values())

    # Matched units: for each required product, count min(required, in_cart)
    u_match = 0
    for pid, req_qty in required_items.items():
        cart_qty = cart_items.get(pid, 0)
        u_match += min(req_qty, cart_qty)

    return f1_score(matched=u_match, predicted=u_cart, actual=u_req)


# ---------------------------------------------------------------------------
# Hallucination rate
# ---------------------------------------------------------------------------


def hallucination_rate(output_ids: list[str], seen_ids: set[str]) -> float:
    """Compute hallucination rate: fraction of output IDs not in the seen set.

    Spec Section 5:
        hall_rate = |{p in L : p not in Seen}| / max(|L|, 1)

    A hallucinated product is one the agent recommends but was never surfaced
    to it via tool results.

    Args:
        output_ids: List of product IDs in the agent's output.
        seen_ids:   Set of product IDs that were surfaced to the agent.

    Returns:
        Hallucination rate in [0, 1].
    """
    if not output_ids:
        return 0.0
    unseen_count = sum(1 for pid in output_ids if pid not in seen_ids)
    return float(unseen_count / max(len(output_ids), 1))


# ---------------------------------------------------------------------------
# Efficiency reward
# ---------------------------------------------------------------------------


def efficiency_reward(turns: int, t_max: int) -> float:
    """Compute turn-efficiency reward.

    Spec Section 5:
        r_eff = 1 - 2*(T-1)/(T_max-1)

    Rewards completing the task in fewer turns. Returns 1.0 for T=1,
    -1.0 for T=T_max, and linearly interpolates between.

    Args:
        turns: Number of turns used (T >= 1).
        t_max: Maximum allowed turns (T_max >= 1).

    Returns:
        Efficiency reward in [-1, 1].
    """
    if t_max <= 1:
        return 1.0 if turns <= 1 else -1.0
    r = 1.0 - 2.0 * (turns - 1) / (t_max - 1)
    return float(np.clip(r, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Hallucination reward
# ---------------------------------------------------------------------------


def hallucination_reward(output_ids: list[str], seen_ids: set[str]) -> float:
    """Compute hallucination penalty reward.

    Spec Section 5:
        r_hall = -clip(hall_rate, 0, 1)

    Always non-positive: 0.0 for no hallucination, -1.0 for all hallucinated.

    Args:
        output_ids: List of product IDs in the agent's output.
        seen_ids:   Set of product IDs surfaced to the agent.

    Returns:
        Hallucination penalty in [-1, 0].
    """
    rate = hallucination_rate(output_ids, seen_ids)
    return float(-np.clip(rate, 0.0, 1.0))
