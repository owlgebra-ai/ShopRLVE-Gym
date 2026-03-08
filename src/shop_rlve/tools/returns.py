"""Returns and exchange engine for ShopRLVE-GYM.

Spec Section 2.2 (Tool list B: Transactional):
    8. return.check_eligibility -- check if an order line is eligible for return
    9. return.initiate          -- initiate a return for an eligible line
   10. return.exchange          -- initiate an exchange for an eligible line

Return eligibility is governed by category-specific return windows:
    - electronics: 15 days
    - clothing: 30 days
    - furniture: 30 days
    - groceries: 0 days (non-returnable)
    - default: 30 days

Cancelled orders are always ineligible for returns.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from shop_rlve.tools.orders import Order, OrderLine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return policy constants
# ---------------------------------------------------------------------------

RETURN_WINDOWS: dict[str, int] = {
    "electronics": 15,
    "clothing": 30,
    "furniture": 30,
    "groceries": 0,  # non-returnable
    "default": 30,
}

VALID_REASON_CODES: list[str] = [
    "defective",
    "wrong_item",
    "not_as_described",
    "changed_mind",
    "too_large",
    "too_small",
    "arrived_late",
    "better_price_found",
]

RETURN_METHODS: list[str] = ["mail", "in_store", "pickup"]

RETURN_FEES: dict[str, float] = {
    "mail": 5.99,
    "in_store": 0.0,
    "pickup": 0.0,
}


# ---------------------------------------------------------------------------
# Pydantic arg models for tool registration
# ---------------------------------------------------------------------------


class ReturnCheckEligibilityArgs(BaseModel):
    """Arguments for return.check_eligibility tool."""

    order_id: str = Field(..., min_length=1, description="Order identifier")
    line_id: str = Field(..., min_length=1, description="Line identifier within the order")


class ReturnInitiateArgs(BaseModel):
    """Arguments for return.initiate tool."""

    order_id: str = Field(..., min_length=1, description="Order identifier")
    line_id: str = Field(..., min_length=1, description="Line identifier within the order")
    reason_code: str = Field(
        ...,
        min_length=1,
        description=(
            "Reason for return. Must be one of: defective, wrong_item, not_as_described, "
            "changed_mind, too_large, too_small, arrived_late, better_price_found"
        ),
    )
    method: str = Field(
        ...,
        min_length=1,
        description="Return method. Must be one of: mail, in_store, pickup",
    )


class ReturnExchangeArgs(BaseModel):
    """Arguments for return.exchange tool."""

    order_id: str = Field(..., min_length=1, description="Order identifier")
    line_id: str = Field(..., min_length=1, description="Line identifier within the order")
    new_product_id: str = Field(
        ..., min_length=1, description="Product ID to exchange for"
    )
    new_variant_id: str | None = Field(
        default=None, description="Optional variant ID for the new product"
    )


# ---------------------------------------------------------------------------
# State accessor helpers
# ---------------------------------------------------------------------------


def _get_orders(state: Any) -> list[Order]:
    """Extract orders list from the episode state."""
    if hasattr(state, "orders"):
        return state.orders

    if isinstance(state, dict) and "orders" in state:
        return state["orders"]

    return []


def _get_today(state: Any) -> str:
    """Extract reference date from state, defaulting to today."""
    if hasattr(state, "today"):
        return state.today

    if isinstance(state, dict) and "today" in state:
        return state["today"]

    return date.today().isoformat()


def _get_products_by_id(state: Any) -> dict[str, Any]:
    """Extract products_by_id mapping from the episode state."""
    if hasattr(state, "products_by_id"):
        return state.products_by_id

    if isinstance(state, dict) and "products_by_id" in state:
        return state["products_by_id"]

    if hasattr(state, "catalog_state") and hasattr(state.catalog_state, "products_by_id"):
        return state.catalog_state.products_by_id

    if (
        isinstance(state, dict)
        and "catalog_state" in state
        and hasattr(state["catalog_state"], "products_by_id")
    ):
        return state["catalog_state"].products_by_id

    raise ValueError(
        "Could not find products_by_id in the provided state object. "
        "Ensure state has a .products_by_id attribute or .catalog_state.products_by_id."
    )


def _get_initiated_returns(state: Any) -> set[str]:
    """Get or create the initiated_returns tracking set on state.

    Returns:
        A mutable set of return/exchange IDs that have been initiated.
    """
    if hasattr(state, "initiated_returns"):
        return state.initiated_returns

    if isinstance(state, dict):
        if "initiated_returns" not in state:
            state["initiated_returns"] = set()
        return state["initiated_returns"]

    # Attach to state object if not present
    state.initiated_returns = set()
    return state.initiated_returns


def _get_next_return_counter(state: Any) -> int:
    """Get and increment the next return counter from state."""
    if hasattr(state, "_next_return_id"):
        counter = state._next_return_id
        state._next_return_id = counter + 1
        return counter

    if isinstance(state, dict):
        counter = state.get("_next_return_id", 1)
        state["_next_return_id"] = counter + 1
        return counter

    state._next_return_id = 2
    return 1


def _get_next_exchange_counter(state: Any) -> int:
    """Get and increment the next exchange counter from state."""
    if hasattr(state, "_next_exchange_id"):
        counter = state._next_exchange_id
        state._next_exchange_id = counter + 1
        return counter

    if isinstance(state, dict):
        counter = state.get("_next_exchange_id", 1)
        state["_next_exchange_id"] = counter + 1
        return counter

    state._next_exchange_id = 2
    return 1


# ---------------------------------------------------------------------------
# Internal: find order and line
# ---------------------------------------------------------------------------


def _find_order_and_line(
    state: Any,
    order_id: str,
    line_id: str,
) -> tuple[Order, OrderLine]:
    """Locate an order and a specific line within it.

    Args:
        state:    Episode state.
        order_id: Order identifier.
        line_id:  Line identifier within the order.

    Returns:
        Tuple of (Order, OrderLine).

    Raises:
        ValueError: If order or line not found.
    """
    orders = _get_orders(state)

    target_order: Order | None = None
    for order in orders:
        if order.order_id == order_id:
            target_order = order
            break

    if target_order is None:
        raise ValueError(f"Order '{order_id}' not found")

    target_line: OrderLine | None = None
    for line in target_order.lines:
        if line.line_id == line_id:
            target_line = line
            break

    if target_line is None:
        raise ValueError(
            f"Line '{line_id}' not found in order '{order_id}'. "
            f"Available lines: {[l.line_id for l in target_order.lines]}"
        )

    return target_order, target_line


def _get_product_category(state: Any, product_id: str) -> str:
    """Look up the category for a product, defaulting to 'general'."""
    try:
        products_by_id = _get_products_by_id(state)
    except ValueError:
        return "general"

    product = products_by_id.get(product_id)
    if product is None:
        return "general"

    if hasattr(product, "cat"):
        return product.cat
    if isinstance(product, dict):
        return product.get("cat", "general")
    return "general"


def _get_product_title(state: Any, product_id: str) -> str:
    """Look up the title for a product."""
    try:
        products_by_id = _get_products_by_id(state)
    except ValueError:
        return "Unknown Product"

    product = products_by_id.get(product_id)
    if product is None:
        return "Unknown Product"

    if hasattr(product, "title"):
        return product.title
    if isinstance(product, dict):
        return product.get("title", "Unknown Product")
    return "Unknown Product"


# ---------------------------------------------------------------------------
# Internal: compute eligibility
# ---------------------------------------------------------------------------


def _compute_eligibility(
    state: Any,
    order: Order,
    line: OrderLine,
) -> dict[str, Any]:
    """Compute return eligibility for an order line.

    Returns a dict with:
        - eligible: bool
        - window_days: int
        - days_since_purchase: int
        - reasons: list[str] (only if eligible)
        - methods: list[str] (only if eligible)
        - fees: dict[str, float] (only if eligible)
        - product_title: str
        - ineligible_reason: str | None (if not eligible)
    """
    today_str = _get_today(state)
    ref_date = date.fromisoformat(today_str)
    order_date = date.fromisoformat(order.order_date)
    days_since = (ref_date - order_date).days

    product_cat = _get_product_category(state, line.product_id)
    product_title = line.product_title

    # Lookup return window by category
    window_days = RETURN_WINDOWS.get(product_cat, RETURN_WINDOWS["default"])

    # Check eligibility conditions
    ineligible_reason: str | None = None

    if order.status == "CANCELLED":
        ineligible_reason = "Order has been cancelled"
    elif window_days == 0:
        ineligible_reason = f"Category '{product_cat}' is non-returnable"
    elif days_since > window_days:
        ineligible_reason = (
            f"Return window expired: {days_since} days since purchase, "
            f"window is {window_days} days"
        )

    eligible = ineligible_reason is None

    result: dict[str, Any] = {
        "eligible": eligible,
        "window_days": window_days,
        "days_since_purchase": days_since,
        "product_title": product_title,
    }

    if eligible:
        result["reasons"] = list(VALID_REASON_CODES)
        result["methods"] = list(RETURN_METHODS)
        result["fees"] = dict(RETURN_FEES)
    else:
        result["ineligible_reason"] = ineligible_reason
        result["reasons"] = []
        result["methods"] = []
        result["fees"] = {}

    return result


# ---------------------------------------------------------------------------
# Tool handler: return.check_eligibility
# ---------------------------------------------------------------------------


def return_check_eligibility(
    order_id: str,
    line_id: str,
    *,
    state: Any = None,
) -> dict[str, Any]:
    """Check whether an order line is eligible for return.

    Evaluates:
        - Order status (cancelled orders are ineligible)
        - Product category return window
        - Days since purchase vs. return window

    Args:
        order_id: Order identifier.
        line_id:  Line identifier within the order.
        state:    Episode state with .orders and .products_by_id.

    Returns:
        Dict with eligibility info:
            - eligible: bool
            - window_days: int
            - days_since_purchase: int
            - reasons: list[str] (valid reason codes, empty if ineligible)
            - methods: list[str] (valid return methods, empty if ineligible)
            - fees: dict[str, float] (per-method fees, empty if ineligible)
            - product_title: str
            - ineligible_reason: str | None

    Raises:
        ValueError: If order_id or line_id not found.
    """
    order, line = _find_order_and_line(state, order_id, line_id)
    return _compute_eligibility(state, order, line)


# ---------------------------------------------------------------------------
# Tool handler: return.initiate
# ---------------------------------------------------------------------------


def return_initiate(
    order_id: str,
    line_id: str,
    reason_code: str,
    method: str,
    *,
    state: Any = None,
) -> dict[str, Any]:
    """Initiate a return for an eligible order line.

    Validates eligibility, reason code, and return method before
    creating the return record.

    Args:
        order_id:    Order identifier.
        line_id:     Line identifier within the order.
        reason_code: Return reason (must be in VALID_REASON_CODES).
        method:      Return method (must be in RETURN_METHODS).
        state:       Episode state with .orders and .products_by_id.

    Returns:
        Dict with:
            - return_id: str
            - order_id: str
            - line_id: str
            - reason_code: str
            - method: str
            - status: "INITIATED"
            - fee: float
            - label_url: str | None (only for mail method)

    Raises:
        ValueError: If ineligible, invalid reason_code, or invalid method.
    """
    order, line = _find_order_and_line(state, order_id, line_id)

    # Check eligibility
    eligibility = _compute_eligibility(state, order, line)
    if not eligibility["eligible"]:
        raise ValueError(
            f"Return not eligible for order '{order_id}' line '{line_id}': "
            f"{eligibility.get('ineligible_reason', 'unknown reason')}"
        )

    # Validate reason_code
    if reason_code not in VALID_REASON_CODES:
        raise ValueError(
            f"Invalid reason_code '{reason_code}'. "
            f"Must be one of: {VALID_REASON_CODES}"
        )

    # Validate method
    if method not in RETURN_METHODS:
        raise ValueError(
            f"Invalid return method '{method}'. "
            f"Must be one of: {RETURN_METHODS}"
        )

    # Generate return ID
    counter = _get_next_return_counter(state)
    return_id = f"ret_{counter:03d}"

    # Track in initiated_returns
    initiated = _get_initiated_returns(state)
    initiated.add(return_id)

    # Build label URL for mail returns
    label_url: str | None = None
    if method == "mail":
        label_url = f"https://returns.shoprlve.example/labels/{return_id}"

    fee = RETURN_FEES.get(method, 0.0)

    return {
        "return_id": return_id,
        "order_id": order_id,
        "line_id": line_id,
        "reason_code": reason_code,
        "method": method,
        "status": "INITIATED",
        "fee": fee,
        "label_url": label_url,
    }


# ---------------------------------------------------------------------------
# Tool handler: return.exchange
# ---------------------------------------------------------------------------


def return_exchange(
    order_id: str,
    line_id: str,
    new_product_id: str,
    new_variant_id: str | None = None,
    *,
    state: Any = None,
) -> dict[str, Any]:
    """Initiate an exchange for an eligible order line.

    Validates return eligibility, then checks that the replacement
    product exists and is in stock before creating the exchange record.

    Args:
        order_id:       Order identifier.
        line_id:        Line identifier within the order.
        new_product_id: Replacement product ID.
        new_variant_id: Optional replacement variant ID.
        state:          Episode state with .orders, .products_by_id.

    Returns:
        Dict with:
            - exchange_id: str
            - return_id: str
            - old_product_id: str
            - old_product_title: str
            - new_product_id: str
            - new_product_title: str
            - new_variant_id: str | None
            - status: "EXCHANGE_INITIATED"

    Raises:
        ValueError: If ineligible, new product not found, or out of stock.
    """
    order, line = _find_order_and_line(state, order_id, line_id)

    # Check return eligibility
    eligibility = _compute_eligibility(state, order, line)
    if not eligibility["eligible"]:
        raise ValueError(
            f"Exchange not eligible for order '{order_id}' line '{line_id}': "
            f"{eligibility.get('ineligible_reason', 'unknown reason')}"
        )

    # Validate new product exists and is in stock
    products_by_id = _get_products_by_id(state)
    new_product = products_by_id.get(new_product_id)
    if new_product is None:
        raise ValueError(f"Replacement product '{new_product_id}' not found")

    new_stock: int
    new_title: str
    if hasattr(new_product, "stock_qty"):
        new_stock = new_product.stock_qty
        new_title = new_product.title
    else:
        new_stock = new_product.get("stock_qty", 0)
        new_title = new_product.get("title", "Unknown Product")

    if new_stock < 1:
        raise ValueError(
            f"Replacement product '{new_product_id}' is out of stock"
        )

    # Generate return and exchange IDs
    return_counter = _get_next_return_counter(state)
    return_id = f"ret_{return_counter:03d}"

    exchange_counter = _get_next_exchange_counter(state)
    exchange_id = f"exc_{exchange_counter:03d}"

    # Track in initiated_returns
    initiated = _get_initiated_returns(state)
    initiated.add(return_id)
    initiated.add(exchange_id)

    return {
        "exchange_id": exchange_id,
        "return_id": return_id,
        "old_product_id": line.product_id,
        "old_product_title": line.product_title,
        "new_product_id": new_product_id,
        "new_product_title": new_title,
        "new_variant_id": new_variant_id,
        "status": "EXCHANGE_INITIATED",
    }


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_return_tools(registry: Any) -> None:
    """Register all return tool handlers with a ToolRegistry instance.

    Registers:
        - return.check_eligibility
        - return.initiate
        - return.exchange

    Args:
        registry: ToolRegistry instance.
    """
    registry.register(
        "return.check_eligibility",
        return_check_eligibility,
        ReturnCheckEligibilityArgs,
    )
    registry.register("return.initiate", return_initiate, ReturnInitiateArgs)
    registry.register("return.exchange", return_exchange, ReturnExchangeArgs)

    logger.info("Registered 3 return tools: check_eligibility, initiate, exchange")
