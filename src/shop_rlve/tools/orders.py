"""Order history generation and transactional order tools for ShopRLVE-GYM.

Spec Section 2.2 (Tool list B: Transactional):
    5. order.list       -- list recent orders with summary info
    6. order.get_status -- retrieve detailed status for a specific order
    7. order.checkout   -- create a new order from the current cart

Includes a deterministic order history generator that produces realistic
past orders for episode initialization. Orders span the last 90 days
with a realistic distribution of statuses.
"""

from __future__ import annotations

import logging
import random
from datetime import date, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from shop_rlve.data.schema import Product
from shop_rlve.tools.cart import CartState, _build_cart_view, _get_cart

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order data models
# ---------------------------------------------------------------------------


class OrderLine(BaseModel):
    """A single line item within an order.

    Attributes:
        line_id:       Unique line identifier within the order.
        product_id:    Product identifier.
        product_title: Human-readable product title.
        variant_id:    Optional variant identifier.
        qty:           Quantity ordered.
        unit_price:    Price per unit at time of order.
    """

    line_id: str = Field(..., description="Unique line identifier within the order")
    product_id: str = Field(..., description="Product identifier")
    product_title: str = Field(..., description="Product title at time of order")
    variant_id: str | None = Field(default=None, description="Optional variant identifier")
    qty: int = Field(..., ge=1, description="Quantity ordered")
    unit_price: float = Field(..., gt=0, description="Price per unit at time of order")


class Order(BaseModel):
    """A complete order record.

    Attributes:
        order_id:            Unique order identifier (e.g. "ord_001").
        order_date:          ISO-format date string (YYYY-MM-DD).
        status:              One of CONFIRMED, SHIPPED, IN_TRANSIT, DELIVERED, CANCELLED.
        eta_date:            Expected delivery date (ISO format) or None.
        tracking_number:     Tracking number string or None.
        shipping_address_id: Address reference (default: "addr_default").
        payment_method_id:   Payment method reference (default: "pay_default").
        lines:               List of OrderLine items.
    """

    order_id: str = Field(..., description="Unique order identifier")
    order_date: str = Field(..., description="Order date in ISO format (YYYY-MM-DD)")
    status: str = Field(
        ...,
        description="Order status: CONFIRMED, SHIPPED, IN_TRANSIT, DELIVERED, or CANCELLED",
    )
    eta_date: str | None = Field(default=None, description="Expected delivery date (ISO format)")
    tracking_number: str | None = Field(default=None, description="Tracking number")
    shipping_address_id: str = Field(
        default="addr_default", description="Shipping address reference"
    )
    payment_method_id: str = Field(default="pay_default", description="Payment method reference")
    lines: list[OrderLine] = Field(default_factory=list, description="Order line items")


# ---------------------------------------------------------------------------
# Status constants and distribution
# ---------------------------------------------------------------------------

ORDER_STATUSES: list[str] = ["CONFIRMED", "SHIPPED", "IN_TRANSIT", "DELIVERED", "CANCELLED"]

# Cumulative distribution for status sampling (spec: 10/15/20/45/10)
_STATUS_WEIGHTS: list[tuple[str, float]] = [
    ("CONFIRMED", 0.10),
    ("SHIPPED", 0.25),    # cumulative: 0.10 + 0.15
    ("IN_TRANSIT", 0.45), # cumulative: 0.25 + 0.20
    ("DELIVERED", 0.90),  # cumulative: 0.45 + 0.45
    ("CANCELLED", 1.00),  # cumulative: 0.90 + 0.10
]


def _sample_status(rng: random.Random) -> str:
    """Sample an order status according to the specified distribution."""
    r = rng.random()
    for status, cum_prob in _STATUS_WEIGHTS:
        if r < cum_prob:
            return status
    return "DELIVERED"  # fallback


# ---------------------------------------------------------------------------
# Order history generator
# ---------------------------------------------------------------------------


def generate_order_history(
    products: list[Product],
    n_orders: int,
    seed: int = 42,
    today: str | None = None,
) -> list[Order]:
    """Generate realistic synthetic order history.

    Creates n_orders orders spanning the last 90 days, each containing
    1-5 line items drawn from the provided product catalog. Order statuses
    follow the distribution: 10% CONFIRMED, 15% SHIPPED, 20% IN_TRANSIT,
    45% DELIVERED, 10% CANCELLED.

    Args:
        products:  List of Product objects to sample from.
        n_orders:  Number of orders to generate.
        seed:      Random seed for reproducibility.
        today:     Reference date as ISO string (YYYY-MM-DD). Defaults to today.

    Returns:
        List of Order objects sorted by order_date descending (most recent first).

    Raises:
        ValueError: If products list is empty or n_orders < 0.
    """
    if not products:
        raise ValueError("Cannot generate order history from empty product list")
    if n_orders < 0:
        raise ValueError(f"n_orders must be >= 0, got {n_orders}")
    if n_orders == 0:
        return []

    rng = random.Random(seed)

    # Parse reference date
    if today is not None:
        ref_date = date.fromisoformat(today)
    else:
        ref_date = date.today()

    orders: list[Order] = []

    for i in range(n_orders):
        order_id = f"ord_{i + 1:03d}"

        # Random date within last 90 days
        days_ago = rng.randint(0, 90)
        order_date = ref_date - timedelta(days=days_ago)
        order_date_str = order_date.isoformat()

        # Sample status
        status = _sample_status(rng)

        # Generate line items (1-5)
        n_lines = rng.randint(1, 5)
        sampled_products = rng.choices(products, k=n_lines)
        order_lines: list[OrderLine] = []
        for j, product in enumerate(sampled_products):
            line = OrderLine(
                line_id=f"{order_id}_line_{j + 1:02d}",
                product_id=product.id,
                product_title=product.title,
                variant_id=None,
                qty=rng.randint(1, 3),
                unit_price=product.price,
            )
            order_lines.append(line)

        # Compute ETA and tracking based on status
        eta_date: str | None = None
        tracking_number: str | None = None

        if status == "DELIVERED":
            # ETA is in the past (delivered already)
            delivery_offset = rng.randint(1, max(1, days_ago))
            eta = order_date + timedelta(days=delivery_offset)
            # Ensure ETA is in the past
            if eta > ref_date:
                eta = ref_date - timedelta(days=rng.randint(0, 3))
            eta_date = eta.isoformat()
            tracking_number = f"TRK{rng.randint(100000000, 999999999)}"

        elif status == "IN_TRANSIT":
            # ETA is in the future
            eta = ref_date + timedelta(days=rng.randint(1, 7))
            eta_date = eta.isoformat()
            tracking_number = f"TRK{rng.randint(100000000, 999999999)}"

        elif status == "SHIPPED":
            # ETA in the future, has tracking
            eta = ref_date + timedelta(days=rng.randint(2, 10))
            eta_date = eta.isoformat()
            tracking_number = f"TRK{rng.randint(100000000, 999999999)}"

        elif status == "CONFIRMED":
            # Estimated delivery, no tracking yet
            eta = order_date + timedelta(days=rng.randint(5, 14))
            eta_date = eta.isoformat()

        # CANCELLED: no ETA, no tracking

        order = Order(
            order_id=order_id,
            order_date=order_date_str,
            status=status,
            eta_date=eta_date,
            tracking_number=tracking_number,
            lines=order_lines,
        )
        orders.append(order)

    # Sort by order_date descending (most recent first)
    orders.sort(key=lambda o: o.order_date, reverse=True)

    return orders


# ---------------------------------------------------------------------------
# Pydantic arg models for tool registration
# ---------------------------------------------------------------------------


class OrderListArgs(BaseModel):
    """Arguments for order.list tool.

    Lists orders within a lookback period (default 30 days).
    """

    days: int = Field(default=30, ge=1, le=365, description="Lookback period in days (default: 30)")


class OrderGetStatusArgs(BaseModel):
    """Arguments for order.get_status tool."""

    order_id: str = Field(..., min_length=1, description="Order identifier to look up")


class OrderCheckoutArgs(BaseModel):
    """Arguments for order.checkout tool.

    Creates a new order from the current cart contents.
    """

    shipping_address_id: str = Field(
        ..., min_length=1, description="Shipping address reference ID"
    )
    payment_method_id: str = Field(..., min_length=1, description="Payment method reference ID")


# ---------------------------------------------------------------------------
# State accessor helpers
# ---------------------------------------------------------------------------


def _get_orders(state: Any) -> list[Order]:
    """Extract orders list from the episode state.

    Supports:
        - state.orders (attribute)
        - state["orders"] (dict)

    Returns:
        List of Order objects (possibly empty).
    """
    if hasattr(state, "orders"):
        return state.orders

    if isinstance(state, dict) and "orders" in state:
        return state["orders"]

    return []


def _get_today(state: Any) -> str:
    """Extract reference date from state, defaulting to today.

    Supports:
        - state.today (attribute, string)
        - state["today"] (dict)
        - Falls back to date.today().isoformat()
    """
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


def _get_next_order_counter(state: Any) -> int:
    """Get and increment the next order counter from state.

    Uses state._next_order_id or state["_next_order_id"]. If absent,
    computes from the existing orders list.
    """
    orders = _get_orders(state)
    existing_max = 0
    for order in orders:
        # Parse "ord_NNN" to get the numeric part
        if order.order_id.startswith("ord_"):
            try:
                num = int(order.order_id[4:])
                existing_max = max(existing_max, num)
            except ValueError:
                pass

    if hasattr(state, "_next_order_id"):
        counter = max(state._next_order_id, existing_max + 1)
        state._next_order_id = counter + 1
        return counter

    if isinstance(state, dict) and "_next_order_id" in state:
        counter = max(state["_next_order_id"], existing_max + 1)
        state["_next_order_id"] = counter + 1
        return counter

    # Fallback: derive from existing orders
    return existing_max + 1


# ---------------------------------------------------------------------------
# Tool handler: order.list
# ---------------------------------------------------------------------------


def order_list(
    days: int = 30,
    *,
    state: Any = None,
) -> list[dict[str, Any]]:
    """List recent orders within a lookback period.

    Args:
        days:  Number of days to look back (default: 30).
        state: Episode state with .orders.

    Returns:
        List of order summary dicts, each containing:
            - order_id: str
            - order_date: str
            - status: str
            - item_count: int (total quantity across all lines)
            - items: list of {product_title, qty}
    """
    orders = _get_orders(state)
    today_str = _get_today(state)
    ref_date = date.fromisoformat(today_str)
    cutoff_date = ref_date - timedelta(days=days)
    cutoff_str = cutoff_date.isoformat()

    summaries: list[dict[str, Any]] = []
    for order in orders:
        if order.order_date >= cutoff_str:
            item_count = sum(line.qty for line in order.lines)
            items = [
                {"product_title": line.product_title, "qty": line.qty}
                for line in order.lines
            ]
            summaries.append({
                "order_id": order.order_id,
                "order_date": order.order_date,
                "status": order.status,
                "item_count": item_count,
                "items": items,
            })

    return summaries


# ---------------------------------------------------------------------------
# Tool handler: order.get_status
# ---------------------------------------------------------------------------


def order_get_status(
    order_id: str,
    *,
    state: Any = None,
) -> dict[str, Any]:
    """Retrieve detailed status for a specific order.

    Args:
        order_id: Order identifier.
        state:    Episode state with .orders.

    Returns:
        Dict with order_id, status, eta_date, tracking_number, and lines.

    Raises:
        ValueError: If order_id not found.
    """
    orders = _get_orders(state)

    for order in orders:
        if order.order_id == order_id:
            return {
                "order_id": order.order_id,
                "status": order.status,
                "order_date": order.order_date,
                "eta_date": order.eta_date,
                "tracking_number": order.tracking_number,
                "shipping_address_id": order.shipping_address_id,
                "payment_method_id": order.payment_method_id,
                "lines": [line.model_dump() for line in order.lines],
            }

    raise ValueError(f"Order '{order_id}' not found")


# ---------------------------------------------------------------------------
# Tool handler: order.checkout
# ---------------------------------------------------------------------------


def order_checkout(
    shipping_address_id: str,
    payment_method_id: str,
    *,
    state: Any = None,
) -> dict[str, Any]:
    """Create a new order from the current cart contents.

    Validates the cart is not empty, creates an Order with status CONFIRMED,
    clears the cart, and appends the order to state.orders.

    Args:
        shipping_address_id: Shipping address reference ID.
        payment_method_id:   Payment method reference ID.
        state:               Episode state with .cart and .orders.

    Returns:
        Dict with order_id, status, order_date, and lines.

    Raises:
        ValueError: If cart is empty.
    """
    cart = _get_cart(state)
    orders = _get_orders(state)
    products_by_id = _get_products_by_id(state)

    if not cart.lines:
        raise ValueError("Cannot checkout: cart is empty")

    # Generate order ID
    order_counter = _get_next_order_counter(state)
    order_id = f"ord_{order_counter:03d}"

    # Get today's date
    today_str = _get_today(state)

    # Convert cart lines to order lines
    order_lines: list[OrderLine] = []
    for cart_line in cart.lines:
        product = products_by_id.get(cart_line.product_id)
        product_title: str
        if product is not None:
            if hasattr(product, "title"):
                product_title = product.title
            else:
                product_title = product.get("title", "Unknown Product")
        else:
            product_title = "Unknown Product"

        order_line = OrderLine(
            line_id=f"{order_id}_line_{len(order_lines) + 1:02d}",
            product_id=cart_line.product_id,
            product_title=product_title,
            variant_id=cart_line.variant_id,
            qty=cart_line.qty,
            unit_price=cart_line.unit_price,
        )
        order_lines.append(order_line)

    # Create order
    new_order = Order(
        order_id=order_id,
        order_date=today_str,
        status="CONFIRMED",
        eta_date=None,
        tracking_number=None,
        shipping_address_id=shipping_address_id,
        payment_method_id=payment_method_id,
        lines=order_lines,
    )

    # Append to orders list
    orders.append(new_order)

    # Clear cart
    cart.lines.clear()

    return {
        "order_id": new_order.order_id,
        "status": new_order.status,
        "order_date": new_order.order_date,
        "lines": [line.model_dump() for line in new_order.lines],
    }


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_order_tools(registry: Any) -> None:
    """Register all order tool handlers with a ToolRegistry instance.

    Registers:
        - order.list
        - order.get_status
        - order.checkout

    Args:
        registry: ToolRegistry instance.
    """
    registry.register("order.list", order_list, OrderListArgs)
    registry.register("order.get_status", order_get_status, OrderGetStatusArgs)
    registry.register("order.checkout", order_checkout, OrderCheckoutArgs)

    logger.info("Registered 3 order tools: list, get_status, checkout")
