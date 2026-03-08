"""In-memory cart with deterministic operations for ShopRLVE-GYM.

Spec Section 2.2 (Tool list B: Transactional):
    1. cart.view         -- view current cart contents
    2. cart.add          -- add a product to the cart
    3. cart.remove       -- remove a line item from the cart
    4. cart.set_quantity -- update quantity for a line item

All operations are deterministic: given the same sequence of calls,
the cart state is always identical. No external I/O is involved.

Debug lever:
    CartState.trace_mode = True  -- log every cart mutation with before/after state.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cart data models
# ---------------------------------------------------------------------------


class CartLine(BaseModel):
    """A single line item in the shopping cart.

    Attributes:
        line_id:    Unique line identifier (e.g. "line_001").
        product_id: Product identifier.
        variant_id: Optional variant identifier.
        qty:        Quantity (>= 1).
        unit_price: Price per unit in USD.
    """

    line_id: str = Field(..., description="Unique line identifier (e.g. 'line_001')")
    product_id: str = Field(..., description="Product identifier")
    variant_id: str | None = Field(default=None, description="Optional variant identifier")
    qty: int = Field(..., ge=1, description="Quantity, must be >= 1")
    unit_price: float = Field(..., gt=0, description="Price per unit in USD")


class CartState(BaseModel):
    """In-memory cart state with deterministic operations.

    Attributes:
        lines:       List of CartLine items currently in the cart.
        trace_mode:  When True, log every mutation with before/after state.

    The _next_line_id counter is tracked as a private attribute for
    generating sequential line IDs like "line_001", "line_002", etc.
    """

    lines: list[CartLine] = Field(default_factory=list, description="Cart line items")
    trace_mode: bool = Field(default=False, description="Enable mutation tracing")
    next_line_id: int = Field(default=1, description="Counter for generating line IDs", exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    def _generate_line_id(self) -> str:
        """Generate the next sequential line ID."""
        line_id = f"line_{self.next_line_id:03d}"
        self.next_line_id += 1
        return line_id

    def _snapshot(self) -> list[dict[str, Any]]:
        """Return a snapshot of current cart lines for tracing."""
        return [line.model_dump() for line in self.lines]

    def _log_mutation(
        self,
        operation: str,
        before: list[dict[str, Any]],
        after: list[dict[str, Any]],
        details: str = "",
    ) -> None:
        """Log a cart mutation if trace_mode is enabled."""
        if self.trace_mode:
            logger.info(
                "CartState TRACE [%s] %s\n  BEFORE: %s\n  AFTER:  %s",
                operation,
                details,
                before,
                after,
            )


# ---------------------------------------------------------------------------
# Pydantic arg models for tool registration
# ---------------------------------------------------------------------------


class CartViewArgs(BaseModel):
    """Arguments for cart.view tool (no arguments required)."""


class CartAddArgs(BaseModel):
    """Arguments for cart.add tool.

    Adds a product to the cart. If the same (product_id, variant_id)
    combination already exists, increments the quantity.
    """

    product_id: str = Field(..., min_length=1, description="Product identifier to add")
    variant_id: str | None = Field(default=None, description="Optional variant identifier")
    quantity: int = Field(default=1, ge=1, description="Quantity to add (default: 1)")


class CartRemoveArgs(BaseModel):
    """Arguments for cart.remove tool."""

    line_id: str = Field(..., min_length=1, description="Line ID to remove from cart")


class CartSetQuantityArgs(BaseModel):
    """Arguments for cart.set_quantity tool.

    Set quantity to 0 to remove the line entirely.
    """

    line_id: str = Field(..., min_length=1, description="Line ID to update")
    quantity: int = Field(..., ge=0, description="New quantity (0 to remove)")


# ---------------------------------------------------------------------------
# Helper: build cart view dict
# ---------------------------------------------------------------------------


def _build_cart_view(cart: CartState) -> dict[str, Any]:
    """Build a standardized cart view dictionary.

    Returns:
        Dict with keys: lines, total_items, total_price.
    """
    lines_data = [line.model_dump() for line in cart.lines]
    total_items = sum(line.qty for line in cart.lines)
    total_price = round(sum(line.qty * line.unit_price for line in cart.lines), 2)

    return {
        "lines": lines_data,
        "total_items": total_items,
        "total_price": total_price,
    }


# ---------------------------------------------------------------------------
# State accessor helpers
# ---------------------------------------------------------------------------


def _get_cart(state: Any) -> CartState:
    """Extract CartState from the episode state object.

    Supports:
        - state.cart (attribute)
        - state["cart"] (dict)
        - state itself is a CartState

    Raises:
        ValueError: If no CartState can be found.
    """
    if isinstance(state, CartState):
        return state

    if hasattr(state, "cart"):
        cart = state.cart
        if isinstance(cart, CartState):
            return cart

    if isinstance(state, dict) and "cart" in state:
        cart = state["cart"]
        if isinstance(cart, CartState):
            return cart

    raise ValueError(
        "Could not find CartState in the provided state object. "
        "Ensure state has a .cart attribute or is a CartState itself."
    )


def _get_products_by_id(state: Any) -> dict[str, Any]:
    """Extract products_by_id mapping from the episode state.

    Supports:
        - state.products_by_id (attribute)
        - state["products_by_id"] (dict)
        - state.catalog_state.products_by_id (nested via CatalogState)

    Raises:
        ValueError: If no products mapping can be found.
    """
    if hasattr(state, "products_by_id"):
        return state.products_by_id

    if isinstance(state, dict) and "products_by_id" in state:
        return state["products_by_id"]

    # Try via catalog_state
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


# ---------------------------------------------------------------------------
# Tool handler: cart.view
# ---------------------------------------------------------------------------


def cart_view(*, state: Any = None) -> dict[str, Any]:
    """View current cart contents.

    Returns:
        Dict with:
            - lines: list of cart line dicts
            - total_items: total number of items (sum of quantities)
            - total_price: total price in USD
    """
    cart = _get_cart(state)
    return _build_cart_view(cart)


# ---------------------------------------------------------------------------
# Tool handler: cart.add
# ---------------------------------------------------------------------------


def cart_add(
    product_id: str,
    variant_id: str | None = None,
    quantity: int = 1,
    *,
    state: Any = None,
) -> dict[str, Any]:
    """Add a product to the cart.

    Validates that the product exists and is in stock. If the same
    (product_id, variant_id) combination already exists in the cart,
    increments the existing line's quantity instead of adding a new line.

    Args:
        product_id: Product identifier.
        variant_id: Optional variant identifier.
        quantity:   Number of units to add (default: 1).
        state:      Episode state with .cart and .products_by_id.

    Returns:
        Updated cart view dict.

    Raises:
        ValueError: If product not found or insufficient stock.
    """
    cart = _get_cart(state)
    products_by_id = _get_products_by_id(state)

    # Validate product exists
    product = products_by_id.get(product_id)
    if product is None:
        raise ValueError(f"Product '{product_id}' not found")

    # Validate stock (product may be a Product model or a dict)
    stock_qty: int
    unit_price: float
    if hasattr(product, "stock_qty"):
        stock_qty = product.stock_qty
        unit_price = product.price
    else:
        stock_qty = product.get("stock_qty", 0)
        unit_price = product.get("price", 0.0)

    if stock_qty < quantity:
        raise ValueError(
            f"Insufficient stock for product '{product_id}': "
            f"requested {quantity}, available {stock_qty}"
        )

    before = cart._snapshot() if cart.trace_mode else []

    # Check if same (product_id, variant_id) already in cart
    for line in cart.lines:
        if line.product_id == product_id and line.variant_id == variant_id:
            # Check total quantity against stock
            new_qty = line.qty + quantity
            if new_qty > stock_qty:
                raise ValueError(
                    f"Insufficient stock for product '{product_id}': "
                    f"requested total {new_qty} (existing {line.qty} + {quantity}), "
                    f"available {stock_qty}"
                )
            line.qty = new_qty

            after = cart._snapshot() if cart.trace_mode else []
            cart._log_mutation(
                "cart.add",
                before,
                after,
                f"incremented qty for {product_id} (variant={variant_id}) to {new_qty}",
            )
            return _build_cart_view(cart)

    # Add new line
    line_id = cart._generate_line_id()
    new_line = CartLine(
        line_id=line_id,
        product_id=product_id,
        variant_id=variant_id,
        qty=quantity,
        unit_price=unit_price,
    )
    cart.lines.append(new_line)

    after = cart._snapshot() if cart.trace_mode else []
    cart._log_mutation(
        "cart.add",
        before,
        after,
        f"added new line {line_id}: {product_id} (variant={variant_id}) x{quantity}",
    )

    return _build_cart_view(cart)


# ---------------------------------------------------------------------------
# Tool handler: cart.remove
# ---------------------------------------------------------------------------


def cart_remove(
    line_id: str,
    *,
    state: Any = None,
) -> dict[str, Any]:
    """Remove a line item from the cart.

    Args:
        line_id: Line identifier to remove.
        state:   Episode state with .cart.

    Returns:
        Updated cart view dict.

    Raises:
        ValueError: If line_id not found in cart.
    """
    cart = _get_cart(state)

    before = cart._snapshot() if cart.trace_mode else []

    for i, line in enumerate(cart.lines):
        if line.line_id == line_id:
            removed = cart.lines.pop(i)

            after = cart._snapshot() if cart.trace_mode else []
            cart._log_mutation(
                "cart.remove",
                before,
                after,
                f"removed line {line_id}: {removed.product_id} x{removed.qty}",
            )
            return _build_cart_view(cart)

    raise ValueError(f"Line '{line_id}' not found in cart")


# ---------------------------------------------------------------------------
# Tool handler: cart.set_quantity
# ---------------------------------------------------------------------------


def cart_set_quantity(
    line_id: str,
    quantity: int,
    *,
    state: Any = None,
) -> dict[str, Any]:
    """Update the quantity for a cart line item.

    If quantity is set to 0, the line is removed entirely.

    Args:
        line_id:  Line identifier to update.
        quantity: New quantity (0 removes the line).
        state:    Episode state with .cart and .products_by_id.

    Returns:
        Updated cart view dict.

    Raises:
        ValueError: If line_id not found in cart or stock insufficient.
    """
    cart = _get_cart(state)

    before = cart._snapshot() if cart.trace_mode else []

    for i, line in enumerate(cart.lines):
        if line.line_id == line_id:
            if quantity == 0:
                removed = cart.lines.pop(i)

                after = cart._snapshot() if cart.trace_mode else []
                cart._log_mutation(
                    "cart.set_quantity",
                    before,
                    after,
                    f"removed line {line_id} (qty set to 0): {removed.product_id}",
                )
                return _build_cart_view(cart)

            # Validate stock for the new quantity
            products_by_id = _get_products_by_id(state)
            product = products_by_id.get(line.product_id)
            if product is not None:
                stock_qty: int
                if hasattr(product, "stock_qty"):
                    stock_qty = product.stock_qty
                else:
                    stock_qty = product.get("stock_qty", 0)

                if quantity > stock_qty:
                    raise ValueError(
                        f"Insufficient stock for product '{line.product_id}': "
                        f"requested {quantity}, available {stock_qty}"
                    )

            old_qty = line.qty
            line.qty = quantity

            after = cart._snapshot() if cart.trace_mode else []
            cart._log_mutation(
                "cart.set_quantity",
                before,
                after,
                f"updated line {line_id} qty: {old_qty} -> {quantity}",
            )
            return _build_cart_view(cart)

    raise ValueError(f"Line '{line_id}' not found in cart")


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_cart_tools(registry: Any) -> None:
    """Register all cart tool handlers with a ToolRegistry instance.

    Registers:
        - cart.view
        - cart.add
        - cart.remove
        - cart.set_quantity

    Args:
        registry: ToolRegistry instance.
    """
    registry.register("cart.view", cart_view, CartViewArgs)
    registry.register("cart.add", cart_add, CartAddArgs)
    registry.register("cart.remove", cart_remove, CartRemoveArgs)
    registry.register("cart.set_quantity", cart_set_quantity, CartSetQuantityArgs)

    logger.info("Registered 4 cart tools: view, add, remove, set_quantity")
