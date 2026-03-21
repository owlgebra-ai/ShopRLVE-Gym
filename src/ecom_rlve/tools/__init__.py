"""EcomRLVE-GYM tool layer: registry, validation, dispatch, and tool implementations.

This package provides the simulated tool API that the LLM calls during episodes:

- **registry**: ToolCall/ToolResult models, ToolRegistry for registering,
  validating, and dispatching tool handlers.
- **catalog**: catalog.search, catalog.rerank, catalog.get_product,
  catalog.get_variants implementations with FAISS retrieval and
  difficulty-aware degradation.
- **cart**: cart.view, cart.add, cart.remove, cart.set_quantity -- in-memory
  cart with deterministic operations.
- **orders**: order.list, order.get_status, order.checkout -- order history
  and checkout, plus synthetic order generation.
- **returns**: return.check_eligibility, return.initiate, return.exchange --
  category-aware return and exchange engine.
- **policy**: policy.search -- deterministic policy knowledge base with
  keyword-based retrieval.
- **datetime_tool**: datetime.now -- current simulated date/time for
  the agent to reason about return windows and order ages.
"""

from ecom_rlve.tools.cart import (
    CartAddArgs,
    CartLine,
    CartRemoveArgs,
    CartSetQuantityArgs,
    CartState,
    CartViewArgs,
    cart_add,
    cart_remove,
    cart_set_quantity,
    cart_view,
    register_cart_tools,
)
from ecom_rlve.tools.catalog import (
    CatalogGetProductArgs,
    CatalogGetVariantsArgs,
    CatalogRerankArgs,
    CatalogSearchArgs,
    CatalogState,
    catalog_get_product,
    catalog_get_variants,
    catalog_rerank,
    catalog_search,
    register_catalog_tools,
)
from ecom_rlve.tools.orders import (
    Order,
    OrderCheckoutArgs,
    OrderGetStatusArgs,
    OrderLine,
    OrderListArgs,
    generate_order_history,
    order_checkout,
    order_get_status,
    order_list,
    register_order_tools,
)
from ecom_rlve.tools.policy import (
    PolicyKB,
    PolicyRule,
    PolicySearchArgs,
    build_default_policy_kb,
    generate_policy_question,
    policy_search,
    register_policy_tools,
)
from ecom_rlve.tools.registry import (
    TOOL_SCHEMA,
    ToolCall,
    ToolRegistry,
    ToolResult,
)
from ecom_rlve.tools.datetime_tool import (
    DatetimeNowArgs,
    datetime_now,
    register_datetime_tools,
)
from ecom_rlve.tools.user import (
    UserGetVisitHistoryArgs,
    user_get_visit_history,
    register_user_tools,
)
from ecom_rlve.tools.returns import (
    RETURN_FEES,
    RETURN_METHODS,
    RETURN_WINDOWS,
    VALID_REASON_CODES,
    ReturnCheckEligibilityArgs,
    ReturnExchangeArgs,
    ReturnInitiateArgs,
    return_check_eligibility,
    return_exchange,
    return_initiate,
    register_return_tools,
)

__all__ = [
    # Registry
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
    "TOOL_SCHEMA",
    # Catalog tools
    "CatalogSearchArgs",
    "CatalogRerankArgs",
    "CatalogGetProductArgs",
    "CatalogGetVariantsArgs",
    "CatalogState",
    "catalog_search",
    "catalog_rerank",
    "catalog_get_product",
    "catalog_get_variants",
    "register_catalog_tools",
    # Cart tools
    "CartLine",
    "CartState",
    "CartViewArgs",
    "CartAddArgs",
    "CartRemoveArgs",
    "CartSetQuantityArgs",
    "cart_view",
    "cart_add",
    "cart_remove",
    "cart_set_quantity",
    "register_cart_tools",
    # Order tools
    "OrderLine",
    "Order",
    "OrderListArgs",
    "OrderGetStatusArgs",
    "OrderCheckoutArgs",
    "generate_order_history",
    "order_list",
    "order_get_status",
    "order_checkout",
    "register_order_tools",
    # Return tools
    "RETURN_WINDOWS",
    "RETURN_METHODS",
    "RETURN_FEES",
    "VALID_REASON_CODES",
    "ReturnCheckEligibilityArgs",
    "ReturnInitiateArgs",
    "ReturnExchangeArgs",
    "return_check_eligibility",
    "return_initiate",
    "return_exchange",
    "register_return_tools",
    # Datetime tools
    "DatetimeNowArgs",
    "datetime_now",
    "register_datetime_tools",
    # User tools
    "UserGetVisitHistoryArgs",
    "user_get_visit_history",
    "register_user_tools",
    # Policy tools
    "PolicyRule",
    "PolicyKB",
    "PolicySearchArgs",
    "build_default_policy_kb",
    "generate_policy_question",
    "policy_search",
    "register_policy_tools",
]
