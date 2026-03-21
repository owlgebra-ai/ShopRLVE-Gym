"""User-facing tools for EcomRLVE-GYM.

Provides a ``user.get_visit_history`` tool that returns a list of products
the customer has recently browsed/visited.  This list always includes the
target items the user *actually* wants (hidden from the agent) mixed with
distractor products so the agent must cross-reference visit history with
catalog search results and user descriptions to identify the correct items.

Design decisions:
    - The visit history is generated during ``generate_problem()`` in
      ``CartBuildingEnv`` and stored in ``extra["visit_history"]``.
    - Each entry is a lightweight card: product_id, title, price, category,
      brand, key_attrs.  This mirrors what a real e-commerce "recently
      viewed" page would show.
    - The handler reads the list from episode state and returns it as-is.
    - The tool is stateless (no mutation).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic arg model
# ---------------------------------------------------------------------------


class UserGetVisitHistoryArgs(BaseModel):
    """Arguments for user.get_visit_history tool (no args required)."""

    pass


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------


def user_get_visit_history(*, state: Any = None) -> list[dict[str, Any]]:
    """Return the customer's recently viewed products.

    The agent should call this tool to see which products the user has
    recently browsed.  Cross-referencing this with catalog search results
    and the user's natural-language request helps narrow down the correct
    items to add to cart.

    Args:
        state: Episode state with ``visit_history`` attribute or key.

    Returns:
        List of product cards the user has recently viewed.  Each card
        contains: product_id, title, price, category, brand, key_attrs.
    """
    # Try attribute access (EpisodeState dataclass)
    if hasattr(state, "visit_history") and state.visit_history:
        return list(state.visit_history)

    # Try dict access (fallback)
    if isinstance(state, dict) and "visit_history" in state:
        return list(state["visit_history"])

    # No visit history available
    logger.debug("user.get_visit_history called but no visit_history in state")
    return []


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_user_tools(registry: Any) -> None:
    """Register user-facing tools with a ToolRegistry instance.

    Args:
        registry: ToolRegistry instance.
    """
    registry.register(
        "user.get_visit_history",
        user_get_visit_history,
        UserGetVisitHistoryArgs,
    )
    logger.info("Registered 1 user tool: user.get_visit_history")
