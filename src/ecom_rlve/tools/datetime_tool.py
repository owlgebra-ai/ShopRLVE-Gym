"""Date/time utility tool for EcomRLVE-GYM.

Provides a ``datetime.now`` tool that returns the current simulated date
and time.  The agent calls this to reason about return eligibility windows,
order recency, and shipping ETAs.

The "current" date is read from the episode state (``state.today``) so that
all time-based decisions are deterministic and reproducible.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic arg model (empty — no arguments needed)
# ---------------------------------------------------------------------------


class DatetimeNowArgs(BaseModel):
    """Arguments for datetime.now tool (no args required)."""

    pass


# ---------------------------------------------------------------------------
# State accessor
# ---------------------------------------------------------------------------


def _get_today(state: Any) -> str:
    """Extract reference date from state, defaulting to today."""
    if hasattr(state, "today") and state.today:
        return state.today

    if isinstance(state, dict) and "today" in state:
        return state["today"]

    return date.today().isoformat()


# ---------------------------------------------------------------------------
# Tool handler: datetime.now
# ---------------------------------------------------------------------------


def datetime_now(*, state: Any = None) -> dict[str, str]:
    """Return the current simulated date and time.

    The agent should call this tool to discover what "today" is before
    reasoning about return windows, order ages, or shipping ETAs.

    Args:
        state: Episode state with a ``.today`` field (ISO date string).

    Returns:
        Dict with:
            - date: str (ISO date, e.g. "2026-03-15")
            - day_of_week: str (e.g. "Sunday")
            - time: str (simulated time, always "12:00:00" for determinism)
    """
    today_str = _get_today(state)
    try:
        today_date = date.fromisoformat(today_str)
    except (ValueError, TypeError):
        today_date = date.today()

    return {
        "date": today_date.isoformat(),
        "day_of_week": today_date.strftime("%A"),
        "time": "12:00:00",
    }


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------


def register_datetime_tools(registry: Any) -> None:
    """Register the datetime.now tool with a ToolRegistry instance.

    Args:
        registry: ToolRegistry instance.
    """
    registry.register("datetime.now", datetime_now, DatetimeNowArgs)
    logger.info("Registered 1 datetime tool: datetime.now")
