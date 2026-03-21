"""Episode state management for EcomRLVE-GYM (Spec Section 8.1).

Per active episode:
- env_id, difficulty, hidden_goal (ProblemParams), persona_weights
- products_by_id (dict), variants_by_product (dict)
- orders (list[Order]), cart (CartState)
- seen_product_ids (set) -- product IDs surfaced to model via tools
- conversation (list[dict]) -- [{role, content}]
- tool_results_history (list) -- all tool calls + results
- turn (int)
- done (bool), reward (float | None)

Also provides the Observation model (what the model sees), the ActionSchema
model (what the model outputs), and a parse_action helper.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, model_validator

from ecom_rlve.data.schema import Product, Variant
from ecom_rlve.rewards.composer import RewardBreakdown
from ecom_rlve.simulator.persona import PersonaWeights
from ecom_rlve.tools.cart import CartState
from ecom_rlve.tools.catalog import CatalogState
from ecom_rlve.tools.orders import Order
from ecom_rlve.tools.policy import PolicyKB
from ecom_rlve.tools.registry import ToolCall

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode state (mutable, internal to the server)
# ---------------------------------------------------------------------------


@dataclass
class EpisodeState:
    """Full mutable state for one active episode.

    Spec Section 8.1: per-episode state includes env_id, difficulty,
    hidden_goal, persona_weights, catalog_snapshot, orders, cart, Seen,
    turn counter T, and done/reward flags.

    This dataclass is *not* exposed to the model -- the model only sees
    :class:`Observation`.  This is the internal bookkeeping object used
    by :class:`~ecom_rlve.server.openenv.EcomRLVEEnv`.

    Attributes:
        env_id:              Environment identifier (PD, SUB, CART, ...).
        difficulty:          Sampled difficulty level for this episode.
        hidden_goal:         ProblemParams that define the verifiable goal.
        persona_weights:     Persona preference weights for the user simulator.
        products_by_id:      Snapshot of product_id -> Product for this episode.
        variants_by_product: Snapshot of product_id -> list[Variant].
        orders:              Order history for STATUS / RETURN envs.
        cart:                In-memory cart state.
        catalog_state:       CatalogState wired to tool handlers.
        policy_kb:           PolicyKB for E_POLICY episodes.
        seen_product_ids:    Product IDs surfaced to the model via tool results.
        conversation:        Full conversation as [{role, content}, ...].
        tool_results_history: All tool calls + results across the episode.
        turn:                Current turn counter (0-indexed; incremented after
                             each assistant turn).
        done:                Whether the episode has terminated.
        reward:              Final episode reward (set on termination).
        initiated_returns:   Set of return IDs initiated during E_RETURN.
        seed:                Episode-level random seed.
        reward_breakdown:    Decomposed reward (debug).
        timing:              Wall-clock timing measurements (debug).
    """

    # Core identifiers
    env_id: str = ""
    difficulty: int = 0
    hidden_goal: Any = None  # ProblemParams
    persona_weights: PersonaWeights | None = None

    # Catalog snapshot
    products_by_id: dict[str, Product] = field(default_factory=dict)
    variants_by_product: dict[str, list[Variant]] = field(default_factory=dict)

    # Transactional state
    orders: list[Order] = field(default_factory=list)
    cart: CartState = field(default_factory=CartState)
    catalog_state: CatalogState | None = None
    policy_kb: PolicyKB | None = None

    # Tracking
    seen_product_ids: set[str] = field(default_factory=set)
    conversation: list[dict[str, str]] = field(default_factory=list)
    tool_results_history: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    done: bool = False
    reward: float | None = None

    # E_RETURN specific
    initiated_returns: set[str] = field(default_factory=set)

    # E_CART specific: recently viewed products for user.get_visit_history
    visit_history: list[dict[str, Any]] = field(default_factory=list)

    # User act tracking: one entry per turn where user responded
    user_act_history: list[str] = field(default_factory=list)

    # Reproducibility
    seed: int = 42

    # Debug
    reward_breakdown: RewardBreakdown | None = None
    timing: dict[str, float] = field(default_factory=dict)

    # Reference date for order tools
    today: str = ""


# ---------------------------------------------------------------------------
# Observation (what the model sees at each step)
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """Observation returned to the model from reset() / step().

    Contains only the information the model is allowed to see.  The
    hidden goal, product catalog internals, and persona weights are
    never included here.

    Attributes:
        conversation:  Full conversation history up to this point.
        tool_results:  Tool call results from the most recent step
                       (empty after reset()).
        turn:          Current turn number.
        env_id:        Environment identifier (optionally disclosed to the
                       model for multi-env training).
        difficulty:    Difficulty level (optionally disclosed).
        done:          Whether the episode has terminated.
        reward:        Episode reward (populated only when done=True).
    """

    conversation: list[dict[str, str]] = Field(
        default_factory=list,
        description="Full conversation history [{role, content}]",
    )
    tool_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tool call results from the most recent assistant turn",
    )
    turn: int = Field(default=0, ge=0, description="Current turn number")
    env_id: str | None = Field(
        default=None,
        description="Environment identifier (optional, for multi-env training)",
    )
    difficulty: int | None = Field(
        default=None,
        description="Difficulty level (optional, for curriculum info)",
    )
    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: float | None = Field(
        default=None,
        description="Episode reward (populated only when done=True)",
    )


# ---------------------------------------------------------------------------
# Answer schema (structured answer portion of the LLM action)
# ---------------------------------------------------------------------------


class AnswerSchema(BaseModel):
    """Structured answer that the agent submits when it believes the task is done.

    Spec Section 2.1:
        "answer": {
            "env": "PD|SUB|CART|RETURN|STATUS|POLICY|BUNDLE|JOURNEY",
            "recommended_product_ids": ["p1", "p2"],
            "selected_order_id": "o123",
            "selected_line_id": "l2",
            "policy_answer": "string_or_number",
            "done": true
        }

    Attributes:
        env:                     Target environment identifier.
        recommended_product_ids: Product IDs the agent recommends (PD, SUB, BUNDLE).
        selected_order_id:       Order ID (STATUS, RETURN).
        selected_line_id:        Line ID within an order (RETURN).
        policy_answer:           Answer to a policy question (POLICY).
        done:                    Whether the agent considers the task complete.
    """

    env: str = Field(..., description="Target environment identifier")
    recommended_product_ids: list[str] = Field(
        default_factory=list,
        description="Product IDs the agent recommends",
    )
    selected_order_id: str | None = Field(
        default=None,
        description="Order ID for STATUS / RETURN environments",
    )
    selected_line_id: str | None = Field(
        default=None,
        description="Line ID within an order for RETURN",
    )
    policy_answer: str | float | None = Field(
        default=None,
        description="Answer to a policy question (POLICY env)",
    )
    done: bool = Field(default=False, description="Whether the agent considers the task complete")


# ---------------------------------------------------------------------------
# Action schema (full LLM action JSON)
# ---------------------------------------------------------------------------


class ActionSchema(BaseModel):
    """Complete LLM action parsed from the JSON output.

    Spec Section 2.1:
        {
            "assistant_message": "...",
            "tool_calls": [{"name": "catalog.search", "args": {...}}],
            "answer": { ... }
        }

    Attributes:
        assistant_message: The text message the agent displays to the user.
        tool_calls:        List of tool invocations (possibly empty).
        answer:            Optional structured answer (set when done).
    """

    assistant_message: str = Field(
        ..., min_length=1, description="Text message displayed to the user"
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool invocations (possibly empty)",
    )
    answer: AnswerSchema | None = Field(
        default=None,
        description="Structured answer (set when the agent believes the task is done)",
    )


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------


def parse_action(action_json: str) -> tuple[ActionSchema | None, bool]:
    """Parse the LLM's action JSON into an ActionSchema.

    Spec Section 8.3 step 1:
        Parse check -- invalid JSON or schema violation triggers
        done=True, reward=-1.

    The function attempts lenient parsing: it first tries strict
    pydantic validation, then falls back to basic dict parsing with
    manual field extraction.

    Args:
        action_json: Raw JSON string from the LLM.

    Returns:
        Tuple of (parsed_action, format_valid):
            - parsed_action: ActionSchema if parsing succeeded, None otherwise.
            - format_valid: True if the action was successfully parsed.
    """
    # Step 1: Parse raw JSON
    try:
        raw = json.loads(action_json)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.debug("parse_action: JSON decode failed: %s", exc)
        return None, False

    if not isinstance(raw, dict):
        logger.debug("parse_action: expected dict, got %s", type(raw).__name__)
        return None, False

    # Step 2: Try strict pydantic validation
    try:
        action = ActionSchema.model_validate(raw)
        return action, True
    except Exception as exc:
        logger.debug("parse_action: strict validation failed: %s", exc)

    # Step 3: Lenient fallback -- extract what we can
    try:
        assistant_message = str(raw.get("assistant_message", ""))
        if not assistant_message:
            logger.debug("parse_action: missing assistant_message")
            return None, False

        # Parse tool_calls
        raw_calls = raw.get("tool_calls", [])
        tool_calls: list[ToolCall] = []
        if isinstance(raw_calls, list):
            for tc in raw_calls:
                if isinstance(tc, dict) and "name" in tc:
                    tool_calls.append(ToolCall(
                        name=str(tc["name"]),
                        args=tc.get("args", {}),
                    ))

        # Parse answer
        answer: AnswerSchema | None = None
        raw_answer = raw.get("answer")
        if isinstance(raw_answer, dict):
            try:
                answer = AnswerSchema.model_validate(raw_answer)
            except Exception:
                # Try minimal extraction
                if "env" in raw_answer:
                    answer = AnswerSchema(
                        env=str(raw_answer["env"]),
                        recommended_product_ids=raw_answer.get("recommended_product_ids", []),
                        selected_order_id=raw_answer.get("selected_order_id"),
                        selected_line_id=raw_answer.get("selected_line_id"),
                        policy_answer=raw_answer.get("policy_answer"),
                        done=raw_answer.get("done", False),
                    )

        action = ActionSchema(
            assistant_message=assistant_message,
            tool_calls=tool_calls,
            answer=answer,
        )
        return action, True

    except Exception as exc:
        logger.debug("parse_action: lenient parsing also failed: %s", exc)
        return None, False
