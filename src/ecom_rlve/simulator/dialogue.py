"""Deterministic dialogue manager for EcomRLVE-GYM user simulator.

Manages the user side of a multi-turn conversation with the LLM agent.
Tracks dialogue state (slots provided, clarifications, satisfaction)
and generates user responses based on templates and goal parameters.

Key behaviors:
    - Initial message: rendered from templates with optional slot omission
    - Clarification: when agent asks for missing info, provide it
    - Ragequit: if satisfaction is poor for 2+ consecutive turns, or T > T_patience
    - Deterministic mode: skip noise, always use template index 0
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class UserAct(Enum):
    """Structured dialogue act indicating *why* the user simulator responded.

    Used by the reward composer to distinguish user-driven turns (which
    should not penalise the agent's efficiency) from agent-error turns
    (which should).
    """

    CONFIRM = "confirm"          # User confirms the agent's action is correct
    CLARIFY = "clarify"          # User provides previously omitted info (agent asked)
    CORRECT = "correct"          # User points out an agent mistake (wrong item/variant/qty)
    ELABORATE = "elaborate"      # User adds new requirements not in original message
    CONTINUE = "continue"        # Generic acknowledgement / conversation continuation
    RAGEQUIT = "ragequit"        # User abandons the conversation
    DONE = "done"                # User confirms task is complete
    DISSATISFIED = "dissatisfied"  # User expresses low satisfaction

from ecom_rlve.simulator.persona import PersonaWeights
from ecom_rlve.simulator.templates import (
    apply_noise,
    render_clarification,
    render_template,
    render_template_deterministic,
)
from ecom_rlve.simulator.llm_backend import (
    build_user_system_prompt,
    detect_clarification_with_llm,
    generate_dialogue_response,
    generate_clarification_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dialogue state
# ---------------------------------------------------------------------------


@dataclass
class DialogueState:
    """Tracks the state of a user-agent dialogue.

    Attributes:
        turn_count:           Number of turns completed (incremented after each exchange).
        provided_slots:       Set of slot names the user has communicated.
        pending_slots:        Set of slot names deliberately omitted (for clarification).
        clarification_count:  Number of clarification responses given.
        satisfaction_history: List of constraint satisfaction scores per turn [0, 1].
        ragequit:             Whether the user has rage-quit the conversation.
    """

    turn_count: int = 0
    provided_slots: set[str] = field(default_factory=set)
    pending_slots: set[str] = field(default_factory=set)
    clarification_count: int = 0
    satisfaction_history: list[float] = field(default_factory=list)
    ragequit: bool = False


# ---------------------------------------------------------------------------
# Satisfaction thresholds
# ---------------------------------------------------------------------------

_SATISFACTION_THRESHOLD: float = 0.3
"""Satisfaction score below which the user considers progress poor."""

_RAGEQUIT_CONSECUTIVE_BAD: int = 2
"""Number of consecutive below-threshold turns before ragequit."""


# ---------------------------------------------------------------------------
# Response generation helpers
# ---------------------------------------------------------------------------

_RAGEQUIT_MESSAGES: list[str] = [
    "This isn't working. I'll look elsewhere.",
    "I'm not finding what I need. Thanks anyway.",
    "I think I'll try a different store.",
    "This is taking too long. Goodbye.",
    "Never mind, I'll figure it out myself.",
]

_SATISFACTION_MESSAGES: list[str] = [
    "That's not quite what I'm looking for.",
    "Hmm, these don't match my requirements well.",
    "Can you try again with different options?",
    "Those suggestions aren't great. Let me clarify what I need.",
    "I was hoping for something more specific.",
]

_CONTINUATION_MESSAGES: list[str] = [
    "Thanks, let me take a look at these.",
    "Okay, these look interesting.",
    "Let me review these options.",
    "Thanks for the suggestions.",
    "Got it, let me consider these.",
]

_DONE_MESSAGES: list[str] = [
    "Great, that's what I needed. Thanks!",
    "Perfect, thank you for the help!",
    "That works, thanks!",
    "Looks good, I appreciate the assistance.",
]


# ---------------------------------------------------------------------------
# Slot detection heuristics
# ---------------------------------------------------------------------------

_CLARIFICATION_KEYWORDS: dict[str, list[str]] = {
    "brand_pref": [
        "brand", "manufacturer", "which brand", "what brand",
        "particular brand", "brand name", "make",
    ],
    "color_pref": [
        "color", "colour", "what color", "which color",
        "shade", "hue", "color preference",
    ],
    "rating_req": [
        "rating", "rated", "reviews", "minimum rating", "stars",
        "review score", "customer rating", "how well rated",
    ],
    "ship_req": [
        "shipping", "delivery", "arrive", "ship", "how fast", "when",
        "deliver by", "shipping speed", "how soon", "timeline",
        "need it by", "urgency", "rush",
    ],
    "price_range": [
        "price", "budget", "cost", "how much", "price range",
        "spend", "afford", "willing to pay", "max price",
    ],
    "size_pref": [
        "size", "what size", "which size",
        "dimensions", "how big", "how large", "small or large",
    ],
    "reason": [
        "reason", "why", "what's wrong", "issue",
        "problem", "what happened", "defective", "damaged",
    ],
    "replacement_req": [
        "replacement", "exchange", "substitute", "instead",
        "swap", "different one", "alternative",
    ],
    "order_ref": [
        "order", "order number", "order id", "which order",
        "order reference", "order details", "purchase",
    ],
    "variant_details": [
        "variant", "option", "which version", "specific",
        "color", "size", "configuration", "model",
        "which one", "preference", "style", "type",
        "particular", "choose",
    ],
    "quantity_details": [
        "quantity", "how many", "how much", "amount",
        "number", "count", "one or", "several",
        "units", "pieces", "each",
    ],
    "budget": [
        "budget", "total budget", "spend", "maximum",
        "total cost", "price limit", "afford",
    ],
    "material_pref": [
        "material", "made of", "fabric", "composition",
        "what material", "which material",
    ],
}


def _detect_clarification_request(
    assistant_message: str,
    pending_slots: set[str],
    seed: int = 0,
    system_prompt: str | None = None,
    use_llm_fallback: bool = True,
) -> str | None:
    """Detect if the assistant is asking for clarification on a pending slot.

    First tries keyword matching, then falls back to LLM-based detection
    if keywords miss. This handles paraphrases like 'do you want one or two?'
    that keyword lists can't cover.

    Args:
        assistant_message: The assistant's message text.
        pending_slots:     Set of slot names that were deliberately omitted.
        seed:              Random seed for LLM fallback.
        system_prompt:     Optional system prompt for LLM fallback context.
        use_llm_fallback:  Whether to try LLM detection when keywords fail.

    Returns:
        The slot name being asked about, or None if no clarification detected.
    """
    lower_msg = assistant_message.lower()

    for slot_name in pending_slots:
        keywords = _CLARIFICATION_KEYWORDS.get(slot_name, [])
        for keyword in keywords:
            if keyword in lower_msg:
                return slot_name

    # Also check for generic clarification patterns
    generic_patterns = [
        r"could you (?:please )?(?:specify|tell me|provide|share)",
        r"what (?:is|are) your (?:prefer|requirement)",
        r"do you have (?:a|any) preference",
        r"can you (?:be more specific|clarify|elaborate)",
        r"any (?:specific|particular) (?:preference|requirement)",
        r"which (?:one|version|model|type|option)",
        r"how many (?:do you|would you|of)",
        r"(?:one|1) or (?:more|two|2|several)",
    ]
    for pattern in generic_patterns:
        if re.search(pattern, lower_msg):
            # Return the first pending slot
            if pending_slots:
                return next(iter(pending_slots))

    # LLM-based fallback: ask the LLM to determine if a slot is being asked
    if use_llm_fallback and pending_slots:
        llm_result = detect_clarification_with_llm(
            assistant_message=assistant_message,
            pending_slots=pending_slots,
            seed=seed,
            system_prompt=system_prompt,
        )
        if llm_result is not None:
            return llm_result

    return None


# ---------------------------------------------------------------------------
# User Simulator
# ---------------------------------------------------------------------------


class UserSimulator:
    """Deterministic user simulator for EcomRLVE-GYM dialogues.

    Manages the user side of a multi-turn conversation. Generates an initial
    message from templates, responds to clarification requests by providing
    previously omitted information, and may ragequit if satisfaction is
    consistently poor.

    Attributes:
        persona_weights: Persona preference weights.
        goal_params:     Goal parameters (slot values for templates).
        env_id:          Environment identifier (PD, SUB, etc.).
        p_missing:       Probability of omitting non-critical slots.
        p_noise:         Probability of character-level noise.
        T_patience:      Maximum turns before forced ragequit (None = no limit).
        seed:            Random seed for reproducibility.
        deterministic_mode: Class-level debug lever. When True, skip noise
                           and always use template index 0.

    Example:
        >>> from ecom_rlve.simulator.persona import sample_persona_weights
        >>> weights = sample_persona_weights(seed=42)
        >>> sim = UserSimulator(
        ...     persona_weights=weights,
        ...     goal_params={"category": "headphones", "price_max": "100"},
        ...     env_id="PD",
        ...     p_missing=0.3,
        ...     p_noise=0.05,
        ...     seed=42,
        ... )
        >>> msg = sim.generate_initial_message()
        >>> sim.get_state().turn_count
        0
    """

    deterministic_mode: bool = False
    """Class-level debug lever. When True, always use template index 0 and skip noise."""

    def __init__(
        self,
        persona_weights: PersonaWeights,
        goal_params: dict[str, Any],
        env_id: str,
        p_missing: float,
        p_noise: float,
        T_patience: int | None = None,
        seed: int = 42,
        goal_summary: str = "",
        persona_summary: str = "",
    ) -> None:
        self.persona_weights = persona_weights
        self.goal_params = dict(goal_params)  # defensive copy
        self.env_id = env_id
        self.p_missing = p_missing
        self.p_noise = p_noise
        self.T_patience = T_patience
        self.seed = seed
        self.goal_summary = goal_summary
        self.persona_summary = persona_summary

        # Build env-aware system prompt for all LLM calls
        if goal_summary:
            self._system_prompt: str | None = build_user_system_prompt(
                env_id=env_id,
                goal_summary=goal_summary,
                persona_summary=persona_summary,
            )
        else:
            self._system_prompt = None

        self._state = DialogueState()
        self._seed_counter: int = seed
        self._initial_message_generated: bool = False

    def _next_seed(self) -> int:
        """Generate a deterministic incrementing seed for each random operation."""
        self._seed_counter += 1
        return self._seed_counter

    # ------------------------------------------------------------------
    # Initial message
    # ------------------------------------------------------------------

    def generate_initial_message(self) -> str:
        """Generate the first user message to start the dialogue.

        Uses templates to render an initial utterance, optionally omitting
        non-critical slots (tracked as pending_slots for later clarification).

        Returns:
            The initial user message string.
        """
        if self._initial_message_generated:
            logger.warning("generate_initial_message() called more than once")

        if self.deterministic_mode:
            message = render_template_deterministic(self.env_id, self.goal_params)
            # In deterministic mode, all slots are provided
            self._state.provided_slots = set(self.goal_params.keys())
            self._state.pending_slots = set()
        else:
            # Track which slots were omitted
            # We do this by comparing the rendered output with a full render
            import random as _rand
            pre_rng = _rand.Random(self._next_seed())

            # Determine which non-critical slots will be omitted
            omitted: set[str] = set()
            provided: set[str] = set()
            for key in self.goal_params:
                if key in _CLARIFICATION_KEYWORDS:
                    # This is a non-critical slot
                    if pre_rng.random() < self.p_missing:
                        omitted.add(key)
                    else:
                        provided.add(key)
                else:
                    provided.add(key)

            self._state.pending_slots = omitted
            self._state.provided_slots = provided

            message = render_template(
                env_id=self.env_id,
                params=self.goal_params,
                p_missing=self.p_missing,
                p_noise=self.p_noise,
                seed=self._next_seed(),
            )

        self._initial_message_generated = True
        return message

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    def generate_response(
        self,
        assistant_message: str,
        tool_results: list[Any],
        progress_info: dict[str, Any] | None = None,
    ) -> tuple[str, bool, UserAct]:
        """Generate a user response to the assistant's latest message.

        Logic:
            1. If the assistant is requesting clarification on a pending slot,
               provide the missing information.
            2. If progress_info is provided, track satisfaction and check
               for ragequit conditions.
            3. For CART env, check cart state against ground truth and
               provide specific feedback on issues.
            4. If T > T_patience, ragequit.
            5. Otherwise, generate a continuation message.

        Args:
            assistant_message: The assistant's most recent message text.
            tool_results:      List of tool results from the assistant's actions.
            progress_info:     Optional dict with 'satisfaction' (float in [0,1]),
                               'done' (bool), and/or 'cart_issues' (list[str]) keys.

        Returns:
            Tuple of (user_message, should_quit, user_act):
                - user_message: The user's response string.
                - should_quit: True if the user wants to end the conversation
                  (ragequit or task completion).
                - user_act: Structured UserAct indicating why the user responded.
        """
        import random as _rand

        self._state.turn_count += 1
        rng = _rand.Random(self._next_seed())

        # Extract progress signals
        satisfaction: float | None = None
        cart_issues: list[str] | None = None
        cart_candidates: list[dict[str, Any]] | None = None
        return_disambiguation: dict[str, Any] | None = None
        if progress_info is not None:
            if progress_info.get("done", False):
                if not self.deterministic_mode:
                    llm_msg = generate_dialogue_response(
                        context="done",
                        assistant_message=assistant_message,
                        seed=self._next_seed(),
                        goal_summary=self.goal_summary,
                        system_prompt=self._system_prompt,
                    )
                    msg = llm_msg if llm_msg is not None else rng.choice(_DONE_MESSAGES)
                else:
                    msg = _DONE_MESSAGES[0]
                return msg, True, UserAct.DONE

            satisfaction = progress_info.get("satisfaction")
            cart_issues = progress_info.get("cart_issues")
            cart_candidates = progress_info.get("cart_candidates")
            return_disambiguation = progress_info.get("return_disambiguation")

        if cart_candidates is not None:
            # Cart candidates take priority over other flows
            pass  # handled below after ragequit check

        if satisfaction is not None:
            self._state.satisfaction_history.append(satisfaction)

        # Check ragequit: T > T_patience
        if self.T_patience is not None and self._state.turn_count >= self.T_patience:
            self._state.ragequit = True
            msg_idx = 0 if self.deterministic_mode else rng.randint(0, len(_RAGEQUIT_MESSAGES) - 1)
            msg = _RAGEQUIT_MESSAGES[msg_idx]
            logger.debug(
                "UserSimulator ragequit: turn %d >= T_patience %d",
                self._state.turn_count,
                self.T_patience,
            )
            return msg, True, UserAct.RAGEQUIT

        # Check ragequit: consecutive low satisfaction
        if len(self._state.satisfaction_history) >= _RAGEQUIT_CONSECUTIVE_BAD:
            recent = self._state.satisfaction_history[-_RAGEQUIT_CONSECUTIVE_BAD:]
            if all(s < _SATISFACTION_THRESHOLD for s in recent):
                self._state.ragequit = True
                msg_idx = (
                    0
                    if self.deterministic_mode
                    else rng.randint(0, len(_RAGEQUIT_MESSAGES) - 1)
                )
                msg = _RAGEQUIT_MESSAGES[msg_idx]
                logger.debug(
                    "UserSimulator ragequit: %d consecutive turns below threshold %.2f. "
                    "Recent scores: %s",
                    _RAGEQUIT_CONSECUTIVE_BAD,
                    _SATISFACTION_THRESHOLD,
                    recent,
                )
                return msg, True, UserAct.RAGEQUIT

        # --- Cart candidate confirmation ---
        # When the agent presents product options, the user confirms which
        # items match their intent based on the hidden target product IDs.
        if cart_candidates:
            if not self.deterministic_mode:
                # Build a natural confirmation message via LLM
                titles = [c.get("title", "that item") for c in cart_candidates]
                qty_parts = []
                for c in cart_candidates:
                    q = c.get("qty", 1)
                    t = c.get("title", "it")
                    if q > 1:
                        qty_parts.append(f"{q} of the {t}")
                    else:
                        qty_parts.append(f"the {t}")

                if len(titles) == 1:
                    confirm_hint = f"Yes, I want {qty_parts[0]}."
                else:
                    confirm_hint = "Yes, I want " + ", ".join(qty_parts[:-1]) + f" and {qty_parts[-1]}."

                llm_msg = generate_dialogue_response(
                    context="cart_confirm",
                    assistant_message=assistant_message,
                    seed=self._next_seed(),
                    goal_summary=self.goal_summary,
                    system_prompt=self._system_prompt,
                    cart_issues=[],  # reuse the cart_issues field
                    confirm_hint=confirm_hint,
                )
                msg = llm_msg if llm_msg is not None else confirm_hint
            else:
                titles = [c.get("title", "item") for c in cart_candidates]
                msg = "Yes, add " + ", ".join(titles) + " to my cart."

            if self.p_noise > 0.0:
                noise_rng = _rand.Random(self._next_seed())
                msg = apply_noise(msg, self.p_noise, noise_rng)

            return msg, False, UserAct.CONFIRM

        # Check if assistant is asking for clarification on a pending slot
        if self._state.pending_slots:
            requested_slot = _detect_clarification_request(
                assistant_message,
                self._state.pending_slots,
                seed=self._next_seed(),
                system_prompt=self._system_prompt,
                use_llm_fallback=not self.deterministic_mode,
            )
            if requested_slot is not None and requested_slot in self.goal_params:
                # Provide the missing information
                self._state.pending_slots.discard(requested_slot)
                self._state.provided_slots.add(requested_slot)
                self._state.clarification_count += 1

                if self.deterministic_mode:
                    msg = f"{requested_slot}: {self.goal_params[requested_slot]}"
                else:
                    llm_msg = generate_clarification_response(
                        slot_name=requested_slot,
                        slot_value=self.goal_params[requested_slot],
                        assistant_message=assistant_message,
                        seed=self._next_seed(),
                        system_prompt=self._system_prompt,
                    )
                    if llm_msg is not None:
                        msg = llm_msg
                    else:
                        msg = render_clarification(
                            slot_name=requested_slot,
                            slot_value=self.goal_params[requested_slot],
                            seed=self._next_seed(),
                        )
                    if self.p_noise > 0.0:
                        noise_rng = _rand.Random(self._next_seed())
                        msg = apply_noise(msg, self.p_noise, noise_rng)

                return msg, False, UserAct.CLARIFY

        # --- Cart ground-truth feedback ---
        # If the env provided cart_issues, give specific feedback about
        # what's wrong with the cart instead of generic messages.
        if cart_issues is not None:
            if cart_issues:  # there are actual issues
                if not self.deterministic_mode:
                    llm_msg = generate_dialogue_response(
                        context="cart_feedback",
                        assistant_message=assistant_message,
                        seed=self._next_seed(),
                        goal_summary=self.goal_summary,
                        system_prompt=self._system_prompt,
                        cart_issues=cart_issues,
                    )
                    if llm_msg is not None:
                        msg = llm_msg
                    else:
                        # Template fallback: list issues directly
                        msg = "Hold on, " + cart_issues[0] + "."
                else:
                    msg = "That's not right. " + cart_issues[0] + "."
                return msg, False, UserAct.CORRECT
            else:
                # Cart is correct — positive acknowledgment
                if not self.deterministic_mode:
                    llm_msg = generate_dialogue_response(
                        context="cart_feedback",
                        assistant_message=assistant_message,
                        seed=self._next_seed(),
                        goal_summary=self.goal_summary,
                        system_prompt=self._system_prompt,
                        cart_issues=[],  # empty = cart is good
                    )
                    msg = llm_msg if llm_msg is not None else "That looks right, thanks!"
                else:
                    msg = "That looks right, thanks!"
                return msg, False, UserAct.CONFIRM

        # --- RETURN order disambiguation ---
        # When the agent lists orders and asks "which order?", the user sim
        # uses the hidden goal to identify the correct order and respond.
        if return_disambiguation is not None:
            target_title = return_disambiguation.get("target_product_title", "")
            target_order_id = return_disambiguation.get("target_order_id", "")
            assistant_lower = assistant_message.lower()

            # Check if the assistant is asking which order / presenting options
            is_asking_order = any(
                kw in assistant_lower
                for kw in [
                    "which order", "which one", "can you confirm",
                    "found", "i see", "order", "multiple",
                    "help me identify", "could you clarify",
                ]
            )

            if is_asking_order:
                if not self.deterministic_mode:
                    llm_msg = generate_dialogue_response(
                        context="continue",
                        assistant_message=assistant_message,
                        seed=self._next_seed(),
                        goal_summary=self.goal_summary,
                        system_prompt=self._system_prompt,
                    )
                    if llm_msg is not None:
                        msg = llm_msg
                    else:
                        msg = f"It's the {target_title} — that should be from order {target_order_id}."
                else:
                    msg = f"It's the {target_title}, order {target_order_id}."

                # Provide the order_ref slot if it was pending
                if "order_ref" in self._state.pending_slots:
                    self._state.pending_slots.discard("order_ref")
                    self._state.provided_slots.add("order_ref")

                if self.p_noise > 0.0:
                    noise_rng = _rand.Random(self._next_seed())
                    msg = apply_noise(msg, self.p_noise, noise_rng)

                return msg, False, UserAct.CLARIFY

        # Low satisfaction but not ragequit yet: express dissatisfaction
        if satisfaction is not None and satisfaction < _SATISFACTION_THRESHOLD:
            if not self.deterministic_mode:
                llm_msg = generate_dialogue_response(
                    context="dissatisfied",
                    assistant_message=assistant_message,
                    seed=self._next_seed(),
                    goal_summary=self.goal_summary,
                    system_prompt=self._system_prompt,
                )
                msg = llm_msg if llm_msg is not None else rng.choice(_SATISFACTION_MESSAGES)
            else:
                msg = _SATISFACTION_MESSAGES[0]
            return msg, False, UserAct.DISSATISFIED

        # Default: acknowledge and continue
        if not self.deterministic_mode:
            llm_msg = generate_dialogue_response(
                context="continue",
                assistant_message=assistant_message,
                seed=self._next_seed(),
                goal_summary=self.goal_summary,
                system_prompt=self._system_prompt,
            )
            msg = llm_msg if llm_msg is not None else rng.choice(_CONTINUATION_MESSAGES)
        else:
            msg = _CONTINUATION_MESSAGES[0]
        return msg, False, UserAct.CONTINUE

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_state(self) -> DialogueState:
        """Return the current dialogue state.

        Returns:
            A copy of the current DialogueState.
        """
        return DialogueState(
            turn_count=self._state.turn_count,
            provided_slots=set(self._state.provided_slots),
            pending_slots=set(self._state.pending_slots),
            clarification_count=self._state.clarification_count,
            satisfaction_history=list(self._state.satisfaction_history),
            ragequit=self._state.ragequit,
        )
