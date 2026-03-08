"""Deterministic dialogue manager for ShopRLVE-GYM user simulator.

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
from typing import Any

from shop_rlve.simulator.persona import PersonaWeights
from shop_rlve.simulator.templates import (
    apply_noise,
    render_clarification,
    render_template,
    render_template_deterministic,
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
    "brand_pref": ["brand", "manufacturer", "which brand", "what brand"],
    "color_pref": ["color", "colour", "what color", "which color"],
    "rating_req": ["rating", "rated", "reviews", "minimum rating", "stars"],
    "ship_req": ["shipping", "delivery", "arrive", "ship", "how fast", "when"],
    "price_range": ["price", "budget", "cost", "how much", "price range"],
    "size_pref": ["size", "what size", "which size"],
    "reason": ["reason", "why", "what's wrong", "issue"],
    "replacement_req": ["replacement", "exchange", "substitute", "instead"],
    "variant_details": ["variant", "option", "which version", "specific"],
    "quantity_details": ["quantity", "how many", "how much", "amount"],
    "budget": ["budget", "total budget", "spend", "maximum"],
    "material_pref": ["material", "made of", "fabric"],
}


def _detect_clarification_request(
    assistant_message: str,
    pending_slots: set[str],
) -> str | None:
    """Detect if the assistant is asking for clarification on a pending slot.

    Scans the assistant message for keywords associated with each pending slot.

    Args:
        assistant_message: The assistant's message text.
        pending_slots:     Set of slot names that were deliberately omitted.

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
    ]
    for pattern in generic_patterns:
        if re.search(pattern, lower_msg):
            # Return the first pending slot
            if pending_slots:
                return next(iter(pending_slots))

    return None


# ---------------------------------------------------------------------------
# User Simulator
# ---------------------------------------------------------------------------


class UserSimulator:
    """Deterministic user simulator for ShopRLVE-GYM dialogues.

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
        >>> from shop_rlve.simulator.persona import sample_persona_weights
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
    ) -> None:
        self.persona_weights = persona_weights
        self.goal_params = dict(goal_params)  # defensive copy
        self.env_id = env_id
        self.p_missing = p_missing
        self.p_noise = p_noise
        self.T_patience = T_patience
        self.seed = seed

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
    ) -> tuple[str, bool]:
        """Generate a user response to the assistant's latest message.

        Logic:
            1. If the assistant is requesting clarification on a pending slot,
               provide the missing information.
            2. If progress_info is provided, track satisfaction and check
               for ragequit conditions.
            3. If T > T_patience, ragequit.
            4. Otherwise, generate a continuation message.

        Args:
            assistant_message: The assistant's most recent message text.
            tool_results:      List of tool results from the assistant's actions.
            progress_info:     Optional dict with 'satisfaction' (float in [0,1])
                               and/or 'done' (bool) keys.

        Returns:
            Tuple of (user_message, should_quit):
                - user_message: The user's response string.
                - should_quit: True if the user wants to end the conversation
                  (ragequit or task completion).
        """
        import random as _rand

        self._state.turn_count += 1
        rng = _rand.Random(self._next_seed())

        # Check if done signal from progress_info
        if progress_info is not None and progress_info.get("done", False):
            msg = rng.choice(_DONE_MESSAGES) if not self.deterministic_mode else _DONE_MESSAGES[0]
            return msg, True

        # Track satisfaction
        satisfaction: float | None = None
        if progress_info is not None:
            satisfaction = progress_info.get("satisfaction")
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
            return msg, True

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
                return msg, True

        # Check if assistant is asking for clarification on a pending slot
        if self._state.pending_slots:
            requested_slot = _detect_clarification_request(
                assistant_message, self._state.pending_slots
            )
            if requested_slot is not None and requested_slot in self.goal_params:
                # Provide the missing information
                self._state.pending_slots.discard(requested_slot)
                self._state.provided_slots.add(requested_slot)
                self._state.clarification_count += 1

                if self.deterministic_mode:
                    msg = f"{requested_slot}: {self.goal_params[requested_slot]}"
                else:
                    msg = render_clarification(
                        slot_name=requested_slot,
                        slot_value=self.goal_params[requested_slot],
                        seed=self._next_seed(),
                    )
                    if self.p_noise > 0.0:
                        noise_rng = _rand.Random(self._next_seed())
                        msg = apply_noise(msg, self.p_noise, noise_rng)

                return msg, False

        # Low satisfaction but not ragequit yet: express dissatisfaction
        if satisfaction is not None and satisfaction < _SATISFACTION_THRESHOLD:
            msg_idx = (
                0
                if self.deterministic_mode
                else rng.randint(0, len(_SATISFACTION_MESSAGES) - 1)
            )
            msg = _SATISFACTION_MESSAGES[msg_idx]
            return msg, False

        # Default: acknowledge and continue
        msg_idx = (
            0
            if self.deterministic_mode
            else rng.randint(0, len(_CONTINUATION_MESSAGES) - 1)
        )
        msg = _CONTINUATION_MESSAGES[msg_idx]
        return msg, False

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
