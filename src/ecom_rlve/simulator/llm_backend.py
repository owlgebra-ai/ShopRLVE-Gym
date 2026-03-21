"""Ollama-based LLM backend for EcomRLVE-GYM user simulation.

Provides a thin wrapper around the Ollama HTTP API for generating
naturalistic user utterances, strategic constraint omission, and
mid-conversation dialogue responses.

Model: qwen3.5 (via local Ollama server)
Endpoint: http://localhost:11434/api/generate

Design principles:
    - Deterministic seeding via Ollama's seed parameter
    - Graceful fallback to template-based generation if Ollama is unavailable
    - Short, focused prompts to minimise latency
    - All LLM outputs are for user-facing text only; verifier logic is unaffected
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "qwen3.5"
OLLAMA_TIMEOUT: int = 30  # seconds per request
OLLAMA_TEMPERATURE: float = 0.7
OLLAMA_MAX_TOKENS: int = 200


# ---------------------------------------------------------------------------
# Low-level Ollama client
# ---------------------------------------------------------------------------


def _ollama_generate(
    prompt: str,
    seed: int = 42,
    temperature: float = OLLAMA_TEMPERATURE,
    max_tokens: int = OLLAMA_MAX_TOKENS,
    system_prompt: str | None = None,
) -> str | None:
    """Call Ollama's /api/chat endpoint with thinking disabled.

    Uses the chat completions API with ``think: false`` so that
    thinking-enabled models (like qwen3.5) produce content directly
    without exhausting the token budget on internal reasoning.

    Args:
        prompt:        The user message string.
        seed:          Deterministic seed for reproducibility.
        temperature:   Sampling temperature.
        max_tokens:    Maximum tokens to generate.
        system_prompt: Optional system prompt to set persona/env context.

    Returns:
        Generated text string, or None if the call fails.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,  # Disable extended thinking for qwen3.5
        "options": {
            "seed": seed,
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content", "").strip()
        # Safety net: strip any residual <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text if text else None
    except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
        logger.warning("Ollama call failed: %s", exc)
        return None


def is_ollama_available() -> bool:
    """Check if the Ollama server is reachable and the model is loaded."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return any(OLLAMA_MODEL in m.get("name", "") for m in models)
    except (requests.RequestException, json.JSONDecodeError):
        return False


# ---------------------------------------------------------------------------
# Environment-aware system prompt for user simulator
# ---------------------------------------------------------------------------

_ENV_OBJECTIVES: dict[str, str] = {
    "PD": "find products matching your specific requirements",
    "SUB": "find a substitute for an out-of-stock product you wanted",
    "CART": "get specific items added to your shopping cart with the correct variants and quantities",
    "RETURN": "return a specific item from a previous order and optionally find a replacement",
    "STATUS": "find out the current status of a specific order",
    "POLICY": "get an answer to a specific store policy question",
    "BUNDLE": "get product recommendations covering specific categories within your budget",
    "JOURNEY": "complete multiple shopping tasks in one conversation",
}


def build_user_system_prompt(
    env_id: str,
    goal_summary: str,
    persona_summary: str = "",
) -> str:
    """Build an env-specific system prompt for user simulator LLM calls.

    Grounds the LLM in the correct shopping scenario so it stays on-topic,
    reflects the persona, and evaluates the assistant's actions against
    the user's actual needs.

    Args:
        env_id:          Environment identifier (PD, CART, etc.).
        goal_summary:    Human-readable description of the user's hidden goal.
        persona_summary: Optional description of persona preferences.

    Returns:
        System prompt string.
    """
    objective = _ENV_OBJECTIVES.get(env_id, "complete your shopping task")

    system = (
        f"You are a realistic online shopping customer in a conversation with a "
        f"shopping assistant.\n"
        f"Your objective: {objective}.\n\n"
        f"Your specific goal: {goal_summary}\n"
    )

    if persona_summary:
        system += f"\nYour preferences: {persona_summary}\n"

    system += (
        "\nIMPORTANT RULES:\n"
        "- Stay on topic. Only discuss matters related to your shopping goal.\n"
        "- Be brief and natural (1-2 sentences per response).\n"
        "- Do NOT invent product names or IDs you haven't been shown.\n"
        "- If the assistant shows you items or modifies your cart, evaluate "
        "them against your actual needs.\n"
        "- If something is wrong (wrong item, wrong quantity, wrong variant), "
        "say so clearly.\n"
        "- If everything looks correct, confirm positively.\n"
    )

    return system


# ---------------------------------------------------------------------------
# RETURN-specific LLM verbalization
# ---------------------------------------------------------------------------


def verbalize_return_request(
    product_title: str,
    reason: str | None,
    replacement_required: bool,
    replacement_constraints: list[dict[str, Any]] | None,
    p_missing: float,
    seed: int,
) -> tuple[str | None, set[str], set[str]]:
    """Generate a natural return request via LLM.

    The user describes the product they want to return WITHOUT mentioning
    the order ID.  The agent must discover the order by scanning order
    history.

    With p_missing > 0.05, the LLM may omit some details (reason,
    replacement request) from the initial message.

    Args:
        product_title:           Title of the product being returned.
        reason:                  Return reason (may be omitted).
        replacement_required:    Whether the user wants a replacement.
        replacement_constraints: Constraints for the replacement product.
        p_missing:               Probability of omitting optional details.
        seed:                    Random seed.

    Returns:
        Tuple of (message, mentioned_attrs, omitted_attrs).
        Returns (None, set(), set()) if LLM is unavailable.
    """
    import random as _rand
    rng = _rand.Random(seed)

    # Decide which optional details to include
    include_reason = reason is not None and rng.random() >= p_missing
    include_replacement = replacement_required and rng.random() >= p_missing

    # Build the product description (never include order ID)
    parts: list[str] = [f"Product: {product_title}"]
    if include_reason:
        parts.append(f"Reason: {reason}")
    if include_replacement:
        if replacement_constraints:
            constraint_desc = _format_constraints_list(replacement_constraints)
            parts.append(f"Replacement needed matching: {constraint_desc}")
        else:
            parts.append("Also want a replacement")

    details_str = "\n".join(f"- {p}" for p in parts)

    prompt = (
        "You are a customer messaging a shopping assistant. "
        "You want to return a product you bought recently. "
        "Do NOT mention any order ID or order number — you don't remember it. "
        "Just describe the product you want to return.\n\n"
        "Stay focused ONLY on initiating the return. "
        "Be natural and conversational. 1-3 sentences max.\n\n"
        f"Details:\n{details_str}\n\n"
        "Write ONLY the customer message:"
    )

    text = _ollama_generate(prompt, seed=seed, max_tokens=150)
    if text is None:
        return None, set(), set()

    # Track mentioned vs omitted slots
    mentioned: set[str] = {"product_desc"}  # always mentioned
    omitted: set[str] = set()

    if include_reason:
        mentioned.add("reason")
    elif reason is not None:
        omitted.add("reason")

    if include_replacement:
        mentioned.add("replacement_req")
    elif replacement_required:
        omitted.add("replacement_req")

    # order_ref is always omitted (by design)
    omitted.add("order_ref")

    return text, mentioned, omitted


# ---------------------------------------------------------------------------
# CART-specific LLM verbalization
# ---------------------------------------------------------------------------


def _build_natural_item_hint(d: dict[str, Any]) -> str:
    """Build a short, natural description of a product for verbalization.

    Real shoppers describe items by what the product *is* (derived from the
    title) + maybe brand — never the full catalog title.

    E.g. "phone case by Anker" not "Nurbo Cute Dragonfly Shape Phone Ring
    360 Degree Rotating Ring Grip Anti Drop Finger Holder for iPhone iPad
    and All Cellphone (White+Black)".

    Strategy:
        1. Extract a short product-type phrase from the title (first 4-6
           meaningful words, stripping the brand prefix if it leads).
        2. Add brand if it's well-known / meaningful.
        3. Add variant preference if any (e.g. "in black", "size M").
        4. Add quantity if > 1.

    Returns a concise hint string for the LLM prompt.
    """
    title = d.get("title", "")
    brand = d.get("brand", "")

    # --- Extract short product-type phrase from the title ---
    product_type = extract_product_type(title, brand)

    # --- Assemble ---
    if brand and brand.lower() not in ("unknown", "generic", "unbranded", ""):
        desc = f"{product_type} by {brand}"
    else:
        desc = product_type

    # Variant
    if d.get("variant_desc"):
        desc += f" ({d['variant_desc']})"

    # Quantity
    if d.get("qty", 1) > 1:
        desc += f" x{d['qty']}"

    return desc


def extract_product_type(title: str, brand: str = "") -> str:
    """Extract a short product-type phrase from a catalog title.

    Strips the brand name if it appears at the start, then takes the
    first 5 meaningful words.  Also strips trailing model numbers,
    parenthetical specs, and size/colour suffixes.

    Examples:
        "MOSNOVO Galaxy S9 Plus Case, Galaxy..." → "Galaxy S9 Plus Case"
        "BLU VIVO 5 Case Soft tpu Silicone..." → "VIVO 5 Case"
        "10x Airbrush 80cc Plastic Bottles Paint..." → "Airbrush Plastic Bottles"
        "Nurbo Cute Dragonfly Shape Phone Ring 360..." → "Dragonfly Shape Phone Ring"
    """
    if not title:
        return "an item"

    import re

    text = title.strip()

    # Truncate at comma or dash — usually signals spec repetition
    # e.g. "Galaxy S9 Plus Case, Galaxy S9 Plus Phone Case" → first part
    for sep in [",", " - ", " — ", " – ", " | "]:
        if sep in text:
            text = text.split(sep)[0].strip()

    # Strip brand prefix (case-insensitive)
    if brand:
        brand_lower = brand.lower().strip()
        text_lower = text.lower()
        if text_lower.startswith(brand_lower):
            text = text[len(brand_lower):].lstrip(" ,-:")
        # Also try first word match (brands sometimes differ in casing)
        elif text.split() and text.split()[0].lower() == brand_lower.split()[0].lower():
            text = " ".join(text.split()[1:])

    # Remove content in parentheses/brackets — often specs or colour info
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)

    # Remove leading quantity patterns like "10x", "2 Pack", "3-Pack"
    text = re.sub(r"^\d+[xX]\s*", "", text)
    text = re.sub(r"^\d+\s*-?\s*[Pp]ack\s*", "", text)

    # Remove common filler/marketing words (only at the start)
    filler = {
        "compatible", "with", "for", "and", "the", "all", "new",
        "premium", "professional", "original", "genuine", "official",
        "high", "quality", "best", "top", "ultra", "super", "deluxe",
    }

    words = text.split()
    # Take first 4 meaningful (non-filler, non-punctuation) words
    kept: list[str] = []
    for w in words:
        clean = w.strip(",-;:.!?™®©")
        if not clean:
            continue
        if clean.lower() in filler and len(kept) == 0:
            continue  # skip leading filler only
        # Skip trailing pure numbers (model numbers, counts)
        if clean.isdigit() and len(kept) >= 2:
            continue
        kept.append(clean)
        if len(kept) >= 4:
            break

    if not kept:
        # Fallback to category leaf
        cat = d.get("category", "") if isinstance(d, dict) else ""
        if cat:
            leaf = cat.split("/")[-1] if "/" in cat else cat
            return leaf.lower() if leaf.isupper() else leaf
        return "an item"

    return " ".join(kept)


# ---------------------------------------------------------------------------
# Diversity injection for cart verbalization
# ---------------------------------------------------------------------------

# Persona + tone combos — sampled per call to break structural monotony
_CART_STYLE_DIRECTIVES: list[str] = [
    "Write as a busy parent quickly firing off a message while multitasking.",
    "Write as a tech-savvy young adult who uses slang and abbreviations.",
    "Write as a polite, slightly formal professional placing a work order.",
    "Write as an impatient shopper who gets straight to the point — no pleasantries.",
    "Write as a friendly, chatty person who adds a little context about why they need the items.",
    "Write as someone who's unsure and a bit tentative — 'I think I need...' style.",
    "Write as a confident shopper who gives direct commands — 'Put X in my cart.'",
    "Write as someone who lists items matter-of-factly, like a grocery list in sentence form.",
    "Write as an excited shopper who just found exactly what they wanted.",
    "Write as a no-nonsense person: short, clipped sentences. No filler words.",
    "Write as a very casual texter — lowercase okay, fragments okay, emoji okay.",
    "Write as someone who explains their need first, then specifies the items.",
    "Write as a cautious buyer who emphasizes they want exactly these items, nothing else.",
    "Write as a repeat customer — mention you've bought similar things before.",
    "Write as someone in a rush — abbreviate, skip articles, be terse.",
]

# Varied opening instructions to avoid structural collapse
_CART_FRAMING_VARIANTS: list[str] = [
    "You're messaging an online store's chat support.",
    "You're typing into a shopping website's chat box.",
    "You're sending a text to a personal shopper.",
    "You're leaving a voice message that got transcribed (natural speech patterns).",
    "You're replying to a 'How can I help?' message from a store assistant.",
    "You're writing a quick note to someone picking up items for you.",
]


def verbalize_cart_request(
    item_details: list[dict[str, Any]],
    p_missing: float,
    seed: int,
) -> tuple[str | None, set[str], set[str]]:
    """Generate a natural cart-building request via LLM.

    Uses short **category + brand** hints — NOT exact product titles.
    A real customer says "I need a phone ring holder and a cross stitch
    kit", not "I need the Nurbo Cute Dragonfly Shape Phone Ring 360 ...".

    The agent must then search the catalog, check the user's visit
    history, and present candidates for the user to confirm.

    Args:
        item_details: List of dicts with keys: title, qty, variant_desc,
                      category, brand, features.
        p_missing:    Controls how much detail to omit (higher = vaguer).
        seed:         Random seed.

    Returns:
        Tuple of (message, mentioned_attrs, omitted_attrs).
        Returns (None, set(), set()) if LLM is unavailable.
    """
    import random as _rng
    rng = _rng.Random(seed)

    # Build SHORT natural descriptions — never the exact title
    items_desc: list[str] = []
    for d in item_details:
        items_desc.append(_build_natural_item_hint(d))

    items_str = "\n".join(f"- {item}" for item in items_desc)

    # --- Diversity injection ---
    style = rng.choice(_CART_STYLE_DIRECTIVES)
    framing = rng.choice(_CART_FRAMING_VARIANTS)

    if p_missing > 0.05:
        detail_instruction = (
            "You may leave out some details like exact quantities or "
            "color/size preferences — you'll share those later if asked."
        )
    else:
        detail_instruction = (
            "Include all relevant details: quantities, color/size/variant "
            "preferences if any."
        )

    prompt = (
        f"Scenario: {framing}\n"
        f"Style: {style}\n\n"
        "You want specific items added to your cart. "
        "Describe what you want in your own words — do NOT copy the "
        "descriptions below verbatim. Rephrase naturally.\n\n"
        f"{detail_instruction}\n\n"
        "RULES:\n"
        "- Do NOT start with \"I'd like to add\" — vary your phrasing.\n"
        "- Do NOT ask for recommendations or alternatives.\n"
        "- 1-3 sentences max. No preamble or sign-off.\n\n"
        f"Items you need:\n{items_str}\n\n"
        "Customer message:"
    )

    text = _ollama_generate(prompt, seed=seed, temperature=1.0, max_tokens=200)
    if text is None:
        return None, set(), set()

    # Detect which details were mentioned vs omitted
    text_lower = text.lower()
    mentioned: set[str] = set()
    omitted: set[str] = set()

    # item_list is always mentioned (the items themselves)
    mentioned.add("item_list")

    # Check variant details
    has_variant_info = False
    for d in item_details:
        if d.get("variant_desc"):
            # Check if the variant value appears in the text
            variant_val = d["variant_desc"].split(":")[-1].strip().lower()
            if variant_val in text_lower:
                has_variant_info = True
    if has_variant_info:
        mentioned.add("variant_details")
    elif any(d.get("variant_desc") for d in item_details):
        omitted.add("variant_details")

    # Check quantity details
    has_qty_info = False
    for d in item_details:
        if d.get("qty", 1) > 1:
            qty_str = str(d["qty"])
            if qty_str in text_lower or _number_word(d["qty"]) in text_lower:
                has_qty_info = True
    if has_qty_info:
        mentioned.add("quantity_details")
    elif any(d.get("qty", 1) > 1 for d in item_details):
        omitted.add("quantity_details")

    return text, mentioned, omitted


def _number_word(n: int) -> str:
    """Convert small integers to word form for detection."""
    words = {2: "two", 3: "three", 4: "four", 5: "five"}
    return words.get(n, str(n))


# ---------------------------------------------------------------------------
# LLM-based clarification detection
# ---------------------------------------------------------------------------


def detect_clarification_with_llm(
    assistant_message: str,
    pending_slots: set[str],
    seed: int,
    system_prompt: str | None = None,
) -> str | None:
    """Use the LLM to detect if the assistant is asking about a pending slot.

    Falls back gracefully if Ollama is unavailable. This is more robust
    than pure keyword matching because it understands paraphrases like
    "do you want one or two?" -> quantity_details.

    Args:
        assistant_message: The assistant's latest message.
        pending_slots:     Set of slot names still omitted.
        seed:              Random seed.
        system_prompt:     Optional system prompt for context.

    Returns:
        The slot name being asked about, or None.
    """
    if not pending_slots:
        return None

    slots_desc: list[str] = []
    slot_explanations: dict[str, str] = {
        "variant_details": "specific variant/color/size preference",
        "quantity_details": "exact quantity or how many",
        "brand_pref": "brand preference",
        "color_pref": "color preference",
        "size_pref": "size preference",
        "rating_req": "minimum rating requirement",
        "ship_req": "shipping speed requirement",
        "price_range": "price range or budget",
        "material_pref": "material preference",
        "reason": "reason for return",
        "replacement_req": "whether a replacement is wanted",
        "order_ref": "order number or order ID for a return",
        "budget": "total budget",
    }
    for slot in pending_slots:
        desc = slot_explanations.get(slot, slot.replace("_", " "))
        slots_desc.append(f"- {slot}: {desc}")

    slots_str = "\n".join(slots_desc)

    prompt = (
        f"A shopping assistant just said:\n"
        f"\"{assistant_message[:300]}\"\n\n"
        f"The customer has these details they haven't shared yet:\n"
        f"{slots_str}\n\n"
        f"Is the assistant asking about any of these? "
        f"Reply with ONLY the slot name (e.g. 'quantity_details') "
        f"or 'none' if the assistant is not asking about any of them."
    )

    result = _ollama_generate(
        prompt, seed=seed, max_tokens=20, system_prompt=system_prompt,
    )
    if result is None:
        return None

    result = result.strip().strip("'\"").lower()
    # Check if the response matches a pending slot
    for slot in pending_slots:
        if slot.lower() in result:
            return slot

    return None


# ---------------------------------------------------------------------------
# High-level generation functions
# ---------------------------------------------------------------------------

# -- Constraint formatting helpers ------------------------------------------

_OP_LABELS: dict[str, str] = {
    "eq": "must be",
    "neq": "must not be",
    "gt": "greater than",
    "gte": "at least",
    "lt": "less than",
    "lte": "at most",
}


def _format_constraint(c: dict[str, Any]) -> str:
    """Convert a constraint dict to a human-readable phrase."""
    attr = c["attr"].replace("_", " ")
    op_label = _OP_LABELS.get(c["op"], c["op"])
    value = c["value"]
    # Format numeric values nicely
    if c["attr"] == "price":
        return f"{attr} {op_label} ${value}"
    elif c["attr"] in ("rating", "ship_days", "wattage", "weight_lbs",
                        "screen_size_inches", "rating_count"):
        return f"{attr} {op_label} {value}"
    else:
        return f"{attr} {op_label} {value}"


def _format_constraints_list(constraints: list[dict[str, Any]]) -> str:
    """Format a list of constraints into a comma-separated description."""
    return ", ".join(_format_constraint(c) for c in constraints)


# ---------------------------------------------------------------------------
# Verbalize constraints into a natural user message (Fix 1)
# ---------------------------------------------------------------------------


def verbalize_constraints(
    category: str,
    constraints: list[dict[str, Any]],
    seed: int,
) -> str | None:
    """Use the LLM to turn structured constraints into a natural user message.

    This replaces template-based utterance generation. All constraint attributes
    (including obscure ones like wattage, material, connector_type) are now
    expressible in natural language.

    Args:
        category:    Product category string.
        constraints: List of constraint dicts with keys: attr, op, value.
        seed:        Random seed for deterministic generation.

    Returns:
        Natural language user message, or None if LLM is unavailable.
    """
    constraint_desc = _format_constraints_list(constraints)

    prompt = (
        "You are a customer shopping online. Write a short, casual message "
        "to a shopping assistant expressing the following needs. "
        "Do NOT use bullet points or structured lists. Be conversational "
        "and natural. Vary your phrasing. Write 1-3 sentences max.\n\n"
        f"Product category: {category}\n"
        f"Requirements: {constraint_desc}\n\n"
        "Customer message:"
    )

    return _ollama_generate(prompt, seed=seed, max_tokens=150)


# ---------------------------------------------------------------------------
# Strategic omission: LLM decides which constraints to mention first (Fix 5)
# ---------------------------------------------------------------------------


def verbalize_with_strategic_omission(
    category: str,
    constraints: list[dict[str, Any]],
    seed: int,
) -> tuple[str | None, set[str], set[str]]:
    """Generate an initial user message that naturally omits some constraints.

    Instead of random p_missing, the LLM decides which requirements to
    mention first (the most important 2-3) and which to reveal later.
    We then detect which constraint values appear in the output to track
    what was mentioned vs omitted.

    Args:
        category:    Product category string.
        constraints: Full list of constraint dicts.
        seed:        Random seed.

    Returns:
        Tuple of (message, mentioned_attrs, omitted_attrs).
        Returns (None, set(), set()) if LLM is unavailable.
    """
    constraint_desc = _format_constraints_list(constraints)

    prompt = (
        "You are a customer messaging a shopping assistant for the first time. "
        "You have these requirements, but you don't need to mention everything "
        "in your first message. Share the 2-3 most important requirements "
        "naturally. You can share the rest if the assistant asks.\n\n"
        f"Product category: {category}\n"
        f"All requirements: {constraint_desc}\n\n"
        "Write ONLY the customer message (1-3 sentences, casual tone):"
    )

    text = _ollama_generate(prompt, seed=seed, max_tokens=150)
    if text is None:
        return None, set(), set()

    # Detect which constraints were actually mentioned by checking for
    # the presence of attribute values in the generated text
    text_lower = text.lower()
    mentioned: set[str] = set()
    omitted: set[str] = set()

    for c in constraints:
        value_str = str(c["value"]).lower()
        attr = c["attr"]

        # Check if the value (or a recognizable form) appears in the text
        if value_str in text_lower:
            mentioned.add(attr)
        elif attr == "price" and "$" in text_lower:
            # Price might be expressed differently ($30, thirty dollars, etc.)
            mentioned.add(attr)
        elif attr == "rating" and ("star" in text_lower or "rated" in text_lower):
            mentioned.add(attr)
        elif attr == "ship_days" and ("ship" in text_lower or "deliver" in text_lower or "day" in text_lower):
            mentioned.add(attr)
        elif attr == "brand" and value_str in text_lower:
            mentioned.add(attr)
        else:
            omitted.add(attr)

    return text, mentioned, omitted


# ---------------------------------------------------------------------------
# Mid-conversation response generation (Fix 6)
# ---------------------------------------------------------------------------


def generate_dialogue_response(
    context: str,
    assistant_message: str,
    seed: int,
    goal_summary: str = "",
    tool_results_summary: str = "",
    satisfaction: float | None = None,
    system_prompt: str | None = None,
    cart_issues: list[str] | None = None,
    confirm_hint: str | None = None,
) -> str | None:
    """Generate a natural mid-conversation user response via LLM.

    Replaces canned response lists (_CONTINUATION_MESSAGES, etc.) with
    dynamic, contextual responses grounded in the env-specific system
    prompt and optional cart ground-truth feedback.

    Args:
        context:              One of 'continue', 'dissatisfied', 'cart_feedback',
                              'cart_confirm', 'done'.
        assistant_message:    The assistant's latest message.
        seed:                 Random seed.
        goal_summary:         Short description of the user's goal.
        tool_results_summary: Summary of products/results shown.
        satisfaction:         Current satisfaction score [0, 1] or None.
        system_prompt:        Env-specific system prompt with persona/goal context.
        cart_issues:          List of specific issues with the cart state
                              (e.g. 'Mouse is missing', 'USB-C Hub qty is 1, need 2').
        confirm_hint:         When context='cart_confirm', a hint about which
                              items to confirm (e.g. "Yes, I want the Anker charger").

    Returns:
        Natural user response string, or None if LLM is unavailable.
    """
    mood_map = {
        "continue": "The suggestions look interesting, you want to continue browsing.",
        "dissatisfied": "The suggestions don't match what you need. Express mild frustration.",
        "done": "You're satisfied and want to wrap up. Thank the assistant.",
        "cart_feedback": "Check what the assistant did against your shopping list and respond.",
        "cart_confirm": "The assistant is showing you product options. Confirm the right ones.",
    }
    mood = mood_map.get(context, "Respond naturally to continue the conversation.")

    prompt = f"Assistant just said: \"{assistant_message[:300]}\"\n"

    if goal_summary:
        prompt += f"Your goal: {goal_summary}\n"
    if tool_results_summary:
        prompt += f"Products shown: {tool_results_summary[:200]}\n"

    # Cart ground-truth feedback: tell the LLM what's wrong so it can
    # generate a natural correction message
    if context == "cart_confirm" and confirm_hint:
        prompt += (
            f"\nThe assistant is presenting options. You want to confirm "
            f"the correct items. Your answer should be something like: "
            f"\"{confirm_hint}\"\n"
            "Rephrase this naturally as a customer would. Be brief. "
            "Do NOT use product IDs. Refer to items by name.\n"
        )
    elif cart_issues:
        prompt += "\nIssues with your cart right now:\n"
        for issue in cart_issues:
            prompt += f"- {issue}\n"
        prompt += (
            "\nPoint out the issues naturally as a customer would. "
            "Do NOT list product IDs. Refer to items by name.\n"
        )
    elif context == "cart_feedback":
        prompt += (
            "\nYour cart looks correct so far. Acknowledge positively.\n"
        )

    prompt += f"Your mood: {mood}\n"
    prompt += "\nRespond in 1-2 sentences. Be brief and realistic.\nCustomer response:"

    return _ollama_generate(
        prompt, seed=seed, max_tokens=100, system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# Clarification response generation
# ---------------------------------------------------------------------------


def generate_clarification_response(
    slot_name: str,
    slot_value: Any,
    seed: int,
    assistant_message: str = "",
    system_prompt: str | None = None,
) -> str | None:
    """Generate a natural clarification response for a previously omitted slot.

    Args:
        slot_name:         Attribute name being clarified (e.g., "color", "brand").
        slot_value:        The value to communicate.
        seed:              Random seed.
        assistant_message: Optional assistant message that triggered the clarification.
        system_prompt:     Env-specific system prompt with persona/goal context.

    Returns:
        Natural clarification response, or None if LLM is unavailable.
    """
    attr_label = slot_name.replace("_", " ")

    prompt = ""
    if assistant_message:
        prompt += f"Assistant asked: \"{assistant_message[:200]}\"\n\n"
    prompt += (
        f"They are asking about your {attr_label} preference.\n"
        f"Your answer: {slot_value}\n\n"
        "Respond naturally in 1 sentence providing this information.\n"
        "Customer response:"
    )

    return _ollama_generate(
        prompt, seed=seed, max_tokens=60, system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# LLM-powered variant attribute generation
# ---------------------------------------------------------------------------

# Module-level cache: subcategory → {attr_name: [values]}
_LLM_VARIANT_CACHE: dict[str, dict[str, list[str]]] = {}


def generate_variant_attrs_for_category(
    subcategory: str,
    product_title: str = "",
    seed: int = 42,
) -> dict[str, list[str]]:
    """Use Ollama to generate realistic variant attributes for a product subcategory.

    Calls the LLM once per subcategory (cached) to produce 3-4 attributes
    that real e-commerce stores use as product variants, with 4-7 plausible
    values each. Excludes color/size/material which are already hardcoded.

    Args:
        subcategory:   Slash-path category like "electronics/audio/headphones".
        product_title: Example product title for context (optional).
        seed:          Random seed for deterministic generation.

    Returns:
        Dict mapping attribute names (snake_case) to lists of string values.
        Empty dict if Ollama is unavailable.
    """
    if subcategory in _LLM_VARIANT_CACHE:
        return _LLM_VARIANT_CACHE[subcategory]

    # Human-readable category for the prompt
    cat_readable = subcategory.replace("/", " > ")

    prompt = (
        f"Product category: {cat_readable}\n"
    )
    if product_title:
        prompt += f"Example product: {product_title}\n"
    prompt += (
        "\nList 3-4 attributes that real e-commerce stores use as product variants "
        "(selectable options on the product page) for this category. "
        "Do NOT include: color, size, material, price, brand, or weight.\n\n"
        "For each attribute, give 5-7 plausible values.\n\n"
        "Reply ONLY in this exact JSON format, no other text:\n"
        '{"attributes": [\n'
        '  {"name": "attribute_name_snake_case", "values": ["val1", "val2", "val3", "val4", "val5"]},\n'
        '  ...\n'
        "]}\n\n"
        "Example for electronics/mobile/phones:\n"
        '{"attributes": [\n'
        '  {"name": "storage_capacity", "values": ["64GB", "128GB", "256GB", "512GB", "1TB"]},\n'
        '  {"name": "ram", "values": ["4GB", "6GB", "8GB", "12GB"]},\n'
        '  {"name": "screen_type", "values": ["OLED", "AMOLED", "LCD", "Mini-LED"]}\n'
        "]}\n\n"
        "Now generate for the category above:"
    )

    text = _ollama_generate(prompt, seed=seed, temperature=0.7, max_tokens=400)
    if text is None:
        _LLM_VARIANT_CACHE[subcategory] = {}
        return {}

    # Parse JSON from LLM response
    result: dict[str, list[str]] = {}
    try:
        # Extract JSON block (LLM might wrap it in markdown)
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            for attr_obj in parsed.get("attributes", []):
                name = attr_obj.get("name", "").strip().lower().replace(" ", "_")
                values = attr_obj.get("values", [])
                # Validate: skip trivially empty or blacklisted attrs
                if (
                    name
                    and name not in ("color", "size", "material", "weight", "brand", "price")
                    and len(values) >= 3
                ):
                    # Ensure all values are strings
                    result[name] = [str(v) for v in values[:8]]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.debug("Failed to parse LLM variant attrs for %s: %s", subcategory, exc)

    _LLM_VARIANT_CACHE[subcategory] = result
    return result


def clear_variant_cache() -> None:
    """Clear the LLM variant attribute cache (useful for testing)."""
    _LLM_VARIANT_CACHE.clear()
