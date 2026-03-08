"""User utterance templates for each ShopRLVE-GYM environment.

Provides template-based utterance generation for the user simulator.
Each environment type has 5-10 template variants for diversity.

Key operations:
    - render_template: fill slots, optionally omit non-critical slots and inject noise
    - apply_noise: character-level typo / abbreviation injection
    - Templates are indexed by env_id (PD, SUB, CART, RETURN, STATUS, POLICY, BUNDLE, JOURNEY)

Slot filling rules:
    - With prob p_missing, each non-critical slot is omitted (forces clarification)
    - With prob p_noise, character-level noise is injected into the text
"""

from __future__ import annotations

import random
import string
from typing import Any


# ---------------------------------------------------------------------------
# Template banks per environment
# ---------------------------------------------------------------------------

# Each template is a format string with named slots.
# Slots marked with {?slot_name?} are considered non-critical and can be omitted.

_TEMPLATES: dict[str, list[str]] = {
    # -----------------------------------------------------------------------
    # E_PD: Product Discovery
    # -----------------------------------------------------------------------
    "PD": [
        "I'm looking for a {category} product under ${price_max}. {?brand_pref?} {?color_pref?} {?rating_req?}",
        "Can you recommend some {category} items? My budget is ${price_max}. {?brand_pref?} {?ship_req?}",
        "I need a {category}. {?brand_pref?} I'd like it under ${price_max} {?color_pref?}.",
        "Show me the best {category} options. {?price_range?} {?brand_pref?} {?rating_req?}",
        "Help me find a good {category}. {?brand_pref?} {?color_pref?} Budget: ${price_max}.",
        "Looking for {category} recommendations. {?price_range?} {?ship_req?} {?rating_req?}",
        "I want to buy a {category}. {?brand_pref?} Price should be under ${price_max}. {?material_pref?}",
        "What {category} products do you have? {?price_range?} {?brand_pref?} {?size_pref?}",
        "I'm in the market for a {category}. Max budget ${price_max}. {?brand_pref?} {?color_pref?} {?rating_req?}",
        "Searching for a {category} that's under ${price_max}. {?brand_pref?} {?ship_req?}",
    ],
    # -----------------------------------------------------------------------
    # E_SUB: Substitution
    # -----------------------------------------------------------------------
    "SUB": [
        "The {original_product} I wanted is out of stock. Can you find me a similar alternative? {?price_range?} {?brand_pref?}",
        "I was going to buy {original_product} but it's unavailable. What's a good substitute? {?color_pref?}",
        "My first choice {original_product} is sold out. Help me find something comparable. {?price_range?}",
        "{original_product} is out of stock. I need a replacement that's similar. {?brand_pref?} {?ship_req?}",
        "Can you suggest an alternative to {original_product}? It seems to be unavailable. {?price_range?} {?rating_req?}",
        "I need a substitute for {original_product} which is currently OOS. {?brand_pref?} {?color_pref?}",
        "The {original_product} I want isn't available. What else would you recommend? {?price_range?}",
        "Looking for something like {original_product} since it's out of stock. {?brand_pref?} {?ship_req?}",
    ],
    # -----------------------------------------------------------------------
    # E_CART: Cart Building
    # -----------------------------------------------------------------------
    "CART": [
        "I need to buy: {item_list}. Can you help me add them to my cart? {?variant_details?}",
        "Please add the following to my cart: {item_list}. {?quantity_details?}",
        "I want to order: {item_list}. {?variant_details?} {?quantity_details?}",
        "Add these items to my cart: {item_list}. {?color_pref?} {?size_pref?}",
        "Can you put {item_list} in my shopping cart? {?variant_details?}",
        "I'd like to buy {item_list}. Help me get them into my cart. {?quantity_details?}",
        "Shopping list: {item_list}. Please add everything. {?variant_details?} {?quantity_details?}",
    ],
    # -----------------------------------------------------------------------
    # E_RETURN: Returns
    # -----------------------------------------------------------------------
    "RETURN": [
        "I want to return the {product_desc} from order {order_ref}. {?reason?} {?replacement_req?}",
        "I need to return an item from my recent order {order_ref}. It's the {product_desc}. {?reason?}",
        "Can I return the {product_desc}? Order number is {order_ref}. {?reason?} {?replacement_req?}",
        "I'd like to initiate a return for {product_desc} from order {order_ref}. {?reason?}",
        "Please help me return the {product_desc} I received in order {order_ref}. {?reason?} {?replacement_req?}",
        "The {product_desc} from order {order_ref} needs to be returned. {?reason?}",
        "I'm not satisfied with the {product_desc} from {order_ref}. How do I return it? {?reason?} {?replacement_req?}",
        "Return request for {product_desc}, order {order_ref}. {?reason?}",
    ],
    # -----------------------------------------------------------------------
    # E_STATUS: Order Status
    # -----------------------------------------------------------------------
    "STATUS": [
        "What's the status of my order {order_ref}? {?specific_item?}",
        "Can you check on order {order_ref} for me? {?tracking_req?}",
        "I'd like to know where my order {order_ref} is. {?eta_req?}",
        "Has my order {order_ref} shipped yet? {?tracking_req?}",
        "When will order {order_ref} arrive? {?specific_item?}",
        "Track my order {order_ref} please. {?eta_req?}",
        "I'm checking on the delivery of order {order_ref}. {?tracking_req?} {?eta_req?}",
    ],
    # -----------------------------------------------------------------------
    # E_POLICY: Policy Q&A
    # -----------------------------------------------------------------------
    "POLICY": [
        "{policy_question}",
        "I have a question about your policies: {policy_question}",
        "Can you help me understand: {policy_question}",
        "Quick question: {policy_question}",
        "I need to know: {policy_question}",
        "What's your policy on this: {policy_question}",
    ],
    # -----------------------------------------------------------------------
    # E_BUNDLE: Bundle Planning
    # -----------------------------------------------------------------------
    "BUNDLE": [
        "I need to buy items from these categories: {category_list}. {?budget?} {?brand_pref?}",
        "Help me build a shopping list. I need: {category_list}. {?budget?}",
        "I'm shopping for: {category_list}. Can you recommend one from each? {?budget?} {?brand_pref?}",
        "Project shopping: I need items from {category_list}. {?budget?} {?quality_pref?}",
        "Please help me pick out: {category_list}. {?budget?} {?brand_pref?}",
        "I have a shopping project. Need one item each from: {category_list}. {?budget?}",
    ],
    # -----------------------------------------------------------------------
    # E_JOURNEY: Multi-Intent Journey
    # -----------------------------------------------------------------------
    "JOURNEY": [
        "I have a few things to take care of. First, {first_task}. {?second_hint?}",
        "Hi, I need help with multiple things. Let's start with: {first_task}.",
        "I've got a couple of requests. {first_task} {?second_hint?}",
        "Hey, first thing: {first_task}. I'll have another question after. {?second_hint?}",
        "I need assistance with something: {first_task}. {?second_hint?}",
    ],
}

# Slots that are NON-CRITICAL (can be omitted to force clarification).
# The format is {?slot_name?} in templates, but the actual param key is slot_name.
_NON_CRITICAL_SLOTS: set[str] = {
    "brand_pref",
    "color_pref",
    "rating_req",
    "ship_req",
    "price_range",
    "material_pref",
    "size_pref",
    "variant_details",
    "quantity_details",
    "reason",
    "replacement_req",
    "specific_item",
    "tracking_req",
    "eta_req",
    "budget",
    "quality_pref",
    "second_hint",
}


# ---------------------------------------------------------------------------
# Common abbreviations and typo patterns for noise injection
# ---------------------------------------------------------------------------

_ABBREVIATIONS: dict[str, str] = {
    "looking": "lookin",
    "something": "smthg",
    "recommend": "reccomend",
    "please": "pls",
    "product": "prodct",
    "shopping": "shoppin",
    "because": "cuz",
    "delivery": "delivry",
    "available": "availble",
    "alternative": "alterntive",
    "approximately": "approx",
    "information": "info",
    "quantity": "qty",
    "replacement": "replacemnt",
    "different": "diff",
    "interested": "intrested",
}


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------


def apply_noise(text: str, p_noise: float, rng: random.Random) -> str:
    """Apply character-level noise to text.

    With probability p_noise per character, one of the following mutations
    is applied:
        - Drop the character (30%)
        - Duplicate the character (30%)
        - Swap with the next character (20%)
        - Replace with a nearby keyboard character (20%)

    Additionally, random words may be replaced with abbreviations.

    Args:
        text:    Input text string.
        p_noise: Per-character noise probability in [0, 1].
        rng:     Random instance for reproducibility.

    Returns:
        Noisy text string.
    """
    if p_noise <= 0.0 or not text:
        return text

    # Phase 1: Word-level abbreviation substitution
    words = text.split(" ")
    for i, word in enumerate(words):
        lower_word = word.lower().rstrip(".,!?;:")
        if lower_word in _ABBREVIATIONS and rng.random() < p_noise * 3:
            # Preserve punctuation
            suffix = word[len(lower_word):]
            words[i] = _ABBREVIATIONS[lower_word] + suffix
    text = " ".join(words)

    # Phase 2: Character-level mutations
    chars = list(text)
    result: list[str] = []
    i = 0
    while i < len(chars):
        if chars[i] in (" ", "\n", "\t") or not chars[i].isalpha():
            result.append(chars[i])
            i += 1
            continue

        if rng.random() < p_noise:
            mutation = rng.random()
            if mutation < 0.30:
                # Drop character
                i += 1
                continue
            elif mutation < 0.60:
                # Duplicate character
                result.append(chars[i])
                result.append(chars[i])
                i += 1
                continue
            elif mutation < 0.80:
                # Swap with next character
                if i + 1 < len(chars) and chars[i + 1].isalpha():
                    result.append(chars[i + 1])
                    result.append(chars[i])
                    i += 2
                    continue
            else:
                # Replace with nearby keyboard key
                nearby = _nearby_key(chars[i])
                result.append(nearby)
                i += 1
                continue

        result.append(chars[i])
        i += 1

    return "".join(result)


def _nearby_key(char: str) -> str:
    """Return a character adjacent on a QWERTY keyboard layout.

    Args:
        char: Single alphabetic character.

    Returns:
        A nearby character, preserving case.
    """
    _KEYBOARD_NEIGHBORS: dict[str, str] = {
        "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg",
        "y": "tuh", "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
        "a": "qsz", "s": "awdx", "d": "sefc", "f": "drgv", "g": "ftbh",
        "h": "gynj", "j": "humk", "k": "jil", "l": "kop",
        "z": "asx", "x": "zdc", "c": "xfv", "v": "cgb", "b": "vhn",
        "n": "bjm", "m": "nk",
    }
    lower = char.lower()
    neighbors = _KEYBOARD_NEIGHBORS.get(lower, lower)
    if neighbors:
        import random as _rand  # local import to avoid naming conflict
        chosen = neighbors[hash(char) % len(neighbors)]
        return chosen.upper() if char.isupper() else chosen
    return char


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def render_template(
    env_id: str,
    params: dict[str, Any],
    p_missing: float,
    p_noise: float,
    seed: int,
) -> str:
    """Render a user utterance from a template for the given environment.

    Selects a template variant, fills required slots from params, optionally
    omits non-critical slots (forcing clarification from the agent), and
    optionally injects character-level noise (typos/abbreviations).

    Args:
        env_id:    Environment identifier: PD, SUB, CART, RETURN, STATUS,
                   POLICY, BUNDLE, or JOURNEY.
        params:    Dictionary of slot values. Must contain all required (non-optional)
                   slots for the selected template.
        p_missing: Probability of omitting each non-critical slot [0, 1].
        p_noise:   Probability of per-character noise injection [0, 1].
        seed:      Random seed for template selection and noise.

    Returns:
        Rendered user utterance string.

    Raises:
        KeyError: If env_id is not recognized.
    """
    rng = random.Random(seed)

    templates = _TEMPLATES.get(env_id)
    if templates is None:
        raise KeyError(
            f"Unknown env_id '{env_id}'. "
            f"Must be one of: {sorted(_TEMPLATES.keys())}"
        )

    # Select template variant
    template_idx = rng.randint(0, len(templates) - 1)
    template = templates[template_idx]

    # Process optional slots: replace {?slot_name?} markers
    rendered = _fill_template(template, params, p_missing, rng)

    # Clean up whitespace (multiple spaces, trailing spaces)
    rendered = " ".join(rendered.split())
    rendered = rendered.strip()

    # Apply noise
    if p_noise > 0.0:
        rendered = apply_noise(rendered, p_noise, rng)

    return rendered


def render_template_deterministic(
    env_id: str,
    params: dict[str, Any],
) -> str:
    """Render a template deterministically (template index 0, no noise, no omission).

    This is the DEBUG lever counterpart: always uses template index 0,
    p_missing=0, p_noise=0.

    Args:
        env_id: Environment identifier.
        params: Dictionary of slot values.

    Returns:
        Rendered user utterance string with all slots filled, no noise.
    """
    templates = _TEMPLATES.get(env_id)
    if templates is None:
        raise KeyError(
            f"Unknown env_id '{env_id}'. "
            f"Must be one of: {sorted(_TEMPLATES.keys())}"
        )

    template = templates[0]
    rng = random.Random(0)
    rendered = _fill_template(template, params, p_missing=0.0, rng=rng)
    rendered = " ".join(rendered.split())
    return rendered.strip()


def _fill_template(
    template: str,
    params: dict[str, Any],
    p_missing: float,
    rng: random.Random,
) -> str:
    """Fill a template string with slot values.

    Handles two slot types:
        - Required: {slot_name} -- always filled from params
        - Optional: {?slot_name?} -- filled with probability (1 - p_missing)

    Args:
        template:  Template string with slot markers.
        params:    Slot values dictionary.
        p_missing: Probability of omitting optional slots.
        rng:       Random instance.

    Returns:
        Filled template string.
    """
    result = template

    # First, handle optional slots {?slot_name?}
    import re
    optional_pattern = re.compile(r"\{\?(\w+)\?\}")
    for match in optional_pattern.finditer(template):
        slot_name = match.group(1)
        full_match = match.group(0)

        if slot_name in params and rng.random() >= p_missing:
            # Include the slot value
            result = result.replace(full_match, str(params[slot_name]), 1)
        else:
            # Omit the slot
            result = result.replace(full_match, "", 1)

    # Then, handle required slots {slot_name}
    # Use a simple replacement approach that doesn't collide with optional markers
    required_pattern = re.compile(r"\{(\w+)\}")
    for match in required_pattern.finditer(result):
        slot_name = match.group(1)
        full_match = match.group(0)
        if slot_name in params:
            result = result.replace(full_match, str(params[slot_name]), 1)

    return result


# ---------------------------------------------------------------------------
# Clarification response templates
# ---------------------------------------------------------------------------

_CLARIFICATION_TEMPLATES: dict[str, list[str]] = {
    "brand_pref": [
        "I prefer {brand_pref}.",
        "The brand I want is {brand_pref}.",
        "I'd like {brand_pref} brand.",
    ],
    "color_pref": [
        "I want it in {color_pref}.",
        "Color: {color_pref}.",
        "I'd prefer {color_pref} color.",
    ],
    "rating_req": [
        "I'd like something rated at least {rating_req}.",
        "Minimum rating should be {rating_req}.",
    ],
    "ship_req": [
        "I need it delivered within {ship_req} days.",
        "Shipping should be under {ship_req} days.",
    ],
    "price_range": [
        "My price range is {price_range}.",
        "Budget is {price_range}.",
    ],
    "size_pref": [
        "Size: {size_pref}.",
        "I need size {size_pref}.",
    ],
    "reason": [
        "The reason is: {reason}.",
        "I want to return it because {reason}.",
    ],
    "replacement_req": [
        "I'd like a replacement: {replacement_req}.",
        "Can you also find me a replacement? {replacement_req}",
    ],
    "variant_details": [
        "For the variants: {variant_details}.",
        "Specifically, {variant_details}.",
    ],
    "quantity_details": [
        "Quantities: {quantity_details}.",
        "I need {quantity_details}.",
    ],
    "budget": [
        "My total budget is {budget}.",
        "I'd like to spend no more than {budget} total.",
    ],
}


def render_clarification(
    slot_name: str,
    slot_value: Any,
    seed: int,
) -> str:
    """Render a clarification response for a previously omitted slot.

    Args:
        slot_name:  Name of the slot being clarified.
        slot_value: Value to fill in.
        seed:       Random seed for template selection.

    Returns:
        User response providing the missing information.
    """
    rng = random.Random(seed)

    templates = _CLARIFICATION_TEMPLATES.get(slot_name)
    if templates is None:
        return f"{slot_name}: {slot_value}"

    template = rng.choice(templates)
    return template.format(**{slot_name: slot_value})


def get_template_count(env_id: str) -> int:
    """Return the number of template variants for an environment.

    Args:
        env_id: Environment identifier.

    Returns:
        Number of templates available.

    Raises:
        KeyError: If env_id is not recognized.
    """
    templates = _TEMPLATES.get(env_id)
    if templates is None:
        raise KeyError(f"Unknown env_id '{env_id}'")
    return len(templates)


def get_available_env_ids() -> list[str]:
    """Return the list of environment IDs with registered templates.

    Returns:
        Sorted list of env_id strings.
    """
    return sorted(_TEMPLATES.keys())
