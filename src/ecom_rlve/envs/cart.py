"""E_CART -- Cart Building environment.

Spec Section 5 / E3:
    Difficulty:
        n_items(d) = 1 + floor(d/3)  (cap at 5)
        p_var(d)   = sigmoid((d-2)/1.5)
        p_qty(d)   = min(0.5, 0.1*d)

    Generator:
        1. Sample n_items target products.
        2. For each, optionally require specific variant with prob p_var(d).
        3. For each, set required quantity:
            q_j = 1 with prob (1 - p_qty(d)), else q_j ~ U{2,4}.

    Reward:
        Unit-level F1:
            prec = U_match / (U_cart + 1e-9)
            rec  = U_match / (U_req  + 1e-9)
            F1   = 2*prec*rec / (prec + rec + 1e-9)
        r_task = 2*F1 - 1

    IsCorrect:
        F1 == 1.0
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from ecom_rlve.data.schema import (
    ATTRIBUTE_ALLOWLIST,
    DENIED_CATEGORIES,
    Product,
    Variant,
    avail,
)
from ecom_rlve.difficulty.mapping import map_difficulty, sigmoid
from ecom_rlve.rewards.verifiers import verify_cart
from ecom_rlve.simulator.llm_backend import (
    extract_product_type,
    generate_variant_attrs_for_category,
    verbalize_cart_request,
)
from ecom_rlve.simulator.templates import render_template

from ecom_rlve.envs.base import (
    BaseEnvironment,
    EpisodeResult,
    ProblemParams,
    register_env,
)


# ---------------------------------------------------------------------------
# Synthetic variant generation — hybrid category-based + data-driven
# ---------------------------------------------------------------------------

# Category -> preferred attributes to vary (ordered by priority)
CATEGORY_VARIANT_ATTRS: dict[str, list[str]] = {
    "Electronics": ["connector_type", "wattage", "color"],
    "Computers": ["connector_type", "color", "size"],
    "Cell Phones & Accessories": ["color", "connector_type", "size"],
    "AMAZON FASHION": ["color", "size", "material"],
    "Clothing, Shoes & Jewelry": ["color", "size", "material"],
    "Amazon Home": ["color", "material", "size"],
    "Home & Kitchen": ["color", "material", "size"],
    "Tools & Home Improvement": ["color", "size", "material"],
    "Beauty & Personal Care": ["item_form", "skin_type", "finish_type"],
    "Health & Household": ["item_form", "size"],
    "Automotive": ["color", "size", "material"],
    "Sports & Outdoors": ["color", "size", "material"],
    "Toys & Games": ["color", "size"],
    "Office Products": ["color", "size"],
    "Pet Supplies": ["size", "color"],
    "Industrial & Scientific": ["size", "material"],
    "Digital Music": ["color"],
    "Books": ["size"],
    "Video Games": ["color"],
}

# Fallback: scan product attrs for these in order
_FALLBACK_VARIANT_ATTRS: list[str] = [
    "color", "size", "material", "connector_type", "wattage",
    "item_form", "finish_type", "skin_type",
]

# Synthetic values for discrete attributes
_SYNTHETIC_VALUES: dict[str, list[str]] = {
    "color": ["black", "white", "silver", "red", "blue", "green", "navy", "gold"],
    "size": ["XS", "S", "M", "L", "XL", "XXL"],
    "material": [
        "cotton", "polyester", "leather", "metal", "plastic",
        "wood", "nylon", "silicone",
    ],
    "connector_type": [
        "USB-A", "USB-C", "Lightning", "Micro-USB",
        "HDMI", "Thunderbolt", "3.5mm",
    ],
    "item_form": ["cream", "gel", "spray", "powder", "liquid", "foam", "stick"],
    "skin_type": ["oily", "dry", "sensitive", "combination", "normal"],
    "finish_type": ["matte", "glossy", "satin", "natural", "metallic"],
}

# Common numeric attribute values (realistic ranges)
_NUMERIC_VARIANT_VALUES: dict[str, list[float]] = {
    "wattage": [5, 10, 15, 20, 30, 45, 65, 100, 140, 200],
    "weight_lbs": [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
    "screen_size_inches": [5.5, 6.1, 11.0, 13.3, 14.0, 15.6, 17.3, 24.0, 27.0],
}


def _match_category(product_cat: str) -> list[str]:
    """Match a product's slash-path category to CATEGORY_VARIANT_ATTRS.

    product.cat looks like "electronics/gaming/consoles"; CATEGORY_VARIANT_ATTRS
    keys are title-case like "Electronics". Match on the first segment.
    """
    if not product_cat:
        return []
    top_level = product_cat.split("/")[0].strip().title()
    # Try exact match first, then prefix match
    if top_level in CATEGORY_VARIANT_ATTRS:
        return CATEGORY_VARIANT_ATTRS[top_level]
    for key in CATEGORY_VARIANT_ATTRS:
        if key.lower().startswith(top_level.lower()):
            return CATEGORY_VARIANT_ATTRS[key]
    return []


def _pick_variant_attr(
    product: Product, rng: random.Random, seed: int = 42,
) -> tuple[str, list[str] | None]:
    """Pick a variant attribute for this product with diversity.

    Combines hardcoded attrs, product attrs, AND LLM-generated category-
    specific attrs. Returns (attr_name, value_pool_or_None).
    When value_pool is not None, use it directly in _synthesize_variants.
    """
    category_prefs = _match_category(product.cat)

    # Collect candidates: {attr_name: (weight, value_pool_or_None)}
    candidates: dict[str, tuple[float, list[str] | None]] = {}

    # ---- LLM-generated attributes (highest priority for diversity) ----
    llm_attrs = generate_variant_attrs_for_category(
        product.cat, product_title=product.title, seed=seed,
    )
    for attr_name, values in llm_attrs.items():
        if attr_name not in candidates and len(values) >= 3:
            candidates[attr_name] = (4.0, values)  # highest weight

    # ---- Hardcoded category-preferred attrs the product has ----
    for attr in category_prefs:
        if attr in product.attrs and attr not in candidates:
            weight = 1.0 if attr in ("color", "size", "material") else 3.0
            candidates[attr] = (weight, None)

    # ---- Fallback attrs the product has ----
    for attr in _FALLBACK_VARIANT_ATTRS:
        if attr in product.attrs and attr not in candidates:
            weight = 1.0 if attr in ("color", "size", "material") else 3.0
            candidates[attr] = (weight, None)

    # ---- Always offer color/size as synthesizable ----
    for attr in ("color", "size"):
        if attr not in candidates:
            candidates[attr] = (0.5, None)

    if not candidates:
        return "color", None

    attrs = list(candidates.keys())
    weights = [candidates[a][0] for a in attrs]
    chosen = rng.choices(attrs, weights=weights, k=1)[0]
    return chosen, candidates[chosen][1]


def _synthesize_variants(
    product: Product,
    n_variants: int,
    rng: random.Random,
) -> tuple[list[Variant], Variant]:
    """Synthesize N variants for a product by varying one category-appropriate attribute.

    Returns (all_variants, target_variant). The target preserves the product's
    actual attribute value when available. Distractor variants get plausible
    alternative values from predefined value pools.

    Args:
        product:    The product to generate variants for.
        n_variants: Total number of variants (including the target).
        rng:        Seeded RNG for reproducibility.

    Returns:
        Tuple of (list of all Variant objects, the target Variant).
    """
    attr_name, llm_pool = _pick_variant_attr(product, rng, seed=rng.randint(0, 2**31))
    attr_spec = ATTRIBUTE_ALLOWLIST.get(attr_name, {})
    is_numeric = attr_spec.get("type") == "numeric"

    # --- Gather the value pool ---
    actual_value: Any = product.attrs.get(attr_name)

    if llm_pool is not None:
        # LLM-generated attribute: use the LLM-provided values
        pool = list(llm_pool)
        # actual_value not meaningful for LLM attrs — pick from pool
        actual_value = None
    elif is_numeric and attr_name in _NUMERIC_VARIANT_VALUES:
        pool = list(_NUMERIC_VARIANT_VALUES[attr_name])
        # Ensure actual value is in the pool
        if actual_value is not None:
            try:
                actual_num = float(actual_value)
                if actual_num not in pool:
                    pool.append(actual_num)
                    pool.sort()
            except (ValueError, TypeError):
                actual_value = None
    elif attr_name in _SYNTHETIC_VALUES:
        pool = list(_SYNTHETIC_VALUES[attr_name])
    else:
        # Try ATTRIBUTE_ALLOWLIST values
        pool = list(attr_spec.get("values", []))

    # If pool is still empty, fall back to color
    if not pool:
        attr_name = "color"
        pool = list(_SYNTHETIC_VALUES["color"])
        actual_value = product.attrs.get("color")

    # --- Pick target value ---
    if actual_value is not None and actual_value in pool:
        target_value = actual_value
    elif actual_value is not None:
        # Product has the attr but value isn't in our pool — add it
        pool.append(actual_value)
        target_value = actual_value
    else:
        target_value = rng.choice(pool)

    # --- Pick distractor values ---
    others = [v for v in pool if v != target_value]
    rng.shuffle(others)
    distractor_values = others[: n_variants - 1]

    # Assemble all values: target first, then distractors
    all_values = [target_value] + distractor_values
    # Shuffle so target isn't always index 0
    rng.shuffle(all_values)

    # --- Build Variant objects ---
    variants: list[Variant] = []
    target_variant: Variant | None = None

    for i, val in enumerate(all_values):
        # Sanitize value for the variant_id
        val_str = str(val).replace(" ", "_").replace("/", "_")[:20]
        vid = f"{product.id}_v_{attr_name}_{val_str}"

        v_color = str(val) if attr_name == "color" else None
        v_size = str(val) if attr_name == "size" else None
        v_attrs: dict[str, Any] = {}
        if attr_name not in ("color", "size"):
            v_attrs[attr_name] = val

        # Small price delta for non-target numeric variants
        price_delta = 0.0
        if is_numeric and val != target_value:
            try:
                price_delta = round((float(val) - float(target_value)) * 0.1, 2)
            except (ValueError, TypeError):
                pass

        variant = Variant(
            variant_id=vid,
            product_id=product.id,
            color=v_color,
            size=v_size,
            attrs=v_attrs,
            price_delta=price_delta,
            stock_qty=max(1, product.stock_qty),
        )
        variants.append(variant)
        if val == target_value:
            target_variant = variant

    assert target_variant is not None, "Target variant must be in the list"
    return variants, target_variant


def _variant_description(variant: Variant) -> str:
    """Build a human-readable description of a variant's differentiating attributes."""
    parts: list[str] = []
    if variant.color:
        parts.append(f"color: {variant.color}")
    if variant.size:
        parts.append(f"size: {variant.size}")
    for k, v in variant.attrs.items():
        parts.append(f"{k}: {v}")
    return ", ".join(parts) if parts else "default"


def _product_to_visit_card(product: Product) -> dict[str, Any]:
    """Build a lightweight visit-history card from a Product.

    Returns the kind of info a "recently viewed" page would show:
    product_id, title, price, category, brand, and key attributes.
    """
    key_attrs: dict[str, Any] = {}
    for attr_key in ("color", "size", "material", "connector_type"):
        val = product.attrs.get(attr_key)
        if val:
            key_attrs[attr_key] = val

    return {
        "product_id": product.id,
        "title": product.title,
        "price": product.price,
        "category": product.cat,
        "brand": product.brand,
        "rating": product.rating,
        "key_attrs": key_attrs,
    }


@register_env
class CartBuildingEnv(BaseEnvironment):
    """E_CART: Cart Building -- add correct items/variants/qty to cart."""

    ENV_ID = "CART"

    # ------------------------------------------------------------------
    # P_d : Problem generator
    # ------------------------------------------------------------------

    def generate_problem(
        self,
        difficulty: int,
        catalog: list[Product],
        seed: int,
        **kwargs: Any,
    ) -> ProblemParams:
        """Sample n_items target products with optional variants and quantities.

        Spec Section 5 / E3 generator:
            n_items(d) = min(5, 1 + floor(d/3))
            p_var(d) = sigmoid((d-2)/1.5)
            p_qty(d) = min(0.5, 0.1*d)
        """
        dp = map_difficulty(difficulty)
        rng = random.Random(seed)

        n_items = min(5, 1 + math.floor(difficulty / 3))
        p_var = sigmoid((difficulty - 2) / 1.5)
        p_qty = min(0.5, 0.1 * difficulty)

        eligible = [p for p in catalog if p.cat not in DENIED_CATEGORIES and avail(p)]
        if len(eligible) < n_items:
            eligible = [p for p in catalog if p.cat not in DENIED_CATEGORIES]
        if len(eligible) < n_items:
            eligible = catalog

        targets = rng.sample(eligible, min(n_items, len(eligible)))

        # Build required items: {product_id -> qty}
        # Also track variant requirements and synthetic variants
        required_items: dict[str, int] = {}
        variant_reqs: dict[str, str | None] = {}
        item_details: list[dict[str, Any]] = []
        all_synthetic_variants: list[Variant] = []

        # Number of variants per product: 3 (1 target + 2 distractors)
        n_variants_per_product = 3

        for product in targets:
            # Quantity
            if rng.random() < p_qty:
                qty = rng.randint(2, 4)
            else:
                qty = 1

            # Variant — synthesize full Variant objects when triggered
            variant_id: str | None = None
            variant_desc: str | None = None
            product_variants: list[Variant] = []

            if rng.random() < p_var:
                product_variants, target_variant = _synthesize_variants(
                    product=product,
                    n_variants=n_variants_per_product,
                    rng=rng,
                )
                variant_id = target_variant.variant_id
                variant_desc = _variant_description(target_variant)
                all_synthetic_variants.extend(product_variants)

            required_items[product.id] = qty
            variant_reqs[product.id] = variant_id

            # Build descriptive clues (category, brand, key features)
            # These are what the user simulator uses instead of exact titles
            desc_parts: list[str] = []
            if product.cat and product.cat != "general":
                # Use the leaf category for conciseness
                cat_leaf = product.cat.split("/")[-1] if "/" in product.cat else product.cat
                desc_parts.append(cat_leaf)
            if product.brand and product.brand.lower() != "unknown":
                desc_parts.append(product.brand)
            # Add 1-2 distinguishing features from the product
            if product.features:
                for feat in product.features[:2]:
                    short_feat = feat[:60].strip()
                    if short_feat:
                        desc_parts.append(short_feat)

            item_details.append({
                "product_id": product.id,
                "title": product.title,
                "category": product.cat,
                "brand": product.brand,
                "price": product.price,
                "features": product.features[:3] if product.features else [],
                "description": desc_parts,
                "qty": qty,
                "variant_id": variant_id,
                "variant_desc": variant_desc,
            })

        # ---------------------------------------------------------
        # Build visit history: target items + distractor products
        # The user.get_visit_history tool returns this shuffled list
        # ---------------------------------------------------------
        target_ids = {p.id for p in targets}
        n_distractors = max(3, n_items * 2)  # 2x targets + padding

        # Pick distractors from same categories as targets for realism
        target_cats = {p.cat for p in targets}
        distractor_pool = [
            p for p in eligible
            if p.id not in target_ids and p.cat in target_cats
        ]
        # If not enough same-category, broaden
        if len(distractor_pool) < n_distractors:
            distractor_pool = [
                p for p in eligible if p.id not in target_ids
            ]

        distractors = rng.sample(
            distractor_pool,
            min(n_distractors, len(distractor_pool)),
        )

        visit_entries: list[dict[str, Any]] = []
        for p in targets:
            visit_entries.append(_product_to_visit_card(p))
        for p in distractors:
            visit_entries.append(_product_to_visit_card(p))
        rng.shuffle(visit_entries)

        return ProblemParams(
            env_id=self.ENV_ID,
            difficulty=difficulty,
            seed=seed,
            target_product_ids=[p.id for p in targets],
            constraints=[],
            extra={
                "n_items": n_items,
                "p_var": p_var,
                "p_qty": p_qty,
                "T_max": dp.T_max_val,
                "p_missing": dp.p_missing_val,
                "p_noise": dp.p_noise_val,
                "required_items": required_items,
                "variant_reqs": variant_reqs,
                "item_details": item_details,
                "visit_history": visit_entries,
                "synthetic_variants": [
                    v.model_dump() for v in all_synthetic_variants
                ],
            },
        )

    # ------------------------------------------------------------------
    # I : Input generator
    # ------------------------------------------------------------------

    def generate_input(self, params: ProblemParams) -> str:
        """Render cart building request.

        Fix: Uses LLM verbalization (via Qwen3.5) first, with strategic
        omission of variant/quantity details controlled by p_missing.
        Falls back to template-based rendering if LLM is unavailable.
        """
        details: list[dict[str, Any]] = params.extra["item_details"]
        p_missing = params.extra.get("p_missing", 0.0)

        # --- LLM-based verbalization (preferred) ---
        text, mentioned, omitted = verbalize_cart_request(
            item_details=details,
            p_missing=p_missing,
            seed=params.seed + 1,
        )
        if text is not None:
            params.extra["mentioned_attrs"] = mentioned
            params.extra["omitted_attrs"] = omitted
            return text

        # --- Fallback: template-based generation ---
        # Use short product-type from title, NOT full title or raw category
        item_parts: list[str] = []
        for d in details:
            brand = d.get("brand", "")
            product_type = extract_product_type(d["title"], brand)
            if brand and brand.lower() not in ("unknown", "generic", "unbranded", ""):
                short_desc = f"{product_type} by {brand}"
            else:
                short_desc = product_type
            part = short_desc
            if d["qty"] > 1:
                part += f" (x{d['qty']})"
            item_parts.append(part)
        item_list = ", ".join(item_parts)

        # Build variant and quantity detail strings (using short desc)
        variant_parts: list[str] = []
        qty_parts: list[str] = []
        for d in details:
            cat = d.get("category", "")
            brand = d.get("brand", "")
            cat_leaf = cat.split("/")[-1] if "/" in cat else cat
            if cat_leaf.isupper():
                cat_leaf = cat_leaf.title()
            label = cat_leaf or d["title"][:30]
            if brand and brand.lower() not in ("unknown", "generic", "unbranded", ""):
                label += f" by {brand}"
            if d["variant_desc"]:
                variant_parts.append(f"{label}: {d['variant_desc']}")
            if d["qty"] > 1:
                qty_parts.append(f"{label}: qty {d['qty']}")

        template_params: dict[str, Any] = {"item_list": item_list}
        if variant_parts:
            template_params["variant_details"] = "; ".join(variant_parts)
        if qty_parts:
            template_params["quantity_details"] = "; ".join(qty_parts)

        return render_template(
            env_id=self.ENV_ID,
            params=template_params,
            p_missing=params.extra.get("p_missing", 0.0),
            p_noise=params.extra.get("p_noise", 0.0),
            seed=params.seed + 1,
        )

    # ------------------------------------------------------------------
    # R : Verifier
    # ------------------------------------------------------------------

    def verify(
        self,
        answer: dict[str, Any],
        params: ProblemParams,
        episode_state: dict[str, Any],
    ) -> EpisodeResult:
        """Compute r_task per Spec Section 5 / E3.

        r_task = 2*F1 - 1 (unit-level, variant-aware)
        IsCorrect = F1 == 1.0
        """
        required_items: dict[str, int] = params.extra["required_items"]
        variant_reqs: dict[str, str | None] = params.extra.get("variant_reqs", {})

        # Support both cart_lines (list[dict]) and cart (CartState object)
        cart_lines: list[dict[str, Any]] = episode_state.get("cart_lines", [])
        if not cart_lines:
            cart_obj = episode_state.get("cart")
            if cart_obj is not None and hasattr(cart_obj, "lines"):
                cart_lines = [
                    {"product_id": l.product_id, "variant_id": l.variant_id, "qty": l.qty}
                    for l in cart_obj.lines
                ]

        products_by_id: dict[str, Product] = episode_state.get("products_by_id", {})

        r_task, is_correct = verify_cart(
            required_items=required_items,
            variant_reqs=variant_reqs,
            cart_lines=cart_lines,
            products_by_id=products_by_id,
        )

        return EpisodeResult(
            r_task=r_task,
            is_correct=is_correct,
            details={
                "n_items": params.extra.get("n_items"),
                "required_items": required_items,
                "variant_reqs": variant_reqs,
                "cart_lines_count": len(cart_lines),
            },
        )
