"""Canonical product schema and attribute definitions for EcomRLVE-GYM.

Product fields (internal canonical form):
- id, title, desc, cat, brand, attrs (dict), price, rating, ship_days, stock_qty
- avail(p) = 1[stock_qty > 0]
- rating_count, store, parent_asin, features  (from real dataset)

Real dataset mapping (Amazebay-catalog on HuggingFace):
- parent_asin  -> id (and parent_asin)
- title        -> title
- description  -> desc  (List[str] joined)
- main_category -> cat
- store / details.Brand -> brand
- details (JSON) -> attrs dict
- price (string, must parse) -> price (float)
- average_rating -> rating
- rating_number -> rating_count
- ship_days: SYNTHESIZED (not in real data)
- stock_qty: SYNTHESIZED (not in real data)
- features (List[str]) -> features
- store -> store

Attribute allowlist for constraints/filters:
- Discrete: cat, brand, color, size, material, connector_type,
            item_form, skin_type, finish_type, age_range
- Numeric: price, rating, ship_days, rating_count, wattage, weight_lbs,
           screen_size_inches

This module is THE canonical data model used everywhere in the project.
All tools, verifiers, and environments import from here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Real dataset column reference (Amazebay-catalog on HuggingFace)
# ---------------------------------------------------------------------------

REAL_DATASET_COLUMNS: list[str] = [
    "main_category",     # str   – top-level category (e.g. "All Beauty")
    "title",             # str   – product title
    "average_rating",    # float – average customer rating (1.0–5.0)
    "rating_number",     # int   – number of ratings
    "features",          # list[str] – bullet-point feature strings
    "description",       # list[str] – product description paragraphs
    "price",             # str   – price as string (may be "None")
    "images",            # struct – {hi_res, large, thumb, variant}
    "videos",            # struct – {title, url, user_id}
    "store",             # str   – seller/store name
    "categories",        # list  – category taxonomy path
    "details",           # str   – JSON-encoded dict of product attributes
    "parent_asin",       # str   – parent ASIN grouping variants
    "bought_together",   # nullable – products frequently bought together
    "subtitle",          # str (nullable) – product subtitle
    "author",            # str (nullable) – author (for books)
]


# ---------------------------------------------------------------------------
# Attribute allowlist: maps attribute name -> type + valid range/values
# ---------------------------------------------------------------------------

ATTRIBUTE_ALLOWLIST: dict[str, dict[str, Any]] = {
    # Discrete attributes (equality-matchable)
    "cat": {
        "type": "discrete",
        "description": "Product category / taxonomy ID",
    },
    "brand": {
        "type": "discrete",
        "description": "Brand name",
    },
    "color": {
        "type": "discrete",
        "description": "Primary color",
        "values": [
            "black", "white", "silver", "gray", "red", "blue", "green",
            "yellow", "orange", "purple", "pink", "brown", "gold", "navy",
            "teal", "beige", "multicolor",
        ],
    },
    "size": {
        "type": "discrete",
        "description": "Size label (XS, S, M, L, XL, XXL, or numeric)",
        "values": ["XS", "S", "M", "L", "XL", "XXL"],
    },
    "material": {
        "type": "discrete",
        "description": "Primary material",
        "values": [
            "cotton", "polyester", "leather", "metal", "plastic", "wood",
            "glass", "ceramic", "rubber", "silicone", "nylon", "stainless_steel",
            "aluminum", "carbon_fiber",
        ],
    },
    "connector_type": {
        "type": "discrete",
        "description": "Connector / interface type",
        "values": [
            "USB-A", "USB-C", "Lightning", "Micro-USB", "HDMI",
            "DisplayPort", "3.5mm", "Thunderbolt", "Bluetooth", "Wi-Fi",
        ],
    },
    # Discrete attributes extracted from real details JSON
    "item_form": {
        "type": "discrete",
        "description": "Product form factor (e.g. cream, gel, spray, powder, liquid)",
    },
    "skin_type": {
        "type": "discrete",
        "description": "Target skin type (e.g. oily, dry, sensitive, combination, all)",
    },
    "finish_type": {
        "type": "discrete",
        "description": "Surface/product finish (e.g. matte, glossy, satin, natural)",
    },
    "age_range": {
        "type": "discrete",
        "description": "Target age range (e.g. adult, teen, child, all ages)",
    },
    "store": {
        "type": "discrete",
        "description": "Seller/store name",
    },
    # Numeric attributes (range-matchable)
    "price": {
        "type": "numeric",
        "description": "Price in USD",
        "min": 0.01,
        "max": 100_000.0,
    },
    "rating": {
        "type": "numeric",
        "description": "Average customer rating",
        "min": 1.0,
        "max": 5.0,
    },
    "ship_days": {
        "type": "numeric",
        "description": "Estimated shipping days (SYNTHESIZED, not in real data)",
        "min": 0,
        "max": 90,
    },
    "rating_count": {
        "type": "numeric",
        "description": "Number of customer ratings (from rating_number); popularity signal",
        "min": 0,
        "max": 10_000_000,
    },
    "wattage": {
        "type": "numeric",
        "description": "Power consumption in watts",
        "min": 0.0,
        "max": 10_000.0,
    },
    "weight_lbs": {
        "type": "numeric",
        "description": "Product weight in pounds",
        "min": 0.0,
        "max": 500.0,
    },
    "screen_size_inches": {
        "type": "numeric",
        "description": "Screen diagonal in inches",
        "min": 0.0,
        "max": 120.0,
    },
}

# ---------------------------------------------------------------------------
# Filter keys: the subset of attribute names valid for catalog.search filters
# ---------------------------------------------------------------------------

FILTER_KEYS: set[str] = {
    # Discrete equality filters
    "cat",
    "brand",
    "color",
    "size",
    "material",
    "connector_type",
    "store",
    "item_form",
    "skin_type",
    "finish_type",
    # Numeric range filters (the search tool accepts _min / _max suffixed keys)
    "price_min",
    "price_max",
    "rating_min",
    "rating_max",
    "rating_count_min",
    "ship_days_max",
    "wattage_min",
    "wattage_max",
    "weight_lbs_max",
    "screen_size_inches_min",
    "screen_size_inches_max",
}

# ---------------------------------------------------------------------------
# Denied categories for safety checks (spec Section 6.2)
# If any recommended product's cat is in this set -> safety_violation = True
# ---------------------------------------------------------------------------

DENIED_CATEGORIES: set[str] = frozenset({
    "weapons",
    "firearms",
    "ammunition",
    "explosives",
    "adult_explicit",
    "controlled_substances",
    "tobacco",
    "counterfeit_goods",
    "surveillance_equipment",
    "hazardous_materials",
})


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class Product(BaseModel):
    """Full product representation (internal canonical form).

    Core fields:
        id:        Unique product identifier (from parent_asin in real data).
        title:     Product title string.
        desc:      Product description (joined from List[str] in real data).
        cat:       Category (from main_category in real data).
        brand:     Brand name (from details.Brand or store in real data).
        attrs:     Attribute dict (parsed from details JSON in real data).
        price:     Price in USD (parsed from string in real data).
        rating:    Average rating in [1, 5] (from average_rating).
        ship_days: Estimated shipping days (SYNTHESIZED, not in real data).
        stock_qty: Units in stock (SYNTHESIZED, not in real data).

    Fields from real dataset:
        rating_count: Number of ratings (from rating_number); popularity signal.
        store:        Seller/store name (from store column).
        parent_asin:  Parent ASIN for variant grouping.
        features:     Feature bullet strings (from features column).
    """

    id: str = Field(..., description="Unique product identifier (parent_asin)")
    title: str = Field(..., min_length=1, description="Product title")
    desc: str = Field(default="", description="Product description (joined from list)")
    cat: str = Field(default="general", description="Category (from main_category)")
    brand: str = Field(default="unknown", description="Brand name")
    attrs: dict[str, Any] = Field(default_factory=dict, description="Attribute key-value pairs (from details JSON)")
    price: float = Field(..., gt=0, description="Price in USD, must be > 0")
    rating: float = Field(default=3.0, ge=1.0, le=5.0, description="Rating in [1, 5]")
    ship_days: int = Field(default=5, ge=0, description="Estimated shipping days (synthesized)")
    stock_qty: int = Field(default=1, ge=0, description="Stock quantity (synthesized)")
    # Fields sourced from real Amazebay-catalog data
    rating_count: int = Field(default=0, ge=0, description="Number of ratings (from rating_number)")
    store: str = Field(default="", description="Seller/store name")
    parent_asin: str = Field(default="", description="Parent ASIN for variant grouping")
    features: list[str] = Field(default_factory=list, description="Feature bullet strings")

    @field_validator("price")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"price must be > 0, got {v}")
        return round(v, 2)

    @field_validator("rating")
    @classmethod
    def rating_in_range(cls, v: float) -> float:
        if not 1.0 <= v <= 5.0:
            raise ValueError(f"rating must be in [1, 5], got {v}")
        return round(v, 2)

    @field_validator("stock_qty")
    @classmethod
    def stock_qty_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"stock_qty must be >= 0, got {v}")
        return v


class ProductCard(BaseModel):
    """Lightweight product card returned by catalog.search (spec Section 2.2).

    Contains only the fields the LLM needs to make initial decisions,
    without the full description or complete attribute dict.
    """

    product_id: str = Field(..., description="Product identifier")
    title: str = Field(..., description="Product title")
    price: float = Field(..., description="Price in USD")
    rating: float = Field(..., description="Rating in [1, 5]")
    ship_days: int = Field(..., description="Estimated shipping days")
    stock_qty: int = Field(..., description="Stock quantity")
    key_attrs: dict[str, Any] = Field(
        default_factory=dict,
        description="Subset of important attributes (color, size, brand, etc.)",
    )


class Variant(BaseModel):
    """Product variant with specific color/size/other differentiating attributes.

    A product can have multiple variants (e.g., different colors/sizes).
    The variant_id is globally unique; product_id links back to the parent.
    """

    variant_id: str = Field(..., description="Unique variant identifier")
    product_id: str = Field(..., description="Parent product identifier")
    color: str | None = Field(default=None, description="Variant color")
    size: str | None = Field(default=None, description="Variant size")
    attrs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional variant-specific attributes",
    )
    price_delta: float = Field(
        default=0.0,
        description="Price adjustment relative to base product price",
    )
    stock_qty: int = Field(default=1, ge=0, description="Variant stock quantity")

    @model_validator(mode="after")
    def must_differ_from_base(self) -> Variant:
        """A variant should have at least one differentiating attribute."""
        if self.color is None and self.size is None and not self.attrs:
            raise ValueError(
                "Variant must have at least one differentiating attribute "
                "(color, size, or attrs)"
            )
        return self


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def product_to_card(product: Product) -> ProductCard:
    """Convert a full Product to a lightweight ProductCard for search results.

    Extracts the most relevant attributes (brand, color, size, material,
    connector_type, store, features preview) into key_attrs so the LLM has
    enough info to decide whether to fetch the full product details.
    """
    # Keys to surface in the card (high-signal attributes)
    card_attr_keys = {"brand", "color", "size", "material", "connector_type"}

    key_attrs: dict[str, Any] = {}
    # Always include brand at top level if known
    if product.brand and product.brand != "unknown":
        key_attrs["brand"] = product.brand
    # Pull relevant attrs from the product's attribute dict
    for key in card_attr_keys - {"brand"}:
        if key in product.attrs:
            key_attrs[key] = product.attrs[key]
    # Surface store name if available
    if product.store:
        key_attrs["store"] = product.store
    # Surface first 2 feature bullets as preview
    if product.features:
        key_attrs["features"] = product.features[:2]

    return ProductCard(
        product_id=product.id,
        title=product.title,
        price=product.price,
        rating=product.rating,
        ship_days=product.ship_days,
        stock_qty=product.stock_qty,
        key_attrs=key_attrs,
    )


def avail(product: Product) -> bool:
    """Availability indicator: avail(p) = 1[stock_qty(p) > 0].

    Spec Section 1.1:
        avail(p) = 1[stock_qty > 0]
    """
    return product.stock_qty > 0
