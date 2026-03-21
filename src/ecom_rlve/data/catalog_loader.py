"""Catalog and persona loading for EcomRLVE-GYM.

Provides three entry points:

1. load_catalog() -- Load the Amazebay catalog from HuggingFace and map
   to our canonical Product schema.

2. generate_synthetic_catalog() -- Generate fake products with realistic
   distributions for testing without downloading real data. This is a
   critical DEBUG lever.

3. load_personas() -- Load the Nemotron Personas USA dataset for persona
   sampling in the user simulator.

Real dataset schemas:
    - Amazebay-catalog (16 columns): parent_asin, title, average_rating,
      rating_number, features (list[str]), description (list[str]),
      price (string), images (struct), videos (struct), store, categories,
      details (JSON string), bought_together, subtitle, author, main_category.
    - Nemotron-Personas-USA (23 columns): uuid, professional_persona,
      sports_persona, arts_persona, travel_persona, culinary_persona,
      persona (general), cultural_background, skills_and_expertise,
      skills_and_expertise_list, hobbies_and_interests,
      hobbies_and_interests_list, career_goals_and_ambitions, sex, age,
      marital_status, education_level, bachelors_field, occupation, city,
      state, zipcode, country.

Spec references:
    - Section 1.1: Product schema (id, title, desc, cat, brand, attrs, price,
      rating, ship_days, stock_qty).
    - AGENTS.md: datasets = thebajajra/Amazebay-catalog, nvidia/Nemotron-Personas-USA.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from typing import Any

import numpy as np

from ecom_rlve.data.schema import ATTRIBUTE_ALLOWLIST, Product, Variant

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants for synthetic catalog generation
# ---------------------------------------------------------------------------

_BRANDS = [
    "TechPro", "NovaStar", "EcoVibe", "UrbanEdge", "PrimeFit",
    "LuxeHome", "SwiftGear", "PureBlend", "SonicWave", "BrightPath",
    "ZenCraft", "FusionLab", "CoreMax", "StellarByte", "GreenLeaf",
    "IronClad", "CloudNine", "BluePeak", "FireForge", "AquaPure",
    "PixelVault", "ThunderBolt", "CrystalClear", "WildTrail", "SilkWay",
]

_CATEGORIES = [
    "electronics/audio/headphones",
    "electronics/audio/speakers",
    "electronics/computing/laptops",
    "electronics/computing/monitors",
    "electronics/computing/keyboards",
    "electronics/computing/mice",
    "electronics/mobile/phones",
    "electronics/mobile/tablets",
    "electronics/mobile/chargers",
    "electronics/mobile/cases",
    "electronics/cameras/dslr",
    "electronics/cameras/action",
    "electronics/gaming/consoles",
    "electronics/gaming/controllers",
    "electronics/smart_home/lights",
    "electronics/smart_home/plugs",
    "home/kitchen/appliances",
    "home/kitchen/cookware",
    "home/kitchen/utensils",
    "home/furniture/desks",
    "home/furniture/chairs",
    "home/furniture/shelves",
    "home/decor/lighting",
    "home/decor/rugs",
    "clothing/mens/shirts",
    "clothing/mens/pants",
    "clothing/womens/dresses",
    "clothing/womens/tops",
    "clothing/shoes/sneakers",
    "clothing/shoes/boots",
    "sports/fitness/weights",
    "sports/fitness/yoga",
    "sports/outdoor/camping",
    "sports/outdoor/hiking",
    "beauty/skincare/moisturizers",
    "beauty/skincare/sunscreen",
    "beauty/hair/shampoo",
    "beauty/hair/styling",
    "toys/educational/stem",
    "toys/games/board",
]

_COLORS = ATTRIBUTE_ALLOWLIST["color"]["values"]
_SIZES = ATTRIBUTE_ALLOWLIST["size"]["values"]
_MATERIALS = ATTRIBUTE_ALLOWLIST["material"]["values"]
_CONNECTORS = ATTRIBUTE_ALLOWLIST["connector_type"]["values"]

_ADJECTIVES = [
    "Premium", "Ultra", "Professional", "Compact", "Wireless", "Smart",
    "Advanced", "Portable", "Heavy-Duty", "Lightweight", "Ergonomic",
    "High-Performance", "All-in-One", "Multi-Purpose", "Eco-Friendly",
    "Classic", "Modern", "Industrial", "Deluxe", "Essential",
]

_PRODUCT_NOUNS = {
    "electronics/audio/headphones": ["Headphones", "Earbuds", "Earphones"],
    "electronics/audio/speakers": ["Speaker", "Soundbar", "Subwoofer"],
    "electronics/computing/laptops": ["Laptop", "Notebook", "Ultrabook"],
    "electronics/computing/monitors": ["Monitor", "Display", "Screen"],
    "electronics/computing/keyboards": ["Keyboard", "Keypad"],
    "electronics/computing/mice": ["Mouse", "Trackball", "Trackpad"],
    "electronics/mobile/phones": ["Smartphone", "Phone"],
    "electronics/mobile/tablets": ["Tablet", "E-Reader"],
    "electronics/mobile/chargers": ["Charger", "Power Adapter", "Power Bank"],
    "electronics/mobile/cases": ["Phone Case", "Tablet Cover"],
    "electronics/cameras/dslr": ["DSLR Camera", "Mirrorless Camera"],
    "electronics/cameras/action": ["Action Camera", "GoPro-Style Camera"],
    "electronics/gaming/consoles": ["Gaming Console", "Game System"],
    "electronics/gaming/controllers": ["Controller", "Gamepad", "Joystick"],
    "electronics/smart_home/lights": ["Smart Bulb", "LED Strip", "Smart Light"],
    "electronics/smart_home/plugs": ["Smart Plug", "Smart Switch"],
    "home/kitchen/appliances": ["Blender", "Toaster", "Coffee Maker", "Air Fryer"],
    "home/kitchen/cookware": ["Pan", "Pot Set", "Skillet", "Wok"],
    "home/kitchen/utensils": ["Knife Set", "Spatula Set", "Cutting Board"],
    "home/furniture/desks": ["Standing Desk", "Computer Desk", "Writing Desk"],
    "home/furniture/chairs": ["Office Chair", "Ergonomic Chair", "Bar Stool"],
    "home/furniture/shelves": ["Bookshelf", "Wall Shelf", "Storage Rack"],
    "home/decor/lighting": ["Floor Lamp", "Table Lamp", "Pendant Light"],
    "home/decor/rugs": ["Area Rug", "Runner", "Mat"],
    "clothing/mens/shirts": ["T-Shirt", "Dress Shirt", "Polo Shirt"],
    "clothing/mens/pants": ["Jeans", "Chinos", "Joggers"],
    "clothing/womens/dresses": ["Maxi Dress", "Cocktail Dress", "Sundress"],
    "clothing/womens/tops": ["Blouse", "Tank Top", "Sweater"],
    "clothing/shoes/sneakers": ["Running Shoes", "Sneakers", "Training Shoes"],
    "clothing/shoes/boots": ["Hiking Boots", "Ankle Boots", "Work Boots"],
    "sports/fitness/weights": ["Dumbbell Set", "Kettlebell", "Weight Bench"],
    "sports/fitness/yoga": ["Yoga Mat", "Yoga Blocks", "Resistance Bands"],
    "sports/outdoor/camping": ["Tent", "Sleeping Bag", "Camp Stove"],
    "sports/outdoor/hiking": ["Backpack", "Trekking Poles", "Water Bottle"],
    "beauty/skincare/moisturizers": ["Face Cream", "Moisturizer", "Serum"],
    "beauty/skincare/sunscreen": ["Sunscreen", "Sun Block", "UV Lotion"],
    "beauty/hair/shampoo": ["Shampoo", "Conditioner", "Hair Wash"],
    "beauty/hair/styling": ["Hair Dryer", "Flat Iron", "Curling Wand"],
    "toys/educational/stem": ["Robot Kit", "Chemistry Set", "Telescope"],
    "toys/games/board": ["Board Game", "Card Game", "Puzzle"],
}

# ---------------------------------------------------------------------------
# Key normalization map: real Amazebay details keys -> ATTRIBUTE_ALLOWLIST keys
# ---------------------------------------------------------------------------

_DETAILS_KEY_MAP: dict[str, str] = {
    "brand": "brand",
    "Brand": "brand",
    "manufacturer": "manufacturer",
    "Manufacturer": "manufacturer",
    "color": "color",
    "Color": "color",
    "Colour": "color",
    "size": "size",
    "Size": "size",
    "material": "material",
    "Material": "material",
    "Fabric Type": "material",
    "connector_type": "connector_type",
    "Connector Type": "connector_type",
    "Item Form": "item_form",
    "Skin Type": "skin_type",
    "Finish Type": "finish_type",
    "Age Range (Description)": "age_range",
    "Product Benefits": "product_benefits",
    "Unit Count": "unit_count",
    "Item Weight": "weight_lbs",
    "Package Dimensions": "package_dimensions",
    "UPC": "upc",
    "Wattage": "wattage",
    "Screen Size": "screen_size_inches",
}


# ---------------------------------------------------------------------------
# Helper: parse price from string
# ---------------------------------------------------------------------------


def _parse_price(price_str: str | None) -> float:
    """Parse price from strings like '$12.99', 'None', '1,299.00'.

    The real Amazebay-catalog stores prices as strings with various formats:
        - "None" or empty string -> missing price
        - "$12.99" -> 12.99
        - "12.99"  -> 12.99
        - "$1,299.00" -> 1299.00

    Returns:
        Parsed float price, or 9.99 as a default for unparseable values.
    """
    if not price_str or str(price_str).strip().lower() in ("none", ""):
        return 9.99  # default for missing
    cleaned = str(price_str).replace("$", "").replace(",", "").strip()
    try:
        p = float(cleaned)
        return p if p > 0 else 9.99
    except ValueError:
        return 9.99


# ---------------------------------------------------------------------------
# Helper: parse details JSON string
# ---------------------------------------------------------------------------


def _parse_details(details_raw: str | dict | None) -> dict[str, Any]:
    """Parse the JSON-encoded details string into a normalized attrs dict.

    Real Amazebay details column contains JSON-encoded dicts like:
        '{"Brand": "Yes To", "Item Form": "Powder", "Skin Type": "Acne Prone"}'
        '{"Package Dimensions": "7.1 x 5.5 x 3 inches", "UPC": "617390882781"}'

    This function:
    1. Parses JSON string (or accepts dict directly).
    2. Normalizes keys to lowercase_snake_case using _DETAILS_KEY_MAP.
    3. Lowercases discrete values for consistency.

    Returns:
        Dict of normalized attribute key -> value.
    """
    if details_raw is None:
        return {}

    # Parse JSON string if needed
    if isinstance(details_raw, str):
        details_raw = details_raw.strip()
        if not details_raw or details_raw.lower() == "none":
            return {}
        try:
            parsed = json.loads(details_raw)
            if not isinstance(parsed, dict):
                return {}
        except (json.JSONDecodeError, TypeError):
            return {}
    elif isinstance(details_raw, dict):
        parsed = dict(details_raw)
    else:
        return {}

    attrs: dict[str, Any] = {}
    for raw_key, value in parsed.items():
        # Normalize key using mapping, or convert to snake_case
        if raw_key in _DETAILS_KEY_MAP:
            norm_key = _DETAILS_KEY_MAP[raw_key]
        else:
            # Convert "Some Key Name" -> "some_key_name"
            norm_key = re.sub(r"[^a-z0-9]+", "_", raw_key.lower()).strip("_")

        if not norm_key or value is None:
            continue

        # Normalize known discrete values to lowercase
        if norm_key in ("color", "material", "item_form", "skin_type",
                        "finish_type", "age_range", "brand", "manufacturer"):
            attrs[norm_key] = str(value).strip().lower()
        elif norm_key == "wattage":
            # Extract numeric wattage from strings like "60 watts"
            try:
                attrs[norm_key] = float(re.sub(r"[^0-9.]", "", str(value)))
            except (ValueError, TypeError):
                pass
        elif norm_key == "weight_lbs":
            # Extract numeric weight from strings like "2.38 Pounds"
            try:
                num_match = re.search(r"[\d.]+", str(value))
                if num_match:
                    weight = float(num_match.group())
                    # Convert ounces to lbs if value seems like ounces (> 100)
                    if "ounce" in str(value).lower() or "oz" in str(value).lower():
                        weight = weight / 16.0
                    attrs[norm_key] = round(weight, 2)
            except (ValueError, TypeError):
                pass
        elif norm_key == "screen_size_inches":
            # Extract numeric screen size from strings like '27 inches'
            try:
                num_match = re.search(r"[\d.]+", str(value))
                if num_match:
                    attrs[norm_key] = float(num_match.group())
            except (ValueError, TypeError):
                pass
        else:
            # Store as string for everything else
            attrs[norm_key] = str(value).strip()

    return attrs


# ---------------------------------------------------------------------------
# Helper: synthesize fields not present in real data
# ---------------------------------------------------------------------------


def _hash_seed(product_id: str, salt: str = "") -> int:
    """Derive a deterministic integer seed from a product_id string."""
    h = hashlib.sha256(f"{product_id}{salt}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _synthesize_ship_days(product_id: str) -> int:
    """Synthesize ship_days deterministically from product_id hash.

    Distribution: geometric(p=0.4), so most products ship in 1-3 days.
    Clamped to [1, 14].

    Args:
        product_id: Product identifier used as random seed.

    Returns:
        Estimated shipping days in [1, 14].
    """
    seed = _hash_seed(product_id, salt="ship")
    rng = np.random.RandomState(seed)
    days = int(rng.geometric(p=0.4))
    return max(1, min(days, 14))


def _synthesize_stock(product_id: str) -> int:
    """Synthesize stock_qty deterministically from product_id hash.

    Distribution: 90% in-stock with Poisson(20), 10% out-of-stock (0).
    Clamped to [0, 200].

    Args:
        product_id: Product identifier used as random seed.

    Returns:
        Stock quantity (0 means out-of-stock).
    """
    seed = _hash_seed(product_id, salt="stock")
    rng = np.random.RandomState(seed)
    if rng.random() < 0.10:
        return 0  # Out of stock
    qty = int(rng.poisson(20))
    return max(1, min(qty, 200))


# ---------------------------------------------------------------------------
# HuggingFace catalog loading
# ---------------------------------------------------------------------------


def load_catalog(
    dataset_name: str = "thebajajra/Amazebay-catalog",
    split: str = "train",
    max_items: int | None = None,
    seed: int = 42,
) -> list[Product]:
    """Load the Amazebay catalog from HuggingFace and map to Product schema.

    Maps HuggingFace dataset columns to our canonical Product model.
    Missing fields are filled with sensible defaults so we never crash
    on incomplete rows.

    The real Amazebay-catalog has 16 columns:
        main_category, title, average_rating, rating_number, features,
        description, price, images, videos, store, categories, details,
        parent_asin, bought_together, subtitle, author.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split:        Dataset split to load (default "train").
        max_items:    If set, randomly sample this many items (for development).
        seed:         Random seed for reproducible sampling.

    Returns:
        List of Product instances.
    """
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install with: uv add datasets"
        ) from exc

    logger.info(
        "Loading catalog: %s (split=%s, max_items=%s)",
        dataset_name,
        split,
        max_items,
    )

    # Support both local saved datasets (from save_to_disk) and HuggingFace hub
    from pathlib import Path

    local_path = Path(dataset_name)
    if local_path.is_dir() and (local_path / "dataset_info.json").exists():
        logger.info("Detected local saved dataset at %s", local_path)
        ds = load_from_disk(str(local_path))
    else:
        ds = load_dataset(dataset_name, split=split)

    # If sampling, shuffle and select
    if max_items is not None and max_items < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_items))

    products: list[Product] = []
    columns = set(ds.column_names)

    for row in ds:
        try:
            product = _map_hf_row_to_product(row, columns)
            products.append(product)
        except Exception as exc:
            logger.debug("Skipping row due to mapping error: %s", exc)
            continue

    logger.info("Loaded %d products from %s", len(products), dataset_name)
    return products


def _map_hf_row_to_product(row: dict[str, Any], columns: set[str]) -> Product:
    """Map a single HuggingFace dataset row to a Product.

    Handles the REAL Amazebay-catalog schema with 16 columns:
        parent_asin, title, average_rating, rating_number, features,
        description (list[str]), price (string), images (struct),
        videos (struct), store, categories, details (JSON string),
        main_category, bought_together, subtitle, author.

    Fields not in the real data (ship_days, stock_qty) are synthesized
    deterministically from the product_id hash for reproducibility.
    """
    # ------------------------------------------------------------------
    # ID: parent_asin (unique product grouping key)
    # ------------------------------------------------------------------
    product_id = str(row.get("parent_asin", "") or uuid.uuid4())

    # ------------------------------------------------------------------
    # Title: direct string field
    # ------------------------------------------------------------------
    title = str(row.get("title", "") or "Untitled Product")

    # ------------------------------------------------------------------
    # Description: JOIN list[str] into a single paragraph
    # ------------------------------------------------------------------
    desc_raw = row.get("description", [])
    if isinstance(desc_raw, list):
        desc = " ".join(str(s) for s in desc_raw if s)
    else:
        desc = str(desc_raw or "")

    # ------------------------------------------------------------------
    # Category: main_category (e.g. "All Beauty", "Electronics")
    # ------------------------------------------------------------------
    cat = str(row.get("main_category", "") or "general")

    # ------------------------------------------------------------------
    # Price: parse string like "$12.99", "None", "12.99", "$1,299.00"
    # ------------------------------------------------------------------
    price = _parse_price(row.get("price", "None"))

    # ------------------------------------------------------------------
    # Rating: average_rating (float in [1.0, 5.0])
    # ------------------------------------------------------------------
    try:
        rating_raw = row.get("average_rating", 3.0)
        rating = float(rating_raw) if rating_raw is not None else 3.0
        rating = max(1.0, min(5.0, rating))
    except (TypeError, ValueError):
        rating = 3.0

    # ------------------------------------------------------------------
    # Rating count: rating_number (int, popularity signal)
    # ------------------------------------------------------------------
    try:
        rc_raw = row.get("rating_number", 0)
        rating_count = int(rc_raw) if rc_raw is not None else 0
        rating_count = max(0, rating_count)
    except (TypeError, ValueError):
        rating_count = 0

    # ------------------------------------------------------------------
    # Store: seller/store name
    # ------------------------------------------------------------------
    store = str(row.get("store", "") or "")

    # ------------------------------------------------------------------
    # Features: list[str] of bullet points
    # ------------------------------------------------------------------
    features_raw = row.get("features", [])
    if isinstance(features_raw, list):
        features = [str(f) for f in features_raw if f]
    else:
        features = []

    # ------------------------------------------------------------------
    # Details: parse JSON-encoded dict to extract Brand, Material, etc.
    # ------------------------------------------------------------------
    details_raw = row.get("details", "{}")
    attrs = _parse_details(details_raw)

    # ------------------------------------------------------------------
    # Brand: from details["brand"] or details["manufacturer"] or store
    # ------------------------------------------------------------------
    brand = (
        attrs.pop("brand", None)
        or attrs.pop("manufacturer", None)
        or store
        or "unknown"
    )

    # ------------------------------------------------------------------
    # Synthesized fields (not present in real data)
    # ------------------------------------------------------------------
    ship_days = _synthesize_ship_days(product_id)
    stock_qty = _synthesize_stock(product_id)

    # ------------------------------------------------------------------
    # Parent ASIN for variant grouping
    # ------------------------------------------------------------------
    parent_asin = str(row.get("parent_asin", "") or "")

    return Product(
        id=product_id,
        title=title,
        desc=desc,
        cat=cat,
        brand=brand,
        attrs=attrs,
        price=price,
        rating=round(rating, 2),
        ship_days=ship_days,
        stock_qty=stock_qty,
        rating_count=rating_count,
        store=store,
        parent_asin=parent_asin,
        features=features,
    )


# ---------------------------------------------------------------------------
# Synthetic catalog generation (DEBUG lever)
# ---------------------------------------------------------------------------


def generate_synthetic_catalog(
    n_products: int = 1000,
    seed: int = 42,
) -> tuple[list[Product], list[Variant]]:
    """Generate a synthetic product catalog for testing without real data.

    This is a critical DEBUG lever: it produces realistic-looking products
    with controlled distributions so all pipeline stages can be tested
    end-to-end without downloading or processing the real 37M-item catalog.

    Distribution choices:
        - prices:    log-normal(mu=3.0, sigma=1.2) -> median ~$20, long tail
        - ratings:   beta(a=5, b=2) scaled to [1, 5] -> right-skewed (most >3)
        - ship_days: geometric(p=0.3) + 1 -> most 1-3 days, some longer
        - stock_qty: 90% in-stock (Poisson(20)), 10% OOS (0)
        - brands:    sampled uniformly from _BRANDS
        - categories: sampled uniformly from _CATEGORIES
        - 2-4 variants per product (color/size combinations)

    Synthetic products populate the same fields as real data, including:
        rating_count, store, parent_asin, features.

    Args:
        n_products: Number of products to generate.
        seed:       Random seed for reproducibility.

    Returns:
        Tuple of (products, variants) where variants is a flat list of all
        variants across all products.
    """
    rng = np.random.RandomState(seed)

    products: list[Product] = []
    variants: list[Variant] = []

    # Synthetic store names
    _STORES = [
        "TechWorld", "HomeGoods Plus", "FashionHub", "SportZone", "BeautyBar",
        "GadgetGalaxy", "FitLife Store", "KidsToys Central", "OutdoorExperts",
        "ChefSupply", "GamerDen", "EcoMart", "LuxeFinds", "DailyDeals",
    ]

    for i in range(n_products):
        product_id = f"syn_{i:06d}"

        # Sample category and product type
        cat = rng.choice(_CATEGORIES)
        nouns = _PRODUCT_NOUNS.get(cat, ["Product"])
        product_noun = rng.choice(nouns)

        # Build title: "{Adjective} {Brand} {ProductNoun} {Model}"
        brand = rng.choice(_BRANDS)
        adjective = rng.choice(_ADJECTIVES)
        model_num = rng.randint(100, 9999)
        title = f"{adjective} {brand} {product_noun} {model_num}"

        # Description
        desc = (
            f"The {brand} {product_noun} {model_num} is a high-quality "
            f"{product_noun.lower()} designed for everyday use. "
            f"Category: {cat.replace('/', ' > ')}."
        )

        # Price: log-normal distribution (median ~$20, range $1-$2000+)
        price = float(np.clip(rng.lognormal(mean=3.0, sigma=1.2), 0.99, 9999.99))
        price = round(price, 2)

        # Rating: beta distribution scaled to [1, 5]
        raw_rating = rng.beta(5, 2)
        rating = round(float(1.0 + 4.0 * raw_rating), 2)

        # Rating count: Poisson distribution (most products have moderate reviews)
        rating_count = max(0, int(rng.poisson(150)))

        # Ship days: geometric distribution (most items ship quickly)
        ship_days = int(rng.geometric(p=0.3))

        # Stock: 90% in stock, 10% OOS
        if rng.random() < 0.10:
            stock_qty = 0
        else:
            stock_qty = max(1, int(rng.poisson(20)))

        # Store name
        store = str(rng.choice(_STORES))

        # Parent ASIN (synthetic)
        parent_asin = f"B{rng.randint(100000000, 999999999)}"

        # Features: 2-5 bullet points
        feature_templates = [
            f"Made with premium {rng.choice(_MATERIALS)} construction",
            f"Available in {rng.choice(_COLORS)} color",
            f"Designed for {cat.split('/')[-1]} enthusiasts",
            f"Backed by {brand} quality guarantee",
            f"Compact and easy to use",
            f"Perfect for home or office",
            f"Eco-friendly and sustainable",
        ]
        n_features = rng.randint(2, 6)
        features = list(rng.choice(feature_templates, size=n_features, replace=False))

        # Attributes (matching what _parse_details extracts from real data)
        color = rng.choice(_COLORS)
        material = rng.choice(_MATERIALS)
        attrs: dict[str, Any] = {
            "color": color,
            "material": material,
        }

        # Category-specific attributes
        if cat.startswith("electronics"):
            connector = rng.choice(_CONNECTORS)
            attrs["connector_type"] = connector
            if "charger" in cat or "power" in product_noun.lower():
                attrs["wattage"] = int(rng.choice([5, 10, 18, 20, 30, 45, 60, 65, 100]))
            if "monitor" in cat or "screen" in product_noun.lower():
                attrs["screen_size_inches"] = float(
                    rng.choice([13.3, 14, 15.6, 17.3, 21.5, 24, 27, 32, 34, 43])
                )
        if cat.startswith("clothing") or cat.startswith("sports"):
            attrs["size"] = rng.choice(_SIZES)
        if cat.startswith("home/furniture") or cat.startswith("sports/fitness"):
            attrs["weight_lbs"] = round(float(rng.lognormal(2.0, 1.0)), 1)
        if cat.startswith("beauty"):
            attrs["item_form"] = str(
                rng.choice(["cream", "gel", "spray", "powder", "liquid", "serum"])
            )
            attrs["skin_type"] = str(
                rng.choice(["oily", "dry", "sensitive", "combination", "all"])
            )

        product = Product(
            id=product_id,
            title=title,
            desc=desc,
            cat=cat,
            brand=brand,
            attrs=attrs,
            price=price,
            rating=rating,
            ship_days=ship_days,
            stock_qty=stock_qty,
            rating_count=rating_count,
            store=store,
            parent_asin=parent_asin,
            features=features,
        )
        products.append(product)

        # Generate 2-4 variants per product
        n_variants = rng.randint(2, 5)
        variant_colors = rng.choice(_COLORS, size=min(n_variants, len(_COLORS)), replace=False)

        for vi in range(n_variants):
            variant_color = variant_colors[vi] if vi < len(variant_colors) else color
            variant_size = None
            variant_attrs: dict[str, Any] = {}

            # Some variants differ by size too (clothing/shoes)
            if cat.startswith("clothing") or cat.startswith("sports"):
                variant_size = rng.choice(_SIZES)

            # Price delta: +-10% for variants
            price_delta = round(float(rng.uniform(-0.1, 0.1) * price), 2)

            # Variant stock
            variant_stock = max(0, int(rng.poisson(10))) if stock_qty > 0 else 0

            variant = Variant(
                variant_id=f"{product_id}_v{vi}",
                product_id=product_id,
                color=variant_color,
                size=variant_size,
                attrs=variant_attrs,
                price_delta=price_delta,
                stock_qty=variant_stock,
            )
            variants.append(variant)

    logger.info(
        "Generated synthetic catalog: %d products, %d variants",
        len(products),
        len(variants),
    )
    return products, variants


# ---------------------------------------------------------------------------
# Persona loading
# ---------------------------------------------------------------------------

# Fields to extract from the real Nemotron-Personas-USA dataset
_PERSONA_FIELDS: list[str] = [
    "uuid",
    "persona",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "cultural_background",
    "skills_and_expertise",
    "skills_and_expertise_list",
    "hobbies_and_interests",
    "hobbies_and_interests_list",
    "career_goals_and_ambitions",
    "sex",
    "age",
    "marital_status",
    "education_level",
    "bachelors_field",
    "occupation",
    "city",
    "state",
    "zipcode",
    "country",
]


def load_personas(
    dataset_name: str = "nvidia/Nemotron-Personas-USA",
    max_items: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load persona data from the Nemotron Personas USA dataset.

    These personas provide realistic user profiles for the simulator's
    latent utility weights (spec Section 1.4).

    The real dataset has 23 columns including structured demographics
    (age, sex, occupation, education_level, marital_status, city, state)
    and free-text persona descriptions (persona, professional_persona,
    sports_persona, arts_persona, travel_persona, culinary_persona).

    Returns dicts with keys: uuid, persona (general text), age, sex,
    occupation, city, state, education_level, marital_status,
    hobbies_and_interests, skills_and_expertise, cultural_background,
    career_goals_and_ambitions, and all other raw fields.

    Args:
        dataset_name: HuggingFace dataset identifier.
        max_items:    If set, randomly sample this many personas.
        seed:         Random seed for reproducible sampling.

    Returns:
        List of persona dicts, each containing structured demographic
        and preference fields from the dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install with: uv add datasets"
        ) from exc

    logger.info(
        "Loading personas from HuggingFace: %s (max_items=%s)",
        dataset_name,
        max_items,
    )

    ds = load_dataset(dataset_name, split="train")

    if max_items is not None and max_items < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_items))

    available_columns = set(ds.column_names)

    personas: list[dict[str, Any]] = []
    for row in ds:
        persona: dict[str, Any] = {}

        # Extract all known fields with type-safe defaults
        for field in _PERSONA_FIELDS:
            if field in available_columns and row.get(field) is not None:
                persona[field] = row[field]

        # Ensure required fields have defaults
        if "uuid" not in persona:
            persona["uuid"] = str(uuid.uuid4())
        if "persona" not in persona:
            # Fallback: concatenate available persona fields
            parts = []
            for key in ("professional_persona", "sports_persona",
                        "arts_persona", "travel_persona", "culinary_persona"):
                if key in persona and persona[key]:
                    parts.append(str(persona[key]))
            persona["persona"] = " ".join(parts) if parts else "General consumer"

        # Normalize age to int
        if "age" in persona:
            try:
                persona["age"] = int(persona["age"])
            except (TypeError, ValueError):
                persona["age"] = 30  # default

        # Normalize sex to standard values
        if "sex" in persona:
            persona["sex"] = str(persona["sex"]).strip()

        # Parse list fields from Python-style string representations
        for list_field in ("hobbies_and_interests_list", "skills_and_expertise_list"):
            if list_field in persona and isinstance(persona[list_field], str):
                try:
                    import ast
                    parsed = ast.literal_eval(persona[list_field])
                    if isinstance(parsed, list):
                        persona[list_field] = [str(item).strip() for item in parsed]
                except (ValueError, SyntaxError):
                    # Keep as string if parsing fails
                    pass

        # Include any extra columns not in _PERSONA_FIELDS (forward-compatible)
        for col in available_columns:
            if col not in persona and row.get(col) is not None:
                persona[col] = row[col]

        personas.append(persona)

    logger.info("Loaded %d personas from %s", len(personas), dataset_name)
    return personas


# ---------------------------------------------------------------------------
# Utility: generate persona weights from persona dict (for simulator)
# ---------------------------------------------------------------------------

# Occupation keywords that suggest higher price sensitivity
_BUDGET_OCCUPATIONS: set[str] = {
    "student", "intern", "cashier", "clerk", "assistant", "receptionist",
    "server", "barista", "babysitter", "freelancer", "part-time",
    "retail", "janitor", "custodian", "housekeeper", "laborer",
}

# Occupation keywords that suggest lower price sensitivity (high income)
_HIGH_INCOME_OCCUPATIONS: set[str] = {
    "doctor", "physician", "surgeon", "lawyer", "attorney", "executive",
    "director", "vp", "vice president", "ceo", "cfo", "cto", "partner",
    "consultant", "engineer", "dentist", "pharmacist", "pilot", "professor",
    "manager", "architect", "analyst", "scientist", "researcher",
}


def generate_persona_weights(
    persona: dict[str, Any] | str,
    seed: int | None = None,
) -> dict[str, float]:
    """Generate latent utility weights from a real persona dict.

    Spec Section 1.4:
        w in R^K, w_k >= 0, sum(w_k) = 1.
        u(p) = sum_k w_k * phi_k(p)

    Uses **all six** structured demographic columns from the
    nvidia/Nemotron-Personas-USA dataset to shape a Dirichlet prior
    before sampling, producing more informed weights than a purely
    random draw.

    Demographic columns used (all from the HuggingFace dataset):
        1. age              – int64
        2. sex              – string (Male / Female)
        3. marital_status   – string (never_married / married_present /
                              divorced / separated / widowed)
        4. education_level  – string (less_than_9th / 9th_12th_no_diploma /
                              high_school / some_college / associates /
                              bachelors / graduate)
        5. bachelors_field  – string (stem / stem_related / business /
                              arts_humanities / education / empty)
        6. occupation       – string (free-form, keyword-matched)

    Additional text columns used for fine-grained adjustment:
        - hobbies_and_interests (keyword search)

    Heuristic summary (alpha index map):
        [0] price  [1] rating  [2] shipping  [3] brand_pref  [4] similarity

    Accepts either a persona dict (with structured fields) or a plain
    string (for backward compatibility).

    Args:
        persona: Persona dict with keys like age, sex, occupation, etc.
                 Or a plain text string for backward compatibility.
        seed:    Optional override seed (otherwise derived from persona).

    Returns:
        Dict mapping weight names to float values, summing to 1.0.
    """
    weight_names = ["price", "rating", "shipping", "brand_preference", "similarity"]

    # Handle backward compatibility: accept plain strings
    if isinstance(persona, str):
        persona_text = persona
        persona_dict: dict[str, Any] = {}
    else:
        persona_dict = persona
        persona_text = str(persona_dict.get("persona", ""))

    # Derive seed from persona content if not provided
    if seed is None:
        seed_material = persona_text or str(persona_dict.get("uuid", ""))
        text_hash = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
        seed = int(text_hash[:8], 16)

    rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Start with base Dirichlet alpha, then adjust based on persona fields
    # ------------------------------------------------------------------
    # Alpha index:  [0=price, 1=rating, 2=shipping, 3=brand_pref, 4=similarity]
    # Base alpha: moderately informative prior
    alpha = np.array([2.0, 3.0, 1.5, 1.0, 1.5])

    # ==================================================================
    # 1) Age-based adjustments
    # ==================================================================
    age = persona_dict.get("age")
    if age is not None:
        try:
            age = int(age)
            if age < 18:
                # Minor / dependent: very price-sensitive, trend-following
                alpha[0] += 1.5   # price (allowance/limited budget)
                alpha[4] += 1.0   # similarity (peer trends)
            elif 18 <= age <= 24:
                # Young adult: trend-following, price-sensitive
                alpha[0] += 1.0   # price
                alpha[4] += 1.5   # similarity (trend-following)
            elif 25 <= age <= 34:
                # Early career: moderate price awareness, brand-curious
                alpha[0] += 0.5   # price
                alpha[4] += 0.8   # similarity
                alpha[3] += 0.3   # brand (starting preferences)
            elif 35 <= age <= 49:
                # Mid-career: balanced, slight brand preference
                alpha[3] += 0.5   # brand preference
                alpha[1] += 0.3   # rating (quality-aware)
            elif 50 <= age <= 64:
                # Pre-retirement: quality-focused, convenience matters
                alpha[1] += 1.5   # rating (quality-focused)
                alpha[2] += 0.5   # shipping (convenience)
                alpha[3] += 0.3   # brand loyalty
            else:  # 65+
                # Senior: quality + convenience dominant
                alpha[1] += 2.0   # rating
                alpha[2] += 1.0   # shipping (strong convenience need)
        except (TypeError, ValueError):
            pass

    # ==================================================================
    # 2) Sex-based adjustments (mild, for training signal diversity)
    # ==================================================================
    sex = str(persona_dict.get("sex", "")).strip().lower()
    if sex == "female":
        alpha[1] += 0.3   # slightly more review-conscious
        alpha[3] += 0.3   # slightly more brand-aware
    elif sex == "male":
        alpha[4] += 0.3   # slightly more spec-comparison
        alpha[0] += 0.2   # slightly more price-comparing

    # ==================================================================
    # 3) Marital status adjustments
    # ==================================================================
    marital = str(persona_dict.get("marital_status", "")).strip().lower()
    if "married" in marital:
        # Dual-income household: convenience matters, brand loyalty
        alpha[2] += 0.8   # shipping (busy household, convenience)
        alpha[3] += 0.5   # brand (household consistency)
        alpha[0] -= 0.3   # slightly less price-sensitive
    elif marital in ("divorced", "separated", "widowed"):
        # Single-income transition: budget-conscious, practical
        alpha[0] += 0.8   # price (single-income budget pressure)
        alpha[1] += 0.3   # rating (practical, reads reviews)
    elif marital == "never_married":
        # Typically younger / independent: trend-aware, budget-flexible
        alpha[4] += 0.5   # similarity (trend-following)
        alpha[0] += 0.3   # price (single budget)

    # ==================================================================
    # 4) Education-level adjustments (full coverage of dataset values)
    # ==================================================================
    education = str(persona_dict.get("education_level", "")).strip().lower()
    if education == "graduate":
        # Graduate degree: research-oriented, brand-aware, less price-driven
        alpha[1] += 0.8   # rating (data-driven, reads reviews)
        alpha[3] += 0.3   # brand (quality preference)
        alpha[0] -= 0.3   # less price-sensitive
    elif education == "bachelors":
        # Bachelor's degree: moderate review/brand awareness
        alpha[1] += 0.4   # rating
        alpha[3] += 0.2   # brand
    elif education in ("some_college", "associates"):
        # Some post-secondary: balanced, slight price awareness
        alpha[0] += 0.3   # price
        alpha[1] += 0.2   # rating
    elif education == "high_school":
        # High school: more price-conscious
        alpha[0] += 0.5   # price
    elif education in ("less_than_9th", "9th_12th_no_diploma"):
        # No diploma: most price-sensitive, practical
        alpha[0] += 0.8   # price (most budget-constrained)
        alpha[2] += 0.3   # shipping (value free/cheap shipping)

    # ==================================================================
    # 5) Bachelor's field adjustments (only ~26% have a value)
    # ==================================================================
    field = str(persona_dict.get("bachelors_field", "")).strip().lower()
    if field in ("stem", "stem_related"):
        # STEM: spec-comparison, data-driven review reading
        alpha[4] += 1.0   # similarity (technical spec-matching)
        alpha[1] += 0.5   # rating (data-driven)
    elif field == "business":
        # Business: brand-conscious, value-oriented
        alpha[3] += 1.0   # brand (business/brand awareness)
        alpha[0] -= 0.3   # less price-sensitive
    elif field == "arts_humanities":
        # Arts: aesthetic/style matching, design-aware brands
        alpha[4] += 0.5   # similarity (style/aesthetic matching)
        alpha[3] += 0.5   # brand (design-aware)
    elif field == "education":
        # Education: practical, review-oriented, budget-aware
        alpha[1] += 0.5   # rating (research-oriented)
        alpha[0] += 0.3   # price (typically public-sector salary)

    # ==================================================================
    # 6) Occupation-based adjustments (keyword matching)
    # ==================================================================
    occupation = str(persona_dict.get("occupation", "")).lower()
    if occupation and occupation not in ("not_in_workforce", ""):
        # Check for budget occupations
        if any(kw in occupation for kw in _BUDGET_OCCUPATIONS):
            alpha[0] += 2.0   # price (very sensitive)
            alpha[3] -= 0.3   # less brand-conscious
        # Check for high-income occupations
        elif any(kw in occupation for kw in _HIGH_INCOME_OCCUPATIONS):
            alpha[0] -= 0.5   # less price-sensitive
            alpha[3] += 1.5   # brand-conscious
            alpha[1] += 0.5   # quality-conscious

    # ==================================================================
    # 7) Hobbies / interests adjustments (keyword matching)
    # ==================================================================
    hobbies = str(persona_dict.get("hobbies_and_interests", "")).lower()
    if hobbies:
        if any(kw in hobbies for kw in ("fashion", "style", "design", "luxury")):
            alpha[3] += 1.0   # brand preference
            alpha[4] += 0.5   # similarity (style-matching)
        if any(kw in hobbies for kw in ("deal", "coupon", "thrift", "budget", "frugal")):
            alpha[0] += 1.5   # price
        if any(kw in hobbies for kw in ("tech", "gadget", "gaming", "computer")):
            alpha[4] += 1.0   # similarity (spec-matching)
            alpha[1] += 0.5   # rating (reviews-focused)

    # Ensure all alpha values are at least 0.5 (valid Dirichlet params)
    alpha = np.maximum(alpha, 0.5)

    # Sample from Dirichlet with adjusted alpha
    raw = rng.dirichlet(alpha)
    weights = {name: round(float(w), 4) for name, w in zip(weight_names, raw)}

    # Ensure sum is exactly 1.0 (fix floating-point drift)
    total = sum(weights.values())
    if total > 0:
        adjustment = 1.0 - total
        weights[weight_names[0]] = round(weights[weight_names[0]] + adjustment, 4)

    return weights
