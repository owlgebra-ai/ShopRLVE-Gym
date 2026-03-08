"""Catalog tool implementations for ShopRLVE-GYM.

Spec Section 2.2 (Tool list A: Catalog + retrieval):
    1. catalog.search   -- vector retrieval + metadata filtering + eps_rank degradation
    2. catalog.rerank   -- re-score candidates by embedding similarity
    3. catalog.get_product  -- full product details
    4. catalog.get_variants -- variant list for a product

Spec Section 3.2 (Difficulty-aware retrieval degradation):
    Given ranked list L = [p_1,...,p_K], create degraded list L':
    - For each rank i, with prob eps_rank(d), replace p_i with a random
      distractor from the same category.
    - p_i' = p_i               w.p. 1 - eps_rank(d)
    - p_i' ~ Uniform(Distractors(cat(p_i)))  w.p. eps_rank(d)
    Then deduplicate and truncate to K.

These handlers are designed to be registered with ToolRegistry and called
via ToolRegistry.execute(). Each handler accepts validated keyword args
plus a 'state' keyword argument containing the episode state.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from shop_rlve.data.schema import (
    FILTER_KEYS,
    Product,
    ProductCard,
    Variant,
    product_to_card,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic arg models for each catalog tool
# ---------------------------------------------------------------------------


class CatalogSearchArgs(BaseModel):
    """Arguments for catalog.search tool.

    Spec Section 2.2 tool 1:
        Args: query (str), filters (dict, optional), top_k (int)
        Returns: list of product cards
    """

    query: str = Field(..., min_length=1, description="Search query string")
    filters: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional metadata filters. Keys from FILTER_KEYS: "
            "cat, brand, color, size, material, connector_type, "
            "price_min, price_max, rating_min, ship_days_max, etc."
        ),
    )
    top_k: int = Field(default=20, ge=1, le=500, description="Number of results to return")


class CatalogRerankArgs(BaseModel):
    """Arguments for catalog.rerank tool.

    Spec Section 2.2 tool 2:
        Args: query (str), candidate_product_ids (list[str]), top_k (int)
        Returns: list of product IDs ordered by reranker score
    """

    query: str = Field(..., min_length=1, description="Reranking query string")
    candidate_product_ids: list[str] = Field(
        ..., min_length=1, description="Product IDs to rerank"
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")


class CatalogGetProductArgs(BaseModel):
    """Arguments for catalog.get_product tool.

    Spec Section 2.2 tool 3:
        Args: product_id (str)
        Returns: full product schema
    """

    product_id: str = Field(..., min_length=1, description="Product identifier to retrieve")


class CatalogGetVariantsArgs(BaseModel):
    """Arguments for catalog.get_variants tool.

    Spec Section 2.2 tool 4:
        Args: product_id (str)
        Returns: list of variants
    """

    product_id: str = Field(..., min_length=1, description="Product identifier")


# ---------------------------------------------------------------------------
# Internal: metadata filter application
# ---------------------------------------------------------------------------


def _apply_filters(
    product: Product,
    filters: dict[str, Any],
) -> bool:
    """Check whether a product passes all metadata filters.

    Supports both discrete equality filters and numeric range filters.
    Unknown filter keys are ignored (but logged).

    Args:
        product: Product to check.
        filters: Dict of filter_key -> value.

    Returns:
        True if the product passes all filters.
    """
    for key, value in filters.items():
        if key not in FILTER_KEYS:
            logger.debug("Ignoring unknown filter key: '%s'", key)
            continue

        # Discrete equality filters
        if key == "cat":
            if product.cat != value:
                return False
        elif key == "brand":
            if product.brand.lower() != str(value).lower():
                return False
        elif key == "store":
            if product.store.lower() != str(value).lower():
                return False
        elif key in ("color", "size", "material", "connector_type",
                     "item_form", "skin_type", "finish_type"):
            product_val = product.attrs.get(key)
            if product_val is None or str(product_val).lower() != str(value).lower():
                return False

        # Numeric range filters
        elif key == "rating_count_min":
            if product.rating_count < int(value):
                return False
        elif key == "price_min":
            if product.price < float(value):
                return False
        elif key == "price_max":
            if product.price > float(value):
                return False
        elif key == "rating_min":
            if product.rating < float(value):
                return False
        elif key == "rating_max":
            if product.rating > float(value):
                return False
        elif key == "ship_days_max":
            if product.ship_days > int(value):
                return False
        elif key == "wattage_min":
            wattage = product.attrs.get("wattage")
            if wattage is None or float(wattage) < float(value):
                return False
        elif key == "wattage_max":
            wattage = product.attrs.get("wattage")
            if wattage is None or float(wattage) > float(value):
                return False
        elif key == "weight_lbs_max":
            weight = product.attrs.get("weight_lbs")
            if weight is None or float(weight) > float(value):
                return False
        elif key == "screen_size_inches_min":
            screen = product.attrs.get("screen_size_inches")
            if screen is None or float(screen) < float(value):
                return False
        elif key == "screen_size_inches_max":
            screen = product.attrs.get("screen_size_inches")
            if screen is None or float(screen) > float(value):
                return False

    return True


def _apply_retrieval_degradation(
    ranked_ids: list[str],
    eps_rank: float,
    products_by_id: dict[str, Product],
    category_index: dict[str, list[str]],
    rng: random.Random,
) -> list[str]:
    """Apply difficulty-aware retrieval degradation.

    Spec Section 3.2:
        Given ranked list L = [p_1,...,p_K], create degraded list L':
        For each rank i:
            p_i' = p_i                                 w.p. 1 - eps_rank(d)
            p_i' ~ Uniform(Distractors(cat(p_i)))      w.p. eps_rank(d)
        Then deduplicate and truncate to K.

    Args:
        ranked_ids:      Original ranked product ID list.
        eps_rank:        Degradation probability (0 = no noise, 0.4 = max).
        products_by_id:  Dict mapping product_id -> Product.
        category_index:  Dict mapping category -> list of product_ids in that category.
        rng:             Random instance for reproducibility.

    Returns:
        Degraded list of product IDs (deduplicated, same length or shorter).
    """
    if eps_rank <= 0.0:
        return ranked_ids

    degraded: list[str] = []
    seen: set[str] = set()

    for pid in ranked_ids:
        if rng.random() < eps_rank:
            # Replace with a random distractor from the same category
            product = products_by_id.get(pid)
            if product is not None:
                cat = product.cat
                distractors = category_index.get(cat, [])
                if distractors:
                    replacement = rng.choice(distractors)
                    if replacement not in seen:
                        degraded.append(replacement)
                        seen.add(replacement)
                    continue
            # Fallback: keep original if no distractors available
            if pid not in seen:
                degraded.append(pid)
                seen.add(pid)
        else:
            if pid not in seen:
                degraded.append(pid)
                seen.add(pid)

    return degraded


# ---------------------------------------------------------------------------
# CatalogState: shared state that catalog tools need access to
# ---------------------------------------------------------------------------


class CatalogState:
    """Holds the catalog data structures that tool handlers need.

    This is initialized once at environment setup and passed as part
    of the episode state. Tool handlers access it via state.catalog_state.

    Attributes:
        products_by_id:  Dict mapping product_id -> Product.
        variants_by_product: Dict mapping product_id -> list of Variant.
        vector_index:    VectorIndex or MockVectorIndex for retrieval.
        embedding_engine: EmbeddingEngine for encoding queries.
        category_index:  Dict mapping category -> list of product_ids.
        eps_rank:        Current retrieval degradation probability (from difficulty).
        rng:             Random instance for reproducibility.
    """

    def __init__(
        self,
        products: list[Product],
        variants: list[Variant] | None = None,
        vector_index: Any = None,
        embedding_engine: Any = None,
        eps_rank: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.products_by_id: dict[str, Product] = {p.id: p for p in products}
        self.variants_by_product: dict[str, list[Variant]] = {}
        if variants:
            for v in variants:
                self.variants_by_product.setdefault(v.product_id, []).append(v)

        self.vector_index = vector_index
        self.embedding_engine = embedding_engine
        self.eps_rank = eps_rank
        self.rng = random.Random(seed)

        # Build category index for retrieval degradation
        self.category_index: dict[str, list[str]] = {}
        for p in products:
            self.category_index.setdefault(p.cat, []).append(p.id)


# ---------------------------------------------------------------------------
# Tool handler: catalog.search
# ---------------------------------------------------------------------------


def catalog_search(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 20,
    *,
    state: Any = None,
) -> list[dict[str, Any]]:
    """Search the product catalog by query with optional metadata filters.

    Spec Section 2.2 tool 1 + Section 3.1 + Section 3.2:
        1. Encode query: e_q = normalize(f_enc(q))
        2. Vector search: retrieve top_k * 3 candidates by score_vec(q,p) = e_q^T e_p
        3. Apply metadata filters from FILTER_KEYS
        4. Apply difficulty-aware retrieval degradation:
           with prob eps_rank(d), replace result with random distractor
        5. Track returned product_ids in state.seen_product_ids

    Args:
        query:   Search query string.
        filters: Optional metadata filters (keys from FILTER_KEYS).
        top_k:   Number of results to return.
        state:   Episode state with .catalog_state and .seen_product_ids.

    Returns:
        List of ProductCard dicts, sorted by relevance.
    """
    catalog_state: CatalogState = _get_catalog_state(state)

    if catalog_state.vector_index is None or catalog_state.embedding_engine is None:
        logger.warning("catalog.search: no vector index or embedding engine available")
        return []

    # Step 1: Encode query
    query_embedding = catalog_state.embedding_engine.encode_query(query)

    # Step 2: Vector search (retrieve extra to allow for filtering)
    retrieval_k = min(top_k * 3, len(catalog_state.products_by_id))
    raw_results = catalog_state.vector_index.search(query_embedding, top_k=retrieval_k)

    # Step 3: Apply metadata filters
    if filters:
        filtered_results: list[tuple[str, float]] = []
        for pid, score in raw_results:
            product = catalog_state.products_by_id.get(pid)
            if product is not None and _apply_filters(product, filters):
                filtered_results.append((pid, score))
        raw_results = filtered_results

    # Extract ranked IDs
    ranked_ids = [pid for pid, _score in raw_results]

    # Step 4: Apply retrieval degradation
    if catalog_state.eps_rank > 0.0:
        ranked_ids = _apply_retrieval_degradation(
            ranked_ids=ranked_ids,
            eps_rank=catalog_state.eps_rank,
            products_by_id=catalog_state.products_by_id,
            category_index=catalog_state.category_index,
            rng=catalog_state.rng,
        )

    # Truncate to top_k
    ranked_ids = ranked_ids[:top_k]

    # Convert to ProductCards
    cards: list[dict[str, Any]] = []
    for pid in ranked_ids:
        product = catalog_state.products_by_id.get(pid)
        if product is not None:
            card = product_to_card(product)
            cards.append(card.model_dump())

    # Step 5: Track seen product IDs
    seen_ids = _get_seen_ids(state)
    if seen_ids is not None:
        for pid in ranked_ids:
            seen_ids.add(pid)

    return cards


# ---------------------------------------------------------------------------
# Tool handler: catalog.rerank
# ---------------------------------------------------------------------------


def catalog_rerank(
    query: str,
    candidate_product_ids: list[str],
    top_k: int = 10,
    *,
    state: Any = None,
) -> list[str]:
    """Re-score candidate products by embedding similarity to query.

    Spec Section 3.1 step 5:
        - Get top K0 candidates by score_vec (done in search).
        - Compute cross-encoder score_rr(q,p) (optional, not yet implemented).
        - For MVP: re-score by cosine similarity of query vs product embeddings.
        - Return top_k by re-ranked score.

    Args:
        query:                Search/reranking query string.
        candidate_product_ids: Product IDs to rerank.
        top_k:                Number of results to return.
        state:                Episode state.

    Returns:
        List of product IDs ordered by descending reranker score.
    """
    catalog_state: CatalogState = _get_catalog_state(state)

    if catalog_state.embedding_engine is None:
        logger.warning("catalog.rerank: no embedding engine available")
        return candidate_product_ids[:top_k]

    # Encode query
    query_embedding = catalog_state.embedding_engine.encode_query(query)

    # Score each candidate by cosine similarity
    scored: list[tuple[str, float]] = []
    for pid in candidate_product_ids:
        product = catalog_state.products_by_id.get(pid)
        if product is None:
            continue

        # Try to get embedding from index first (avoids re-encoding)
        product_embedding = None
        if catalog_state.vector_index is not None:
            product_embedding = catalog_state.vector_index.get_embedding(pid)

        if product_embedding is None:
            # Fall back to encoding the product
            product_embedding = catalog_state.embedding_engine.encode_product(product)

        sim = catalog_state.embedding_engine.cosine_similarity(
            query_embedding, product_embedding
        )
        scored.append((pid, sim))

    # Sort by descending score
    scored.sort(key=lambda x: x[1], reverse=True)

    return [pid for pid, _score in scored[:top_k]]


# ---------------------------------------------------------------------------
# Tool handler: catalog.get_product
# ---------------------------------------------------------------------------


def catalog_get_product(
    product_id: str,
    *,
    state: Any = None,
) -> dict[str, Any] | None:
    """Retrieve full product details by ID.

    Spec Section 2.2 tool 3:
        Args: product_id
        Returns: full product schema (including attrs)

    Also tracks the product_id in state.seen_product_ids for
    hallucination detection (spec Section 5, r_hall).

    Args:
        product_id: Product identifier.
        state:      Episode state.

    Returns:
        Full product dict, or None if not found.
    """
    catalog_state: CatalogState = _get_catalog_state(state)

    product = catalog_state.products_by_id.get(product_id)
    if product is None:
        return None

    # Track seen product IDs
    seen_ids = _get_seen_ids(state)
    if seen_ids is not None:
        seen_ids.add(product_id)

    return product.model_dump()


# ---------------------------------------------------------------------------
# Tool handler: catalog.get_variants
# ---------------------------------------------------------------------------


def catalog_get_variants(
    product_id: str,
    *,
    state: Any = None,
) -> list[dict[str, Any]]:
    """Retrieve all variants for a product.

    Spec Section 2.2 tool 4:
        Args: product_id
        Returns: variants list (variant_id, color, size, etc.)

    Args:
        product_id: Product identifier.
        state:      Episode state.

    Returns:
        List of variant dicts. Empty list if product not found or has no variants.
    """
    catalog_state: CatalogState = _get_catalog_state(state)

    variants = catalog_state.variants_by_product.get(product_id, [])
    return [v.model_dump() for v in variants]


# ---------------------------------------------------------------------------
# State accessor helpers
# ---------------------------------------------------------------------------


def _get_catalog_state(state: Any) -> CatalogState:
    """Extract CatalogState from the episode state object.

    Supports multiple state shapes:
        - state.catalog_state (attribute)
        - state["catalog_state"] (dict)
        - state itself is a CatalogState

    Raises:
        ValueError: If no CatalogState can be found.
    """
    if isinstance(state, CatalogState):
        return state

    if hasattr(state, "catalog_state"):
        cs = state.catalog_state
        if isinstance(cs, CatalogState):
            return cs

    if isinstance(state, dict) and "catalog_state" in state:
        cs = state["catalog_state"]
        if isinstance(cs, CatalogState):
            return cs

    raise ValueError(
        "Could not find CatalogState in the provided state object. "
        "Ensure state has a .catalog_state attribute or is a CatalogState itself."
    )


def _get_seen_ids(state: Any) -> set[str] | None:
    """Extract the seen_product_ids set from the episode state.

    Spec Section 8.1:
        Seen = set of product IDs surfaced to the model

    Returns None if the state doesn't have a seen_product_ids attribute
    (e.g., in testing).
    """
    if hasattr(state, "seen_product_ids"):
        return state.seen_product_ids

    if isinstance(state, dict) and "seen_product_ids" in state:
        return state["seen_product_ids"]

    # CatalogState used directly (no episode wrapper) -- no tracking
    return None


# ---------------------------------------------------------------------------
# Registration helper: register all catalog tools with a ToolRegistry
# ---------------------------------------------------------------------------


def register_catalog_tools(registry: Any) -> None:
    """Register all catalog tool handlers with a ToolRegistry instance.

    Call this during environment initialization to make catalog tools
    available for LLM tool calls.

    Args:
        registry: ToolRegistry instance.
    """
    registry.register("catalog.search", catalog_search, CatalogSearchArgs)
    registry.register("catalog.rerank", catalog_rerank, CatalogRerankArgs)
    registry.register("catalog.get_product", catalog_get_product, CatalogGetProductArgs)
    registry.register("catalog.get_variants", catalog_get_variants, CatalogGetVariantsArgs)

    logger.info("Registered 4 catalog tools: search, rerank, get_product, get_variants")
