"""Persona sampling and latent utility computation (Spec Section 1.4).

Persona weights w in R^K with w_k >= 0 and sum(w_k) = 1.
Utility: u(p) = sum_k w_k * phi_k(p) in [0, 1]

Feature functions:
    phi_price(p)  = 1 - clip((price - P_low) / (P_high - P_low), 0, 1)
    phi_rating(p) = clip((rating - 1) / 4, 0, 1)
    phi_ship(p)   = 1 - clip(ship_days / S_max, 0, 1),  S_max=14
    phi_brand(p)  = 1[brand == brand_pref]
    phi_sim(p;p0) = (e_p^T e_{p0} + 1) / 2

The persona weights are sampled from a Dirichlet distribution to ensure
they form a valid probability simplex. The compute_utility function
aggregates all feature functions into a scalar utility score in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Persona weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PersonaWeights:
    """Persona preference weights forming a probability simplex.

    Spec Section 1.4:
        w in R^K with w_k >= 0 and sum(w_k) = 1.

    Attributes:
        w_price:      Weight for price sensitivity.
        w_rating:     Weight for rating preference.
        w_ship:       Weight for fast shipping preference.
        w_brand:      Weight for brand loyalty.
        w_similarity: Weight for similarity to a reference product.
    """

    w_price: float
    w_rating: float
    w_ship: float
    w_brand: float
    w_similarity: float

    def __post_init__(self) -> None:
        """Validate that all weights are non-negative and sum to 1."""
        weights = [self.w_price, self.w_rating, self.w_ship, self.w_brand, self.w_similarity]
        for w in weights:
            if w < -1e-9:
                raise ValueError(f"Persona weights must be non-negative, got {w}")
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Persona weights must sum to 1.0, got {total}")

    def as_array(self) -> np.ndarray:
        """Return weights as a numpy array of shape (5,)."""
        return np.array(
            [self.w_price, self.w_rating, self.w_ship, self.w_brand, self.w_similarity],
            dtype=np.float64,
        )

    def as_dict(self) -> dict[str, float]:
        """Return weights as a dictionary."""
        return {
            "w_price": self.w_price,
            "w_rating": self.w_rating,
            "w_ship": self.w_ship,
            "w_brand": self.w_brand,
            "w_similarity": self.w_similarity,
        }

    # -- DEBUG levers: deterministic persona presets --------------------------

    @classmethod
    def uniform(cls) -> PersonaWeights:
        """DEBUG lever: equal weights (0.2 each) for deterministic debugging."""
        return cls(w_price=0.2, w_rating=0.2, w_ship=0.2, w_brand=0.2, w_similarity=0.2)

    @classmethod
    def price_focused(cls) -> PersonaWeights:
        """DEBUG lever: price-obsessed persona."""
        return cls(w_price=0.6, w_rating=0.15, w_ship=0.15, w_brand=0.05, w_similarity=0.05)

    @classmethod
    def quality_focused(cls) -> PersonaWeights:
        """DEBUG lever: quality-obsessed persona (rating + brand)."""
        return cls(w_price=0.05, w_rating=0.45, w_ship=0.05, w_brand=0.35, w_similarity=0.10)

    @classmethod
    def from_dict(cls, d: dict) -> PersonaWeights:
        """Create from dict, normalizing to sum=1.

        Accepts both internal key format (``w_price``, ``w_rating``, …) and
        the short key format returned by
        :func:`~shop_rlve.data.catalog_loader.generate_persona_weights`
        (``price``, ``rating``, ``shipping``, ``brand_preference``,
        ``similarity``).
        """
        # Support both "w_price" and "price" key formats
        _get = lambda long, short, default=0.2: d.get(long, d.get(short, default))  # noqa: E731
        raw = [
            _get("w_price", "price"),
            _get("w_rating", "rating"),
            _get("w_ship", "shipping"),
            _get("w_brand", "brand_preference"),
            _get("w_similarity", "similarity"),
        ]
        total = sum(raw)
        if total <= 0:
            raw = [0.2, 0.2, 0.2, 0.2, 0.2]
            total = 1.0
        norm = [w / total for w in raw]
        return cls(w_price=norm[0], w_rating=norm[1], w_ship=norm[2], w_brand=norm[3], w_similarity=norm[4])


# ---------------------------------------------------------------------------
# Persona sampling
# ---------------------------------------------------------------------------


def sample_persona_weights(seed: int, alpha: np.ndarray | None = None) -> PersonaWeights:
    """Sample persona weights from a Dirichlet distribution.

    Spec Section 1.4:
        w ~ Dirichlet(alpha), ensuring w_k >= 0 and sum(w_k) = 1.

    The default alpha = [2, 2, 1, 1, 1] gives moderate concentration toward
    price and rating while allowing diversity in other dimensions.

    Args:
        seed:  Random seed for reproducibility.
        alpha: Dirichlet concentration parameters of shape (5,). Defaults
               to [2.0, 2.0, 1.0, 1.0, 1.0].

    Returns:
        PersonaWeights sampled from Dirichlet(alpha).
    """
    rng = np.random.default_rng(seed)
    if alpha is None:
        alpha = np.array([2.0, 2.0, 1.0, 1.0, 1.0])

    weights = rng.dirichlet(alpha)
    return PersonaWeights(
        w_price=float(weights[0]),
        w_rating=float(weights[1]),
        w_ship=float(weights[2]),
        w_brand=float(weights[3]),
        w_similarity=float(weights[4]),
    )


# ---------------------------------------------------------------------------
# Feature functions (phi_k)
# ---------------------------------------------------------------------------


def phi_price(price: float, p_low: float, p_high: float) -> float:
    """Price feature: lower price is better.

    Spec Section 1.4:
        phi_price(p) = 1 - clip((price - P_low) / (P_high - P_low), 0, 1)

    Args:
        price:  Product price in USD.
        p_low:  Lower bound of the desired price range.
        p_high: Upper bound of the desired price range.

    Returns:
        Feature value in [0, 1]. Returns 1.0 when price <= p_low,
        0.0 when price >= p_high.
    """
    if p_high <= p_low:
        # Degenerate range: exact match gives 1, anything else 0
        return 1.0 if price <= p_low else 0.0
    ratio = (price - p_low) / (p_high - p_low)
    return 1.0 - float(np.clip(ratio, 0.0, 1.0))


def phi_rating(rating: float) -> float:
    """Rating feature: higher rating is better.

    Spec Section 1.4:
        phi_rating(p) = clip((rating - 1) / 4, 0, 1)

    Maps rating from [1, 5] to [0, 1].

    Args:
        rating: Product rating in [1, 5].

    Returns:
        Feature value in [0, 1].
    """
    return float(np.clip((rating - 1.0) / 4.0, 0.0, 1.0))


def phi_ship(ship_days: int | float, s_max: int = 14) -> float:
    """Shipping speed feature: fewer days is better.

    Spec Section 1.4:
        phi_ship(p) = 1 - clip(ship_days / S_max, 0, 1),  S_max=14

    Args:
        ship_days: Estimated shipping days.
        s_max:     Maximum shipping days for normalization (default: 14).

    Returns:
        Feature value in [0, 1]. Returns 1.0 for 0-day shipping,
        0.0 for ship_days >= s_max.
    """
    if s_max <= 0:
        return 1.0 if ship_days <= 0 else 0.0
    return 1.0 - float(np.clip(ship_days / s_max, 0.0, 1.0))


def phi_brand(brand: str, brand_pref: str) -> float:
    """Brand match feature: exact match indicator.

    Spec Section 1.4:
        phi_brand(p) = 1[brand == brand_pref]

    Case-insensitive comparison.

    Args:
        brand:      Product brand name.
        brand_pref: Preferred brand name.

    Returns:
        1.0 if brands match (case-insensitive), 0.0 otherwise.
    """
    return 1.0 if brand.lower().strip() == brand_pref.lower().strip() else 0.0


def phi_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Similarity feature: cosine similarity scaled to [0, 1].

    Spec Section 1.4:
        phi_sim(p;p0) = (e_p^T e_{p0} + 1) / 2

    Assumes embeddings are L2-normalized. The raw cosine similarity
    in [-1, 1] is mapped to [0, 1].

    Args:
        embedding_a: L2-normalized embedding vector of shape (dim,).
        embedding_b: L2-normalized reference embedding vector of shape (dim,).

    Returns:
        Similarity feature value in [0, 1].
    """
    cos_sim = float(np.dot(embedding_a, embedding_b))
    return (cos_sim + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Utility computation
# ---------------------------------------------------------------------------


def compute_utility(
    product: Any,
    weights: PersonaWeights,
    *,
    p_low: float,
    p_high: float,
    brand_pref: str | None = None,
    ref_embedding: np.ndarray | None = None,
    s_max: int = 14,
) -> float:
    """Compute the latent utility u(p) for a product given persona weights.

    Spec Section 1.4:
        u(p) = sum_k w_k * phi_k(p) in [0, 1]

    Each feature phi_k evaluates one aspect of product quality relative
    to the persona's preferences. The utility is a weighted sum of all
    features, guaranteed to be in [0, 1] since all phi_k are in [0, 1]
    and all weights are non-negative summing to 1.

    Args:
        product:       Product object (must have .price, .rating, .ship_days,
                       .brand attributes, or be a dict with those keys).
        weights:       PersonaWeights instance.
        p_low:         Lower bound of the desired price range.
        p_high:        Upper bound of the desired price range.
        brand_pref:    Preferred brand name. If None, brand feature contributes 0.
        ref_embedding: Reference product embedding for similarity. If None,
                       similarity feature contributes 0.
        s_max:         Maximum shipping days for normalization (default: 14).

    Returns:
        Utility score in [0, 1].
    """
    # Extract product attributes (support both Pydantic models and dicts)
    if hasattr(product, "price"):
        price = product.price
        rating = product.rating
        ship_days_val = product.ship_days
        brand = product.brand
    else:
        price = product["price"]
        rating = product["rating"]
        ship_days_val = product["ship_days"]
        brand = product.get("brand", "unknown")

    # Compute individual feature values
    f_price = phi_price(price, p_low, p_high)
    f_rating = phi_rating(rating)
    f_ship = phi_ship(ship_days_val, s_max)

    f_brand = 0.0
    if brand_pref is not None:
        f_brand = phi_brand(brand, brand_pref)

    f_sim = 0.0
    if ref_embedding is not None:
        # Get product embedding if available
        product_embedding: np.ndarray | None = None
        if hasattr(product, "_embedding"):
            product_embedding = product._embedding
        elif isinstance(product, dict) and "_embedding" in product:
            product_embedding = product["_embedding"]

        if product_embedding is not None:
            f_sim = phi_similarity(product_embedding, ref_embedding)

    # Weighted sum
    utility = (
        weights.w_price * f_price
        + weights.w_rating * f_rating
        + weights.w_ship * f_ship
        + weights.w_brand * f_brand
        + weights.w_similarity * f_sim
    )

    # Clamp to [0, 1] for safety (should already be in range)
    return float(np.clip(utility, 0.0, 1.0))
