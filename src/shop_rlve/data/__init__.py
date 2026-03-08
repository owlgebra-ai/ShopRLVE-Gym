"""ShopRLVE-GYM data layer: schemas, embeddings, indexing, and catalog loading.

This package provides the foundation data infrastructure for the RL environment:

- **schema**: Canonical Product/Variant/ProductCard models, attribute allowlist,
  filter keys, denied categories, and helper functions.
- **embeddings**: EmbeddingEngine wrapping SentenceTransformer with debug mode.
- **index**: FAISS VectorIndex and brute-force MockVectorIndex for retrieval.
- **catalog_loader**: Load real (Amazebay) or synthetic catalogs and personas.
"""

from shop_rlve.data.catalog_loader import (
    generate_persona_weights,
    generate_synthetic_catalog,
    load_catalog,
    load_personas,
)
from shop_rlve.data.embeddings import EmbeddingEngine
from shop_rlve.data.index import MockVectorIndex, VectorIndex
from shop_rlve.data.schema import (
    ATTRIBUTE_ALLOWLIST,
    DENIED_CATEGORIES,
    FILTER_KEYS,
    REAL_DATASET_COLUMNS,
    Product,
    ProductCard,
    Variant,
    avail,
    product_to_card,
)

__all__ = [
    # Schema
    "Product",
    "ProductCard",
    "Variant",
    "ATTRIBUTE_ALLOWLIST",
    "FILTER_KEYS",
    "REAL_DATASET_COLUMNS",
    "DENIED_CATEGORIES",
    "product_to_card",
    "avail",
    # Embeddings
    "EmbeddingEngine",
    # Index
    "VectorIndex",
    "MockVectorIndex",
    # Catalog loading
    "load_catalog",
    "generate_synthetic_catalog",
    "load_personas",
    "generate_persona_weights",
]
