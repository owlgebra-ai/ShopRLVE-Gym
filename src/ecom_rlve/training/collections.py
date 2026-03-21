"""Environment collection definitions for EcomRLVE-GYM (Spec Section 9).

Collections group atomic environments into curriculum stages:

    C1 = ["PD"]                                              -- single env baseline
    C2 = ["PD", "SUB"]                                       -- add substitution
    C4 = ["PD", "SUB", "CART", "RETURN"]                     -- add transactional
    C8 = ["PD", "SUB", "CART", "RETURN",
          "STATUS", "POLICY", "BUNDLE", "JOURNEY"]            -- all 8

Training progresses C1 -> C2 -> C4 -> C8, with adaptive difficulty
independently tracked per environment within each collection.

Usage:
    from ecom_rlve.training.collections import get_collection, COLLECTIONS

    env_ids = get_collection("C4")
    # ["PD", "SUB", "CART", "RETURN"]
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Canonical collection definitions
# ---------------------------------------------------------------------------

COLLECTIONS: dict[str, list[str]] = {
    "C1": ["PD"],
    "C2": ["PD", "SUB"],
    "C4": ["PD", "SUB", "CART", "RETURN"],
    "C8": ["PD", "SUB", "CART", "RETURN", "STATUS", "POLICY", "BUNDLE", "JOURNEY"],
}

# All valid environment IDs (union of all collections)
ALL_ENV_IDS: frozenset[str] = frozenset(COLLECTIONS["C8"])


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_collection(name: str) -> list[str]:
    """Get the list of environment IDs for a named collection.

    Args:
        name: Collection name (C1, C2, C4, or C8).

    Returns:
        List of environment ID strings.

    Raises:
        ValueError: If *name* is not a recognized collection.
    """
    if name not in COLLECTIONS:
        available = sorted(COLLECTIONS.keys())
        raise ValueError(
            f"Unknown collection '{name}'. Available: {available}"
        )
    return list(COLLECTIONS[name])


def validate_collection(name: str) -> bool:
    """Check whether a collection name is valid.

    Args:
        name: Collection name to validate.

    Returns:
        True if *name* is a recognized collection, False otherwise.
    """
    return name in COLLECTIONS


def validate_env_ids(env_ids: list[str]) -> tuple[bool, list[str]]:
    """Validate a list of environment IDs against the known set.

    Args:
        env_ids: List of environment ID strings to validate.

    Returns:
        Tuple of (all_valid, invalid_ids):
            - all_valid: True if every ID is recognized.
            - invalid_ids: List of unrecognized IDs (empty if all valid).
    """
    invalid = [eid for eid in env_ids if eid not in ALL_ENV_IDS]
    return len(invalid) == 0, invalid


def collection_info() -> dict[str, dict[str, Any]]:
    """Return metadata about all collections.

    Returns:
        Dict mapping collection name -> info dict with keys:
            - env_ids: list of env ID strings
            - size: number of environments
            - description: human-readable description
    """
    descriptions: dict[str, str] = {
        "C1": "Single environment baseline (Product Discovery only)",
        "C2": "Product Discovery + Substitution",
        "C4": "Core transactional environments (PD, SUB, CART, RETURN)",
        "C8": "All 8 atomic environments",
    }
    return {
        name: {
            "env_ids": list(env_ids),
            "size": len(env_ids),
            "description": descriptions.get(name, ""),
        }
        for name, env_ids in COLLECTIONS.items()
    }
