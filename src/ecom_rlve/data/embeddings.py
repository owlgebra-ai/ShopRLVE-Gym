"""Embedding pipeline for EcomRLVE-GYM.

Spec Section 1.2:
    - product embedding:  e_p = normalize(f_enc(title(p) + " " + desc(p)))
    - query embedding:    e_q = normalize(f_enc(q))
    - cosine similarity:  sim_vec(q,p) = e_q^T e_p  (already normalized)
    - scaled similarity:  sim01(q,p) = (sim_vec(q,p) + 1) / 2

The EmbeddingEngine wraps SentenceTransformer with lazy loading,
batch encoding, and a debug mode that returns deterministic random
embeddings (seeded) for testing without loading a real model.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ecom_rlve.data.schema import Product

logger = logging.getLogger(__name__)

# Default embedding dimension for thenlper/gte-small (and all-MiniLM-L6-v2)
_DEFAULT_DIM = 384


class EmbeddingEngine:
    """Embedding engine for products and queries.

    Wraps a SentenceTransformer model with lazy initialization, batch
    encoding, and a debug mode for fast, deterministic testing.

    Attributes:
        model_name: Name of the SentenceTransformer model.
        device:     Device string (e.g. "cpu", "cuda"). None = auto-detect.
        debug_mode: When True, skip the real model and return deterministic
                    random embeddings seeded from input text. This is the
                    primary DEBUG lever for the embedding pipeline.
        dim:        Embedding dimension. Set automatically when model loads,
                    or defaults to 384 in debug mode.

    Example:
        >>> engine = EmbeddingEngine(debug_mode=True)
        >>> vec = engine.encode_query("wireless headphones")
        >>> vec.shape
        (384,)
        >>> np.linalg.norm(vec)  # normalized
        1.0
    """

    debug_mode: bool = False

    def __init__(
        self,
        model_name: str = "thenlper/gte-small",
        device: str | None = None,
        *,
        debug_mode: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None  # Lazy-loaded

        # Instance-level override takes precedence over class-level default
        if debug_mode is not None:
            self.debug_mode = debug_mode

        # Dimension: set when model loads; use default for debug mode
        self.dim: int = _DEFAULT_DIM

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the SentenceTransformer model on first real encode call."""
        if self._model is not None:
            return

        if self.debug_mode:
            logger.info(
                "EmbeddingEngine: debug_mode=True, skipping model load. "
                "Returning deterministic random embeddings."
            )
            return

        logger.info("EmbeddingEngine: loading model '%s' ...", self.model_name)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for real embeddings. "
                "Install it with: uv add sentence-transformers"
            ) from exc

        kwargs = {}
        if self.device is not None:
            kwargs["device"] = self.device

        self._model = SentenceTransformer(self.model_name, **kwargs)
        # Update dim from the loaded model
        self.dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            "EmbeddingEngine: model loaded (dim=%d, device=%s)",
            self.dim,
            self._model.device,
        )

    # ------------------------------------------------------------------
    # Debug embedding (deterministic, seeded from text hash)
    # ------------------------------------------------------------------

    def _debug_encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Generate deterministic pseudo-random embeddings from text hashes.

        Each text is hashed (SHA-256) to derive a seed, then a random vector
        is generated. If normalize=True, vectors are L2-normalized.
        This ensures identical inputs always produce identical outputs.
        """
        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Derive a deterministic seed from text content
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            seed = int(text_hash[:8], 16)  # Use first 8 hex chars
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
            if normalize:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            embeddings[i] = vec
        return embeddings

    # ------------------------------------------------------------------
    # Core encoding methods
    # ------------------------------------------------------------------

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Batch-encode a list of text strings into embeddings.

        Args:
            texts:     List of strings to encode.
            normalize: If True, L2-normalize each embedding vector.

        Returns:
            np.ndarray of shape (len(texts), dim), dtype float32.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        self._load_model()

        if self.debug_mode:
            return self._debug_encode(texts, normalize=normalize)

        # Real model encoding
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_product(self, product: Product) -> np.ndarray:
        """Encode a product into an embedding vector.

        Spec Section 1.2 / 3.1:
            e_p = normalize(f_enc(title(p) + " " + desc(p)))

        Args:
            product: Product instance.

        Returns:
            1-D np.ndarray of shape (dim,), L2-normalized.
        """
        text = f"{product.title} {product.desc}".strip()
        embeddings = self.encode([text], normalize=True)
        return embeddings[0]

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a search query into an embedding vector.

        Spec Section 1.2:
            e_q = normalize(f_enc(q))

        Args:
            query: Search query string.

        Returns:
            1-D np.ndarray of shape (dim,), L2-normalized.
        """
        embeddings = self.encode([query], normalize=True)
        return embeddings[0]

    # ------------------------------------------------------------------
    # Similarity functions
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two (already L2-normalized) vectors.

        Spec Section 1.2:
            sim_vec(q,p) = e_q^T e_p  (since both are normalized)

        Result is in [-1, 1].

        Args:
            a: 1-D normalized vector.
            b: 1-D normalized vector.

        Returns:
            Cosine similarity as a Python float.
        """
        return float(np.dot(a, b))

    @staticmethod
    def sim01(a: np.ndarray, b: np.ndarray) -> float:
        """Similarity scaled to [0, 1].

        Spec Section 1.2:
            sim01(q,p) = (sim_vec(q,p) + 1) / 2

        Args:
            a: 1-D normalized vector.
            b: 1-D normalized vector.

        Returns:
            Scaled similarity in [0, 1] as a Python float.
        """
        return float((np.dot(a, b) + 1.0) / 2.0)
