"""FAISS vector index for ShopRLVE-GYM catalog retrieval.

Spec Section 3.1:
    - Build ANN index (IVF-PQ/HNSW/Flat) over product embeddings.
    - query scoring: score_vec(q,p) = e_q^T e_p
    - Support metadata filtering as a post-retrieval step.

Provides:
    - VectorIndex: FAISS-backed index with build/search/save/load.
    - MockVectorIndex: Brute-force cosine similarity in a dict for
      testing without FAISS. This is the DEBUG lever for the index layer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class VectorIndex:
    """FAISS-backed vector index for product embeddings.

    Supports building from numpy arrays, searching by query embedding,
    and persisting/loading from disk.

    Args:
        dim:           Embedding dimension (default 384 for all-MiniLM-L6-v2).
        index_factory: FAISS index factory string. Examples:
                       "Flat" (exact search, small catalogs),
                       "IVF256,PQ32" (ANN for large catalogs),
                       "HNSW32" (graph-based ANN).
        use_gpu:       If True, attempt to move index to GPU (requires faiss-gpu).

    Example:
        >>> import numpy as np
        >>> idx = VectorIndex(dim=384, index_factory="Flat")
        >>> embeddings = np.random.randn(100, 384).astype(np.float32)
        >>> embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        >>> ids = [f"p_{i}" for i in range(100)]
        >>> idx.build(embeddings, ids)
        >>> results = idx.search(embeddings[0], top_k=5)
        >>> len(results)
        5
    """

    def __init__(
        self,
        dim: int = 384,
        index_factory: str = "Flat",
        use_gpu: bool = False,
    ) -> None:
        self.dim = dim
        self.index_factory = index_factory
        self.use_gpu = use_gpu

        self._index = None  # faiss.Index, built lazily
        self._id_map: list[str] = []  # positional index -> product_id
        self._id_to_pos: dict[str, int] = {}  # product_id -> positional index

    def __len__(self) -> int:
        """Number of indexed vectors."""
        if self._index is None:
            return 0
        return self._index.ntotal

    @property
    def is_built(self) -> bool:
        """Whether the index has been built with embeddings."""
        return self._index is not None and self._index.ntotal > 0

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None:
        """Build the FAISS index from embeddings and product IDs.

        Args:
            embeddings: np.ndarray of shape (n, dim), dtype float32.
                        Should be L2-normalized for cosine similarity.
            ids:        List of product ID strings, length n.

        Raises:
            ValueError: If embeddings shape doesn't match dim or ids length.
            ImportError: If faiss is not installed.
        """
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for VectorIndex. "
                "Install with: uv add faiss-cpu"
            ) from exc

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Expected embeddings of shape (n, {self.dim}), "
                f"got {embeddings.shape}"
            )
        if embeddings.shape[0] != len(ids):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings vs {len(ids)} ids"
            )

        n = embeddings.shape[0]
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Build FAISS index using the factory string
        logger.info(
            "VectorIndex: building index (factory='%s', n=%d, dim=%d)",
            self.index_factory,
            n,
            self.dim,
        )
        index = faiss.index_factory(self.dim, self.index_factory, faiss.METRIC_INNER_PRODUCT)

        # Some index types (IVF) require training
        if not index.is_trained:
            logger.info("VectorIndex: training index ...")
            # Use a subset if dataset is very large
            train_n = min(n, max(256 * 100, n))  # at least 256*nlist samples
            train_data = embeddings[:train_n]
            index.train(train_data)

        index.add(embeddings)

        # Optionally move to GPU
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("VectorIndex: moved index to GPU")
            except Exception:
                logger.warning("VectorIndex: GPU transfer failed, staying on CPU")

        self._index = index
        self._id_map = list(ids)
        self._id_to_pos = {pid: pos for pos, pid in enumerate(ids)}

        logger.info("VectorIndex: index built successfully (%d vectors)", n)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
    ) -> list[tuple[str, float]]:
        """Search the index for the most similar product embeddings.

        Spec Section 3.1:
            score_vec(q,p) = e_q^T e_p  (inner product on normalized vectors)

        Args:
            query_embedding: 1-D np.ndarray of shape (dim,), L2-normalized.
            top_k:           Number of results to return.

        Returns:
            List of (product_id, score) tuples, sorted by descending score.
            Score is the inner product (= cosine similarity for normalized vectors).
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("VectorIndex: search called on empty index")
            return []

        # Clamp top_k to available vectors
        effective_k = min(top_k, self._index.ntotal)

        # FAISS expects a 2-D query array
        query = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32
        )

        distances, indices = self._index.search(query, effective_k)

        results: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                # FAISS uses -1 for missing results
                continue
            product_id = self._id_map[idx]
            results.append((product_id, float(dist)))

        return results

    def get_embedding(self, product_id: str) -> np.ndarray | None:
        """Retrieve the stored embedding for a product ID.

        Only works for index types that support reconstruct (e.g., Flat).

        Args:
            product_id: Product identifier.

        Returns:
            1-D np.ndarray of shape (dim,) or None if not found / unsupported.
        """
        if product_id not in self._id_to_pos or self._index is None:
            return None
        pos = self._id_to_pos[product_id]
        try:
            return self._index.reconstruct(pos).astype(np.float32)
        except RuntimeError:
            logger.warning(
                "VectorIndex: reconstruct not supported for index type '%s'",
                self.index_factory,
            )
            return None

    def save(self, path: str) -> None:
        """Persist the FAISS index and ID mapping to disk.

        Creates two files:
            - {path}.index : the FAISS binary index
            - {path}.ids.json : the product ID list

        Args:
            path: Base path (without extension).
        """
        import faiss

        if self._index is None:
            raise RuntimeError("Cannot save: index has not been built")

        index_path = Path(f"{path}.index")
        ids_path = Path(f"{path}.ids.json")

        # Ensure parent directory exists
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # If index is on GPU, move to CPU before saving
        try:
            cpu_index = faiss.index_gpu_to_cpu(self._index)
        except Exception:
            cpu_index = self._index

        faiss.write_index(cpu_index, str(index_path))

        with open(ids_path, "w") as f:
            json.dump(self._id_map, f)

        logger.info(
            "VectorIndex: saved to %s (.index + .ids.json)", path
        )

    def load(self, path: str) -> None:
        """Load a FAISS index and ID mapping from disk.

        Args:
            path: Base path used in save() (without extension).

        Raises:
            FileNotFoundError: If index or ID files are missing.
        """
        import faiss

        index_path = Path(f"{path}.index")
        ids_path = Path(f"{path}.ids.json")

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not ids_path.exists():
            raise FileNotFoundError(f"ID file not found: {ids_path}")

        self._index = faiss.read_index(str(index_path))

        with open(ids_path) as f:
            self._id_map = json.load(f)

        self._id_to_pos = {pid: pos for pos, pid in enumerate(self._id_map)}
        self.dim = self._index.d

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
            except Exception:
                logger.warning("VectorIndex: GPU transfer failed on load")

        logger.info(
            "VectorIndex: loaded %d vectors from %s", len(self), path
        )


class MockVectorIndex:
    """Brute-force cosine similarity index for testing without FAISS.

    This is the DEBUG lever for the index layer. Stores embeddings in a
    plain dict and computes exact cosine similarity on every search.
    Functionally equivalent to VectorIndex with index_factory="Flat"
    but requires no external dependencies.

    Example:
        >>> import numpy as np
        >>> idx = MockVectorIndex(dim=384)
        >>> emb = np.random.randn(10, 384).astype(np.float32)
        >>> emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        >>> ids = [f"p_{i}" for i in range(10)]
        >>> idx.build(emb, ids)
        >>> results = idx.search(emb[0], top_k=3)
        >>> results[0][0]  # best match is itself
        'p_0'
    """

    def __init__(self, dim: int = 384, **_kwargs: object) -> None:
        self.dim = dim
        self._embeddings: np.ndarray | None = None
        self._id_map: list[str] = []
        self._id_to_pos: dict[str, int] = {}

    def __len__(self) -> int:
        if self._embeddings is None:
            return 0
        return self._embeddings.shape[0]

    @property
    def is_built(self) -> bool:
        return self._embeddings is not None and self._embeddings.shape[0] > 0

    def build(self, embeddings: np.ndarray, ids: list[str]) -> None:
        """Store embeddings and IDs for brute-force search.

        Args:
            embeddings: np.ndarray of shape (n, dim), dtype float32.
            ids:        List of product ID strings.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Expected shape (n, {self.dim}), got {embeddings.shape}"
            )
        if embeddings.shape[0] != len(ids):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings vs {len(ids)} ids"
            )
        self._embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._id_map = list(ids)
        self._id_to_pos = {pid: pos for pos, pid in enumerate(ids)}
        logger.info("MockVectorIndex: built with %d vectors", len(ids))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 100,
    ) -> list[tuple[str, float]]:
        """Brute-force cosine similarity search.

        Args:
            query_embedding: 1-D np.ndarray of shape (dim,).
            top_k:           Number of results.

        Returns:
            List of (product_id, score) tuples sorted by descending score.
        """
        if self._embeddings is None or len(self._id_map) == 0:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        # Inner product (= cosine sim for normalized vectors)
        scores = (self._embeddings @ query.T).squeeze(axis=1)

        effective_k = min(top_k, len(self._id_map))
        # Partial sort for efficiency
        top_indices = np.argpartition(-scores, effective_k)[:effective_k]
        # Sort the top-k by score
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [
            (self._id_map[idx], float(scores[idx]))
            for idx in top_indices
        ]

    def get_embedding(self, product_id: str) -> np.ndarray | None:
        """Retrieve the stored embedding for a product ID."""
        if product_id not in self._id_to_pos or self._embeddings is None:
            return None
        pos = self._id_to_pos[product_id]
        return self._embeddings[pos].copy()

    def save(self, path: str) -> None:
        """Persist embeddings and IDs to disk as numpy/json files.

        Args:
            path: Base path (without extension).
        """
        if self._embeddings is None:
            raise RuntimeError("Cannot save: index has not been built")

        emb_path = Path(f"{path}.mock.npy")
        ids_path = Path(f"{path}.mock.ids.json")
        emb_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(str(emb_path), self._embeddings)
        with open(ids_path, "w") as f:
            json.dump(self._id_map, f)

        logger.info("MockVectorIndex: saved to %s", path)

    def load(self, path: str) -> None:
        """Load embeddings and IDs from disk.

        Args:
            path: Base path used in save() (without extension).
        """
        emb_path = Path(f"{path}.mock.npy")
        ids_path = Path(f"{path}.mock.ids.json")

        if not emb_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {emb_path}")
        if not ids_path.exists():
            raise FileNotFoundError(f"ID file not found: {ids_path}")

        self._embeddings = np.load(str(emb_path)).astype(np.float32)
        with open(ids_path) as f:
            self._id_map = json.load(f)
        self._id_to_pos = {pid: pos for pos, pid in enumerate(self._id_map)}
        self.dim = self._embeddings.shape[1]
        logger.info("MockVectorIndex: loaded %d vectors from %s", len(self), path)
