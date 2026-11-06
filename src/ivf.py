"""Inverted file (IVF) index for approximate nearest neighbor search.

Groups unit vectors into clusters with spherical k-means, then answers a query
by searching only the clusters whose centroids are closest to it.
"""

from typing import Self

import numpy as np

from .engine import l2_normalize, select_top_k, top_k_cosine

ASSIGN_CHUNK_SIZE = 16_384


def assign_to_centroids(
    vectors: np.ndarray, centroids: np.ndarray, chunk_size: int = ASSIGN_CHUNK_SIZE
) -> np.ndarray:
    """Finds the most similar centroid for every vector.

    Works through the vectors in chunks, so it never holds the full
    (N, n_centroids) similarity matrix in memory at once.

    Args:
        vectors (np.ndarray): Unit vectors, shape (N, D).
        centroids (np.ndarray): Unit centroids, shape (C, D).
        chunk_size (int): Number of vectors scored at a time.

    Returns:
        An int64 array of shape (N,) holding each vector's centroid index.
    """
    labels = np.empty(len(vectors), dtype=np.int64)
    for start in range(0, len(vectors), chunk_size):
        chunk = vectors[start : start + chunk_size]
        labels[start : start + chunk_size] = np.argmax(chunk @ centroids.T, axis=1)
    return labels


def kmeans_plus_plus(
    vectors: np.ndarray, n_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    """Picks starting centroids that are spread out across the data.

    The first centroid is a random vector. Each later one is a vector chosen
    with probability proportional to its squared cosine distance from the
    nearest centroid picked so far, so far-away regions are likely to get a
    centroid of their own.

    Args:
        vectors (np.ndarray): Unit vectors, shape (N, D).
        n_clusters (int): Number of centroids to pick.
        rng (np.random.Generator): Random generator for the choices.

    Returns:
        An array of shape (n_clusters, D) holding copies of the chosen vectors.
    """
    centroids = np.empty((n_clusters, vectors.shape[1]), dtype=vectors.dtype)
    centroids[0] = vectors[rng.integers(len(vectors))]
    best_similarity = vectors @ centroids[0]

    for i in range(1, n_clusters):
        distances = np.maximum(1.0 - best_similarity, 0.0).astype(np.float64) ** 2
        total = distances.sum()
        if total > 0:
            index = rng.choice(len(vectors), p=distances / total)
        else:
            index = rng.integers(len(vectors))
        centroids[i] = vectors[index]
        np.maximum(best_similarity, vectors @ centroids[i], out=best_similarity)

    return centroids


def spherical_kmeans(
    vectors: np.ndarray, n_clusters: int, n_iter: int = 20, seed: int = 0
) -> np.ndarray:
    """Clusters unit vectors by direction.

    Like k-means, but similarity is the dot product and every centroid is
    rescaled to unit length after averaging, which suits cosine similarity.
    Starts from k-means++ centroids and stops early once no vector changes
    cluster. A cluster that ends up empty keeps its previous centroid.

    Args:
        vectors (np.ndarray): Unit vectors to cluster, shape (N, D).
        n_clusters (int): Number of clusters, from 1 to N.
        n_iter (int): Maximum number of refinement rounds. Defaults to 20.
        seed (int): Seed for the random choices, so results are repeatable.

    Returns:
        Unit-length centroids of shape (n_clusters, D).

    Raises:
        ValueError: If vectors is not 2D, n_clusters is outside 1 to N, or
            n_iter is less than 1.
    """
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape {vectors.shape}.")
    if not 1 <= n_clusters <= len(vectors):
        raise ValueError(
            f"n_clusters must be between 1 and {len(vectors)}, got {n_clusters}."
        )
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")

    rng = np.random.default_rng(seed)
    centroids = kmeans_plus_plus(vectors, n_clusters, rng)
    labels = None

    for _ in range(n_iter):
        new_labels = assign_to_centroids(vectors, centroids)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        sums = np.zeros_like(centroids)
        np.add.at(sums, labels, vectors)
        non_empty = np.bincount(labels, minlength=n_clusters) > 0
        centroids[non_empty] = l2_normalize(sums[non_empty])

    return centroids


class IVFIndex:
    """An inverted file index for approximate search over unit vectors.

    Every vector belongs to the list of its nearest k-means centroid. Vectors
    are stored sorted by list, so each list is one contiguous block of memory.
    A search scores the query against the centroids, then scores only the
    vectors in the n_probe closest lists.

    Create an index with IVFIndex.build rather than the constructor.

    Attributes:
        centroids (np.ndarray): Unit centroid of each list, shape (L, D).
        vectors (np.ndarray): The indexed vectors, sorted by list, shape (N, D).
        ids (np.ndarray): Original row position of each sorted vector, shape (N,).
        offsets (np.ndarray): List i holds sorted rows offsets[i] up to, but not
            including, offsets[i + 1]. Shape (L + 1,).
    """

    def __init__(
        self,
        centroids: np.ndarray,
        vectors: np.ndarray,
        ids: np.ndarray,
        offsets: np.ndarray,
    ) -> None:
        """Stores the arrays of an already-built index.

        Args:
            centroids (np.ndarray): Unit centroid of each list, shape (L, D).
            vectors (np.ndarray): Vectors sorted by list, shape (N, D).
            ids (np.ndarray): Original row position of each vector, shape (N,).
            offsets (np.ndarray): Start of each list, then N, shape (L + 1,).
        """
        self.centroids = centroids
        self.vectors = vectors
        self.ids = ids
        self.offsets = offsets

    @property
    def n_lists(self) -> int:
        """The number of lists, which is also the number of centroids."""
        return len(self.centroids)

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        n_lists: int,
        train_size: int | None = None,
        n_iter: int = 20,
        seed: int = 0,
    ) -> Self:
        """Clusters the embeddings and builds an index over them.

        Args:
            embeddings (np.ndarray): Unit vectors to index, shape (N, D).
            n_lists (int): Number of lists, from 1 to N. About the square
                root of N is a common starting point.
            train_size (int | None): If set, k-means trains on this many
                randomly chosen embeddings instead of all of them, which is
                much faster. Must be at least n_lists. Defaults to None.
            n_iter (int): Maximum k-means rounds. Defaults to 20.
            seed (int): Seed for sampling and k-means, so builds are repeatable.

        Returns:
            A new IVFIndex holding its own copy of the embeddings.

        Raises:
            ValueError: If embeddings is not 2D, n_lists is outside 1 to N, or
                train_size is smaller than n_lists.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}.")
        if not 1 <= n_lists <= len(embeddings):
            raise ValueError(
                f"n_lists must be between 1 and {len(embeddings)}, got {n_lists}."
            )
        if train_size is not None and train_size < n_lists:
            raise ValueError(
                f"train_size must be at least n_lists ({n_lists}), got {train_size}."
            )

        training = embeddings
        if train_size is not None and train_size < len(embeddings):
            rng = np.random.default_rng(seed)
            sample = rng.choice(len(embeddings), size=train_size, replace=False)
            training = embeddings[sample]
        centroids = spherical_kmeans(training, n_lists, n_iter=n_iter, seed=seed)

        labels = assign_to_centroids(embeddings, centroids)
        ids = np.argsort(labels, kind="stable")
        counts = np.bincount(labels, minlength=n_lists)
        offsets = np.concatenate(([0], np.cumsum(counts)))

        return cls(centroids, embeddings[ids], ids, offsets)

    def search(
        self, query_vector: np.ndarray, k: int, n_probe: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Finds approximately the k indexed vectors most similar to a query.

        Only vectors in the n_probe lists whose centroids are closest to the
        query are scored, so a true neighbor in a list that is not probed will
        be missed. Probing more lists finds more of them but takes longer. With
        n_probe equal to n_lists, every vector is scored and the result is exact.

        Args:
            query_vector (np.ndarray): Unit query vector, shape (D,).
            k (int): Maximum number of results, at least 1.
            n_probe (int): Number of lists to search, from 1 to n_lists.
                Defaults to 1.

        Returns:
            A tuple (ids, scores), highest score first. The ids are row
            positions in the embeddings the index was built from. There are
            fewer than k results if the probed lists hold fewer than k vectors.

        Raises:
            ValueError: If k is less than 1 or n_probe is outside 1 to n_lists.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}.")
        if not 1 <= n_probe <= self.n_lists:
            raise ValueError(
                f"n_probe must be between 1 and {self.n_lists}, got {n_probe}."
            )

        nearest_lists, _ = top_k_cosine(self.centroids, query_vector, n_probe)

        score_parts = []
        id_parts = []
        for list_id in nearest_lists:
            start, end = self.offsets[list_id], self.offsets[list_id + 1]
            score_parts.append(self.vectors[start:end] @ query_vector)
            id_parts.append(self.ids[start:end])
        scores = np.concatenate(score_parts)
        ids = np.concatenate(id_parts)

        if len(scores) == 0:
            return ids, scores

        top, top_scores = select_top_k(scores, min(k, len(scores)))
        return ids[top], top_scores
