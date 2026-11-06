"""Inverted file (IVF) index for approximate nearest neighbor search.

Groups unit vectors into clusters with spherical k-means, then answers a query
by searching only the clusters whose centroids are closest to it.
"""

import numpy as np

from .engine import l2_normalize

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
