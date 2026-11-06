"""Tests for the IVF index and its spherical k-means clustering."""

import numpy as np
import pytest

from src.engine import l2_normalize
from src.ivf import assign_to_centroids, spherical_kmeans


def make_clusters(n_clusters, points_per_cluster, dimension=16, seed=0):
    """Builds tight, well-separated clusters of unit vectors.

    Returns:
        The vectors, and the true cluster number of each one.
    """
    rng = np.random.default_rng(seed)
    centers = l2_normalize(rng.standard_normal((n_clusters, dimension)))
    labels = np.repeat(np.arange(n_clusters), points_per_cluster)
    noise = 0.05 * rng.standard_normal((len(labels), dimension))
    vectors = l2_normalize(centers[labels] + noise).astype(np.float32)
    return vectors, labels


def test_spherical_kmeans_recovers_separated_clusters():
    vectors, true_labels = make_clusters(n_clusters=8, points_per_cluster=50)

    centroids = spherical_kmeans(vectors, n_clusters=8)
    labels = assign_to_centroids(vectors, centroids)

    # Cluster numbers can differ from the true ones, but every true cluster
    # must map to exactly one found cluster, and no two to the same one.
    found = [set(labels[true_labels == c].tolist()) for c in range(8)]
    assert all(len(ids) == 1 for ids in found)
    assert len(set.union(*found)) == 8


def test_spherical_kmeans_returns_unit_centroids():
    vectors, _ = make_clusters(n_clusters=4, points_per_cluster=25)

    centroids = spherical_kmeans(vectors, n_clusters=4)

    assert centroids.shape == (4, 16)
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-5)


def test_spherical_kmeans_is_repeatable_with_same_seed():
    vectors, _ = make_clusters(n_clusters=4, points_per_cluster=25)

    first = spherical_kmeans(vectors, n_clusters=4, seed=1)
    second = spherical_kmeans(vectors, n_clusters=4, seed=1)

    assert np.array_equal(first, second)


@pytest.mark.parametrize("bad_n_clusters", [0, 101])
def test_spherical_kmeans_invalid_cluster_count_raises(bad_n_clusters):
    vectors, _ = make_clusters(n_clusters=4, points_per_cluster=25)
    with pytest.raises(ValueError, match="n_clusters"):
        spherical_kmeans(vectors, n_clusters=bad_n_clusters)


def test_assign_to_centroids_same_result_in_small_chunks():
    vectors, _ = make_clusters(n_clusters=4, points_per_cluster=25)
    centroids = spherical_kmeans(vectors, n_clusters=4)

    whole = assign_to_centroids(vectors, centroids)
    chunked = assign_to_centroids(vectors, centroids, chunk_size=7)

    assert np.array_equal(whole, chunked)
