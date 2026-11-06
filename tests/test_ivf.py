"""Tests for the IVF index and its spherical k-means clustering."""

import numpy as np
import pytest

from src.engine import l2_normalize, top_k_cosine
from src.ivf import IVFIndex, assign_to_centroids, spherical_kmeans


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


def random_unit_vectors(n, dimension=16, seed=0):
    """Builds unit vectors with no cluster structure at all."""
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.standard_normal((n, dimension))).astype(np.float32)


@pytest.mark.parametrize("train_size", [None, 100])
def test_ivf_search_probing_every_list_matches_exact_search(train_size):
    embeddings = random_unit_vectors(500)
    index = IVFIndex.build(embeddings, n_lists=10, train_size=train_size)
    query = embeddings[0]

    ids, scores = index.search(query, k=10, n_probe=10)

    expected_ids, expected_scores = top_k_cosine(embeddings, query, 10)
    assert ids.tolist() == expected_ids.tolist()
    assert np.allclose(scores, expected_scores)


def test_ivf_search_one_probe_finds_neighbors_in_tight_clusters():
    vectors, _ = make_clusters(n_clusters=8, points_per_cluster=50)
    index = IVFIndex.build(vectors, n_lists=8)
    query = vectors[0]

    ids, _ = index.search(query, k=5, n_probe=1)

    expected_ids, _ = top_k_cosine(vectors, query, 5)
    assert ids.tolist() == expected_ids.tolist()


def test_ivf_build_stores_vectors_grouped_by_list():
    vectors, _ = make_clusters(n_clusters=4, points_per_cluster=25)

    index = IVFIndex.build(vectors, n_lists=4)

    assert np.array_equal(index.vectors, vectors[index.ids])
    labels = assign_to_centroids(index.vectors, index.centroids)
    list_sizes = np.diff(index.offsets)
    assert np.array_equal(labels, np.repeat(np.arange(4), list_sizes))


def test_ivf_search_returns_fewer_results_when_probed_lists_are_small():
    vectors, _ = make_clusters(n_clusters=4, points_per_cluster=25)
    index = IVFIndex.build(vectors, n_lists=4)

    ids, scores = index.search(vectors[0], k=100, n_probe=1)

    assert len(ids) == len(scores) == 25


@pytest.mark.parametrize("bad_n_probe", [0, 5])
def test_ivf_search_invalid_n_probe_raises(bad_n_probe):
    vectors, _ = make_clusters(n_clusters=4, points_per_cluster=25)
    index = IVFIndex.build(vectors, n_lists=4)
    with pytest.raises(ValueError, match="n_probe"):
        index.search(vectors[0], k=5, n_probe=bad_n_probe)


def test_ivf_build_training_sample_smaller_than_n_lists_raises():
    embeddings = random_unit_vectors(100)
    with pytest.raises(ValueError, match="train_size"):
        IVFIndex.build(embeddings, n_lists=10, train_size=5)
