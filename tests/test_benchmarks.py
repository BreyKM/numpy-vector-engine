"""Tests for the search benchmark helpers."""

import numpy as np

from benchmarks.bench_search import (
    benchmark_size,
    full_sort_top_k,
    random_unit_vectors,
)
from src.engine import top_k_cosine


def test_random_unit_vectors_have_model_shape_and_unit_length():
    vectors = random_unit_vectors(np.random.default_rng(seed=0), 5)

    assert vectors.shape == (5, 384)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_full_sort_baseline_matches_top_k_cosine():
    rng = np.random.default_rng(seed=0)
    embeddings = random_unit_vectors(rng, 1000)
    query = random_unit_vectors(rng, 1)[0]

    expected_indices, expected_scores = top_k_cosine(embeddings, query, 10)
    indices, scores = full_sort_top_k(embeddings, query, 10)

    assert indices.tolist() == expected_indices.tolist()
    assert np.allclose(scores, expected_scores)


def test_benchmark_size_reports_one_row_per_method():
    rows = benchmark_size(np.random.default_rng(seed=0), n=1000, repeats=5)

    assert len(rows) == 3
    assert rows[0].startswith("| 1,000 |")
