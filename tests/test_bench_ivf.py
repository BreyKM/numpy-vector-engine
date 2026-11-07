"""Tests for the IVF recall benchmark helpers."""

import sys

import numpy as np

from benchmarks import bench_ivf
from benchmarks.bench_ivf import benchmark_n_probe, recall_at_k, scanned_fraction
from benchmarks.bench_search import random_unit_vectors
from src.engine import top_k_cosine
from src.ivf import IVFIndex


def test_recall_at_k_averages_found_share_and_ignores_order():
    found = [np.array([2, 1]), np.array([3, 4])]
    truth = [np.array([1, 2]), np.array([3, 5])]

    assert recall_at_k(found, truth) == 0.75


def test_recall_at_k_counts_missing_results_as_misses():
    found = [np.array([7])]
    truth = [np.array([7, 8, 9, 10])]

    assert recall_at_k(found, truth) == 0.25


def test_scanned_fraction_counts_the_size_of_each_probed_list():
    # Two lists of unequal size: list 0 holds 1 vector, list 1 holds 3.
    centroids = np.eye(2, dtype=np.float32)
    vectors = np.array([[1, 0], [0, 1], [0, 1], [0, 1]], dtype=np.float32)
    offsets = np.array([0, 1, 4])
    index = IVFIndex(centroids, vectors, ids=np.arange(4), offsets=offsets)
    queries = [np.array([0, 1], dtype=np.float32)]

    assert scanned_fraction(index, queries, n_probe=1) == 0.75
    assert scanned_fraction(index, queries, n_probe=2) == 1.0


def test_benchmark_n_probe_has_full_recall_when_every_list_is_probed():
    rng = np.random.default_rng(seed=0)
    corpus = random_unit_vectors(rng, 200)
    queries = list(random_unit_vectors(rng, 5))
    truth = [top_k_cosine(corpus, query, 10)[0] for query in queries]
    index = IVFIndex.build(corpus, n_lists=4)

    row = benchmark_n_probe(index, queries, truth, n_probe=4, exact_p50_ms=1.0)

    assert row.startswith("| 4 | 100.0 | 1.000 |")


def test_main_prints_a_row_for_exact_search_and_each_n_probe(
    tmp_path, monkeypatch, capsys
):
    rng = np.random.default_rng(seed=0)
    np.save(tmp_path / "corpus.npy", random_unit_vectors(rng, 200))
    np.save(tmp_path / "queries.npy", random_unit_vectors(rng, 5))
    monkeypatch.setattr(bench_ivf, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["bench_ivf", "--n-lists", "4", "--n-probes", "1", "4"]
    )

    bench_ivf.main()

    lines = capsys.readouterr().out.splitlines()
    assert lines[-3].startswith("| exact | 100.0 | 1.000 |")
    assert lines[-2].startswith("| 1 |")
    assert lines[-1].startswith("| 4 | 100.0 | 1.000 |")
