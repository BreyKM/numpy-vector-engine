"""Benchmarks IVF search on real embeddings: recall against speed.

Exact search over every vector is the ground truth. For each n_probe, the
benchmark reports recall@k (the share of the true top k that IVF also found),
the share of vectors IVF actually scored, and its latency percentiles.

Create the embeddings first, then run from the project root:

    python -m benchmarks.prepare_wikipedia
    python -m benchmarks.bench_ivf
"""

import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np

from benchmarks.bench_search import TOP_K, percentile_cells, time_ms
from src.engine import top_k_cosine
from src.ivf import IVFIndex

# Must match where benchmarks/prepare_wikipedia.py saves its output. That
# module is not imported here because it needs pyarrow, which CI does not install.
DATA_DIR = Path("data") / "wikipedia"
CORPUS_FILE = "corpus.npy"
QUERIES_FILE = "queries.npy"
DEFAULT_N_LISTS = 1_000
DEFAULT_TRAIN_SIZE = 100_000
DEFAULT_N_PROBES = [1, 2, 4, 8, 16, 32, 64, 128]


def recall_at_k(found: list[np.ndarray], truth: list[np.ndarray]) -> float:
    """Measures how many of the true nearest neighbors a search found.

    Order is ignored: a true neighbor counts as found anywhere in the results.

    Args:
        found (list[np.ndarray]): Ids the search returned, one array per query.
        truth (list[np.ndarray]): True top-k ids, one array per query.

    Returns:
        The found share of each query's true neighbors, averaged over all
        queries, from 0.0 to 1.0.
    """
    hits = [
        len(np.intersect1d(ids, true_ids)) / len(true_ids)
        for ids, true_ids in zip(found, truth, strict=True)
    ]
    return float(np.mean(hits))


def scanned_fraction(index: IVFIndex, queries: list[np.ndarray], n_probe: int) -> float:
    """Measures the average share of all vectors a search scores.

    Lists differ in size, so this is usually not exactly n_probe / n_lists.

    Args:
        index (IVFIndex): The index being searched.
        queries (list[np.ndarray]): Unit query vectors, each shape (D,).
        n_probe (int): Number of lists each search probes.

    Returns:
        The share of indexed vectors scored per query, from 0.0 to 1.0.
    """
    list_sizes = np.diff(index.offsets)
    scanned = [
        list_sizes[top_k_cosine(index.centroids, query, n_probe)[0]].sum()
        for query in queries
    ]
    return float(np.mean(scanned)) / len(index.vectors)


def benchmark_n_probe(
    index: IVFIndex,
    queries: list[np.ndarray],
    truth: list[np.ndarray],
    n_probe: int,
    exact_p50_ms: float,
) -> str:
    """Measures recall and latency of IVF search at one n_probe.

    Args:
        index (IVFIndex): The index to search.
        queries (list[np.ndarray]): Unit query vectors, each shape (D,).
        truth (list[np.ndarray]): Exact top-k ids for each query.
        n_probe (int): Number of lists each search probes.
        exact_p50_ms (float): Median exact search time, for the speed-up.

    Returns:
        One markdown table row.
    """
    search = partial(index.search, k=TOP_K, n_probe=n_probe)
    found = [search(query)[0] for query in queries]
    durations = time_ms(search, queries)
    speed_up = exact_p50_ms / np.percentile(durations, 50)
    return (
        f"| {n_probe} | {100 * scanned_fraction(index, queries, n_probe):.1f} | "
        f"{recall_at_k(found, truth):.3f} | {percentile_cells(durations)} | "
        f"{speed_up:.1f}x |"
    )


def main() -> None:
    """Builds an IVF index and prints recall and latency for each n_probe."""
    parser = argparse.ArgumentParser(description="Benchmark IVF recall and speed.")
    parser.add_argument("--n-lists", type=int, default=DEFAULT_N_LISTS)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--n-probes", type=int, nargs="+", default=DEFAULT_N_PROBES)
    args = parser.parse_args()

    if not (DATA_DIR / CORPUS_FILE).exists():
        parser.error(
            f"{DATA_DIR / CORPUS_FILE} not found. "
            "Run python -m benchmarks.prepare_wikipedia first."
        )
    corpus = np.load(DATA_DIR / CORPUS_FILE)
    queries = list(np.load(DATA_DIR / QUERIES_FILE))

    exact = partial(top_k_cosine, corpus, k=TOP_K)
    truth = [exact(query)[0] for query in queries]
    exact_ms = time_ms(exact, queries)
    exact_p50_ms = float(np.percentile(exact_ms, 50))

    start = time.perf_counter()
    index = IVFIndex.build(corpus, args.n_lists, train_size=args.train_size)
    build_s = time.perf_counter() - start

    print(
        f"{len(corpus):,} vectors, {len(queries):,} held-out queries, top {TOP_K}. "
        f"{args.n_lists:,} lists, k-means trained on "
        f"{min(args.train_size, len(corpus)):,} vectors, built in {build_s:.1f} s.\n"
    )
    print(
        "| n_probe | Vectors scored (%) | Recall@10 | p50 (ms) | p95 (ms) "
        "| p99 (ms) | Speed-up (p50) |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    print(f"| exact | 100.0 | 1.000 | {percentile_cells(exact_ms)} | 1.0x |")
    for n_probe in args.n_probes:
        print(benchmark_n_probe(index, queries, truth, n_probe, exact_p50_ms))


if __name__ == "__main__":
    main()
