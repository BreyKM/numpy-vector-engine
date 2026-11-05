"""Benchmarks exact vector search: latency percentiles and memory use.

The index is filled with random unit vectors rather than encoded documents.
Brute-force search does the same work whatever the values are, and encoding
a million real documents on a CPU would take hours. Query encoding is timed
separately with the real model, because every call to search pays that cost
on top of the vector search.

Run from the project root:

    python -m benchmarks.bench_search
"""

import argparse
import os
import platform
import time
import tracemalloc
from collections.abc import Callable
from functools import partial

import numpy as np

from src.engine import VectorIndex, l2_normalize, top_k_cosine

DIMENSION = 384
TOP_K = 10
DEFAULT_SIZES = [10_000, 100_000, 1_000_000]
DEFAULT_REPEATS = 200
WARMUP_CALLS = 3
SAMPLE_QUERIES = [
    "something to keep my coffee warm",
    "cheap phone with a good camera",
    "comfortable shoes for standing all day",
    "lightweight laptop for travel",
    "carry-on bag with wheels",
]


def random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generates random unit vectors with the same dimension as the model.

    Args:
        rng (np.random.Generator): Seeded generator, so runs are repeatable.
        n (int): Number of vectors to generate.

    Returns:
        A float32 array of shape (n, DIMENSION) whose rows have length 1.
    """
    return l2_normalize(rng.standard_normal((n, DIMENSION), dtype=np.float32))


def full_sort_top_k(
    embeddings: np.ndarray, query_vector: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline for top_k_cosine that fully sorts every score, O(N log N).

    Args:
        embeddings (np.ndarray): Unit vectors to search, shape (N, D).
        query_vector (np.ndarray): Unit query vector, shape (D,).
        k (int): Number of results to return.

    Returns:
        A tuple (indices, scores) of two length-k arrays, highest score first.
    """
    scores = embeddings @ query_vector
    top_indices = np.argsort(-scores)[:k]
    return top_indices, scores[top_indices]


def scores_only(embeddings: np.ndarray, query_vector: np.ndarray, k: int) -> np.ndarray:
    """Scores every vector without selecting a top k, to isolate that cost.

    Args:
        embeddings (np.ndarray): Unit vectors to search, shape (N, D).
        query_vector (np.ndarray): Unit query vector, shape (D,).
        k (int): Ignored. Present so all methods share one signature.

    Returns:
        The cosine similarity of every vector to the query, shape (N,).
    """
    return embeddings @ query_vector


def time_ms(func: Callable[[object], object], inputs: list) -> np.ndarray:
    """Times one call of func per input, after a few untimed warm-up calls.

    Warm-up calls let one-off costs, such as first-touch memory allocation and
    CPU frequency scaling, settle before anything is measured.

    Args:
        func (Callable): The function to time. It takes one argument.
        inputs (list): The argument for each timed call.

    Returns:
        The duration of each timed call in milliseconds.
    """
    for item in inputs[:WARMUP_CALLS]:
        func(item)

    durations = np.empty(len(inputs))
    for i, item in enumerate(inputs):
        start = time.perf_counter_ns()
        func(item)
        durations[i] = (time.perf_counter_ns() - start) / 1e6
    return durations


def peak_memory_mb(func: Callable[[object], object], arg: object) -> float:
    """Measures the most memory allocated at once during one call of func.

    Args:
        func (Callable): The function to measure. It takes one argument.
        arg (object): The argument to call it with.

    Returns:
        The peak of newly allocated memory during the call, in megabytes.
    """
    tracemalloc.start()
    try:
        func(arg)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1e6


def percentile_cells(durations_ms: np.ndarray) -> str:
    """Formats the p50, p95, and p99 of some durations as table cells."""
    p50, p95, p99 = np.percentile(durations_ms, [50, 95, 99])
    return f"{p50:.2f} | {p95:.2f} | {p99:.2f}"


def benchmark_size(rng: np.random.Generator, n: int, repeats: int) -> list[str]:
    """Benchmarks every search method on an index of n random vectors.

    Args:
        rng (np.random.Generator): Seeded generator for vectors and queries.
        n (int): Number of vectors in the index.
        repeats (int): Number of timed queries per method.

    Returns:
        One markdown table row per method.
    """
    embeddings = random_unit_vectors(rng, n)
    queries = list(random_unit_vectors(rng, repeats))
    index_mb = embeddings.nbytes / 1e6

    rows = []
    for name, method in (
        ("scoring only, no top-k", scores_only),
        ("argpartition (top_k_cosine)", top_k_cosine),
        ("full argsort", full_sort_top_k),
    ):
        search = partial(method, embeddings, k=TOP_K)
        durations = time_ms(search, queries)
        extra_mb = peak_memory_mb(search, queries[0])
        rows.append(
            f"| {n:,} | {index_mb:,.0f} | {name} | "
            f"{percentile_cells(durations)} | {extra_mb:.1f} |"
        )
    return rows


def main() -> None:
    """Runs every benchmark and prints the results as markdown tables."""
    parser = argparse.ArgumentParser(description="Benchmark exact vector search.")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"{platform.system()} {platform.release()}, "
        f"{os.cpu_count()} logical CPUs"
    )
    print(f"Dimension {DIMENSION}, top {TOP_K}, {args.repeats} timed queries\n")

    index = VectorIndex()
    texts = SAMPLE_QUERIES * (args.repeats // len(SAMPLE_QUERIES) + 1)
    encode = partial(index.model.encode, show_progress_bar=False)
    encode_ms = time_ms(encode, texts[: args.repeats])

    print("Query encoding (all-MiniLM-L6-v2 on CPU)\n")
    print("| p50 (ms) | p95 (ms) | p99 (ms) |")
    print("|---:|---:|---:|")
    print(f"| {percentile_cells(encode_ms)} |\n")

    print(f"Vector search (top {TOP_K})\n")
    print(
        "| Vectors | Index (MB) | Method | p50 (ms) | p95 (ms) | p99 (ms) "
        "| Peak extra memory (MB) |"
    )
    print("|---:|---:|---|---:|---:|---:|---:|")
    rng = np.random.default_rng(args.seed)
    for n in args.sizes:
        for row in benchmark_size(rng, n, args.repeats):
            print(row)


if __name__ == "__main__":
    main()
