"""Encodes Wikipedia sentences into embeddings for the IVF recall benchmark.

Recall is only meaningful on real data. Random vectors have no cluster
structure, so there is nothing for k-means to find. Real sentence embeddings
group by topic, and IVF depends on exactly that.

The script downloads one file of the sentence-transformers/wikipedia-en-sentences
dataset (about 340 MB, 3.9M sentences), draws a random sample from it, and
encodes the sample with the same model as VectorIndex. The queries are held out
of the corpus: a query that was also indexed would always find itself first,
which would inflate recall.

Encoding a million sentences on a CPU takes about 20 minutes. Install the extra
dependencies, then run from the project root:

    pip install -r requirements-bench.txt
    python -m benchmarks.prepare_wikipedia
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from src.engine import VectorIndex, l2_normalize

DATASET_REPO = "sentence-transformers/wikipedia-en-sentences"
PARQUET_FILE = "data/train-00000-of-00002.parquet"
OUTPUT_DIR = Path("data") / "wikipedia"
CORPUS_FILE = "corpus.npy"
QUERIES_FILE = "queries.npy"
DEFAULT_N_DOCS = 1_000_000
DEFAULT_N_QUERIES = 1_000


def sample_sentences(
    n_docs: int, n_queries: int, seed: int
) -> tuple[list[str], list[str]]:
    """Downloads the dataset file and draws two separate random samples from it.

    The file is cached by huggingface_hub, so it is only downloaded once.

    Args:
        n_docs (int): Number of corpus sentences, at least 1.
        n_queries (int): Number of query sentences, at least 1.
        seed (int): Seed for the sample, so runs are repeatable.

    Returns:
        A tuple (corpus, queries). No sentence row is in both.

    Raises:
        ValueError: If n_docs or n_queries is less than 1, or together they
            are more than the file holds.
    """
    if n_docs < 1 or n_queries < 1:
        raise ValueError(
            f"n_docs and n_queries must be at least 1, got {n_docs} and {n_queries}."
        )

    path = hf_hub_download(DATASET_REPO, PARQUET_FILE, repo_type="dataset")
    sentences = pq.read_table(path, columns=["sentence"]).column("sentence")

    n_total = n_docs + n_queries
    if n_total > len(sentences):
        raise ValueError(
            f"Asked for {n_total} sentences, but the file holds {len(sentences)}."
        )

    rng = np.random.default_rng(seed)
    rows = rng.choice(len(sentences), size=n_total, replace=False)
    sample = sentences.take(rows).to_pylist()
    return sample[n_queries:], sample[:n_queries]


def main() -> None:
    """Samples, encodes, and saves the corpus and query embeddings."""
    parser = argparse.ArgumentParser(description="Encode Wikipedia sentences.")
    parser.add_argument("--n-docs", type=int, default=DEFAULT_N_DOCS)
    parser.add_argument("--n-queries", type=int, default=DEFAULT_N_QUERIES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    corpus, queries = sample_sentences(args.n_docs, args.n_queries, args.seed)
    model = VectorIndex().model

    start = time.perf_counter()
    corpus_vectors = l2_normalize(model.encode(corpus, show_progress_bar=True))
    query_vectors = l2_normalize(model.encode(queries))
    minutes = (time.perf_counter() - start) / 60

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / QUERIES_FILE, query_vectors)
    np.save(OUTPUT_DIR / CORPUS_FILE, corpus_vectors)

    total = len(corpus) + len(queries)
    print(
        f"Encoded {total:,} sentences in {minutes:.1f} minutes "
        f"({total / (minutes * 60):,.0f} per second)."
    )
    print(
        f"Saved {corpus_vectors.shape} corpus and {query_vectors.shape} query "
        f"embeddings to {OUTPUT_DIR}."
    )


if __name__ == "__main__":
    main()
