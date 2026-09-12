"""Demo script showing semantic search over a small catalog

Each query is matched by meaning rather than keywords: none of the
queries share vocabulary with the document they retrieve.
"""

from src.engine import VectorIndex

SAMPLE_DOCS = [
    "Stainless steel vacuum flask keeps beverages hot for twelve hours.",
    "Ultralight 13-inch notebook computer with all-day battery life.",
    "Memory foam sneakers designed for nurses and retail workers.",
    "Budget Android handset with a 50-megapixel main sensor.",
    "Compact carry-on suitcase with four spinner wheels.",
]

queries = [
    "something to keep my coffee warm",
    "a portable computer for school",
    "comfortable footwear for long shifts",
    "cheap phone with a good camera",
    "luggage for a weekend trip",
    "who won the Battle of Hastings?",
]

TOP_K = 3


def build_index():
    """Creates an index and loads the sample docs into it."""

    print("Loading embedding model...")
    index = VectorIndex()
    index.add_documents(SAMPLE_DOCS)
    print(f"Indexed {len(SAMPLE_DOCS)} documents.\n")
    return index


def run_query(index: VectorIndex, query: str) -> None:
    """Searches the index and prints the top results with their scores."""
    print(f"Query: {query}")
    for score, doc in index.search(query, top_k=TOP_K):
        print(f" {score:.4f} {doc}")

    print()


def main():
    index = build_index()
    for query in queries:
        run_query(index, query)


if __name__ == "__main__":
    main()
