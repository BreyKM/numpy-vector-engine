"""Vector search engine core index module.

Provides in-memory vector storage and embedding ingestion using NumPy
and SentenceTransformers."""

import numpy as np
from sentence_transformers import SentenceTransformer


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scales vectors to unit length (an L2 norm of 1).

    Works on a single vector of shape (D,) or a matrix of shape (N, D).

    Args:
        vectors (np.ndarray): The vector or matrix of row vectors to normalize.

    Returns:
        An array of the same shape where every vector has length 1.
        All-zero vectors are returned unchanged instead of being normalized to NaN.
    """

    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)

    return vectors / np.maximum(norms, 1e-12)


class VectorIndex:
    """An in-memory index for storing and managing high-dimensional text embeddings.

    Attributes:
        model (SentenceTransformer): The embedding model used for encoding text.
        documents (List[str]): A list of raw text documents added to the index.
        embeddings (Optional[np.ndarray]): A 2D NumPy array of shape (N, D)
            storing vector representations, where N is the number of documents
            and D is the embedding dimension.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initializes the vector index and loads the transformer weights into memory.

        Args:
            model_name (str): HuggingFace model identifier for the embedding model.
            Defaults to 'all-MiniLM-L6-v2'.
        """

        print(f"Loading model '{model_name}'...")

        self.model = SentenceTransformer(model_name)

        self.documents: list[str] = []

        self.embeddings: np.ndarray | None = None

    def add_documents(self, docs: list[str]) -> None:
        """encodes raw text and appends their embeddings to the index.

        Args:
            docs (list[str]): A list of strings to encode and index.

        Raises:
            TypeError: If a single string is passed instead of a list.
            ValueError: If the input list is empty or invalid.
        """
        if isinstance(docs, str):
            raise TypeError("docs must be a list of strings, not a single string.")
        if not docs:
            raise ValueError("Cannot add an empty list of documents to the index.")

        print(f"Encoding {len(docs)} documents...")
        new_embeddings = self.model.encode(docs, show_progress_bar=False)
        new_embeddings = l2_normalize(new_embeddings.astype(np.float32))

        self.documents.extend(docs)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, new_embeddings))
        print("Index updated")

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        """Finds the documents most similar to the query using cosine similarity.

        This is an exact (brute-force) search: the query is compared against
        every stored vector, so it costs 0(N * D) per query.

        Args:
            query (str): The text to search for.
            top_k (int): The maximum number of results to return. Defaults to 5.

        Returns:
            Up to top_k (score, document) tuples, highest score first.
            Scores range from -1 to 1, where higher means more similar.

        Raises:
            TypeError: If query is not a string.
            ValueError: If the index is empty, the query is empty, or
                top_k is less than 1.
        """

        if self.embeddings is None:
            raise ValueError("Index is empty. Add documents before searching,")
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self.model.encode(query, show_progress_bar=False)
        query_vector = l2_normalize(query_vector.astype(np.float32))

        scores = self.embeddings @ query_vector

        k = min(top_k, len(self.documents))
        top_unsorted = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_unsorted[np.argsort(-scores[top_unsorted])]

        return [(float(scores[i]), self.documents[i]) for i in top_indices]
