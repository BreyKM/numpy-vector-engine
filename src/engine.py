"""Vector search engine core index module.

Provides in-memory vector storage, embedding ingestion, exact cosine search,
and saving and loading indexes on disk, using NumPy and SentenceTransformers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

FORMAT_VERSION = 1
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "index.json"


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
        model_name (str): The embedding model identifier passed to __init__.
        model (SentenceTransformer): The embedding model used for encoding text.
        documents (list[str]): A list of raw text documents added to the index.
        embeddings (np.ndarray | None): A 2D NumPy array of shape (N, D)
            storing vector representations, where N is the number of documents
            and D is the embedding dimension.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initializes the vector index and loads the transformer weights into memory.

        Args:
            model_name (str): HuggingFace model identifier for the embedding model.
                Defaults to 'all-MiniLM-L6-v2'.
        """

        self.model_name: str = model_name

        self.model = SentenceTransformer(model_name)

        self.documents: list[str] = []

        self.embeddings: np.ndarray | None = None

    def add_documents(self, docs: list[str]) -> None:
        """Encodes documents and appends their embeddings to the index.

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

        new_embeddings = self.model.encode(docs, show_progress_bar=False)
        new_embeddings = l2_normalize(new_embeddings.astype(np.float32))

        self.documents.extend(docs)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, new_embeddings))

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        """Finds the documents most similar to the query using cosine similarity.

        This is an exact (brute-force) search: the query is compared against
        every stored vector, so it costs O(N * D) per query.

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
            raise ValueError("Index is empty. Add documents before searching.")
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

    def save(self, path: str | Path) -> None:
        """Writes the index to a directory so it can be reloaded later.

        The directory holds embeddings.npy with the vectors and index.json with
        the documents and metadata. index.json is removed first and written
        last, so a save that fails partway leaves no index.json and load
        refuses the directory instead of reading mismatched files.

        Args:
            path (str | Path): Directory to save into. Created if missing.
                An index already saved there is overwritten.

        Raises:
            ValueError: If the index is empty.
        """
        if self.embeddings is None:
            raise ValueError("Cannot save an empty index. Add documents first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        metadata_path = path / METADATA_FILE
        metadata_path.unlink(missing_ok=True)

        np.save(path / EMBEDDINGS_FILE, self.embeddings)

        metadata = {
            "format_version": FORMAT_VERSION,
            "model_name": self.model_name,
            "dimension": self.embeddings.shape[1],
            "documents": self.documents,
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> VectorIndex:
        """Loads an index previously written by save.

        The model name is read from index.json, so the loaded index encodes
        queries with the same model that produced the stored vectors. The files
        are validated before the model is loaded, and the index is only built
        once every check has passed.

        Args:
            path (str | Path): Directory that save wrote to.

        Returns:
            A new VectorIndex holding the saved documents and embeddings.

        Raises:
            FileNotFoundError: If embeddings.npy or index.json is missing.
            ValueError: If the files are corrupt, use an unsupported format
                version, or do not match each other or the model.
        """
        path = Path(path)
        embeddings_path = path / EMBEDDINGS_FILE
        metadata_path = path / METADATA_FILE

        for file in (embeddings_path, metadata_path):
            if not file.is_file():
                raise FileNotFoundError(f"Missing index file: {file}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        required_keys = {"format_version", "model_name", "dimension", "documents"}
        if not isinstance(metadata, dict) or not required_keys <= metadata.keys():
            raise ValueError(f"{metadata_path} is missing required fields.")
        if metadata["format_version"] != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format version {metadata['format_version']!r}, "
                f"expected {FORMAT_VERSION}."
            )

        documents = metadata["documents"]
        dimension = metadata["dimension"]
        if not isinstance(documents, list):
            raise ValueError("documents in index.json must be a list.")

        embeddings = np.load(embeddings_path, allow_pickle=False)
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}.")
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"index.json has {len(documents)} documents but embeddings.npy "
                f"has {embeddings.shape[0]} rows."
            )
        if dimension != embeddings.shape[1]:
            raise ValueError(
                f"index.json says dimension {dimension} but embeddings.npy "
                f"has {embeddings.shape[1]} columns."
            )

        index = cls(model_name=metadata["model_name"])

        model_dimension = index.model.get_embedding_dimension()
        if model_dimension != dimension:
            raise ValueError(
                f"Model {metadata['model_name']!r} produces {model_dimension}-dim "
                f"vectors but the saved embeddings are {dimension}-dim."
            )

        index.documents = documents
        index.embeddings = embeddings.astype(np.float32)
        return index
