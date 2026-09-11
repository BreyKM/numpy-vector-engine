"""Vector search engine core index module.

Provides in-memory vector storage and embedding ingestion using NumPy
and SentenceTransformers."""

import numpy as np
from sentence_transformers import SentenceTransformer


class VectorIndex:
    """An in-memory index for storing and managing high-dimensional text embeddings.

    Attributes:
        model (SentenceTransformer): The embedding model used for encoding text.
        documents (List[str]): A list of raw text documents added to the index.
        embeddings (Optional[np.ndarray]): A 2D NumPy array of shape (N, D) storing vector
            representations, where N is the number of documents and D is the embedding dimension.
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
            ValueError: If the input list is empty or invalid.
        """
        if isinstance(docs, str):
            raise TypeError("docs must be a list of strings, not a single string.")
        if not docs:
            raise ValueError("Cannot add an empty list of documents to the index.")

        print(f"Encoding {len(docs)} documents...")
        new_embeddings = self.model.encode(docs, show_progress_bar=False)

        self.documents.extend(docs)

        if self.embeddings is None:
            self.embeddings = np.array(new_embeddings, dtype=np.float32)
        else:
            self.embeddings = np.vstack((self.embeddings, new_embeddings))
        print("Index updated")
