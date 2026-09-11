import numpy as np
import pytest

from src.engine import VectorIndex


@pytest.fixture
def empty_index():
    return VectorIndex()


def test_initial_state(empty_index):
    assert len(empty_index.documents) == 0
    assert empty_index.embeddings is None


def test_add_documents_dimensions(empty_index):
    sample_docs = ["First sentence.", "Second sentence."]
    empty_index.add_documents(sample_docs)

    assert len(empty_index.documents) == 2
    assert isinstance(empty_index.embeddings, np.ndarray)
    assert empty_index.embeddings.shape == (2, 384)


def test_subsequent_ingestion_stacking(empty_index):
    empty_index.add_documents(["Batch 1 sentence."])
    empty_index.add_documents(["Batch 2 sentence.", "Batch 2 extra."])

    assert len(empty_index.documents) == 3
    assert empty_index.embeddings.shape == (3, 384)


def test_empty_ingestion_raises_error(empty_index):
    with pytest.raises(ValueError):
        empty_index.add_documents([])


def test_single_string_raises_error(empty_index):
    with pytest.raises(TypeError):
        empty_index.add_documents("not a list")
