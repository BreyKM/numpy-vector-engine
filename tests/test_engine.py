"""Tests for VectorIndex: ingestion, search, and saving and loading."""

import json

import numpy as np
import pytest

from src.engine import VectorIndex, l2_normalize


@pytest.fixture
def empty_index():
    return VectorIndex()


def test_initial_state(empty_index):
    assert len(empty_index.documents) == 0
    assert empty_index.embeddings is None


def test_default_model_name_is_stored(empty_index):
    assert empty_index.model_name == "all-MiniLM-L6-v2"


def test_explicit_model_name_is_stored():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    index = VectorIndex(model_name=model_name)
    assert index.model_name == model_name


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


SAMPLE_DOCS = [
    "I love programming in Python.",
    "The weather is sunny and warm today.",
    "This recipe uses garlic, butter, and fresh basil.",
    "Machine learning models need a lot of training data.",
]


@pytest.fixture(scope="module")
def populated_index():
    """One shared index for read-only search tests (loads the model only once)."""
    index = VectorIndex()
    index.add_documents(SAMPLE_DOCS)
    return index


def test_l2_normalize_produces_unit_vectors():
    vectors = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    normalized = l2_normalize(vectors)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    assert np.allclose(normalized[0], [0.6, 0.8])


def test_l2_normalize_leaves_zero_vectors_as_zero():
    normalized = l2_normalize(np.zeros((1, 3), dtype=np.float32))
    assert not np.isnan(normalized).any()
    assert np.allclose(normalized, 0.0)


def test_stored_embeddings_are_normalized(populated_index):
    norms = np.linalg.norm(populated_index.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_search_finds_semantically_closest_document(populated_index):
    results = populated_index.search("What language should I use to write code?")
    assert results[0][1] == "I love programming in Python."


def test_search_results_sorted_by_score(populated_index):
    results = populated_index.search("food", top_k=4)
    scores = [score for score, _ in results]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(score, float) for score in scores)


def test_search_matches_brute_force_cosine(populated_index):
    query = "artificial intelligence"
    raw_docs = populated_index.model.encode(SAMPLE_DOCS)
    raw_query = populated_index.model.encode(query)
    expected = (raw_docs @ raw_query) / (
        np.linalg.norm(raw_docs, axis=1) * np.linalg.norm(raw_query)
    )
    expected_order = [SAMPLE_DOCS[i] for i in np.argsort(-expected)]

    results = populated_index.search(query, top_k=len(SAMPLE_DOCS))
    assert [text for _, text in results] == expected_order
    assert np.allclose([score for score, _ in results], np.sort(expected)[::-1])


def test_top_k_larger_than_index_returns_all(populated_index):
    results = populated_index.search("anything", top_k=100)
    assert len(results) == len(SAMPLE_DOCS)


def test_search_on_empty_index_raises(empty_index):
    with pytest.raises(ValueError):
        empty_index.search("hello")


@pytest.mark.parametrize("bad_top_k", [0, -1])
def test_invalid_top_k_raises(populated_index, bad_top_k):
    with pytest.raises(ValueError):
        populated_index.search("hello", top_k=bad_top_k)


def test_blank_query_raises(populated_index):
    with pytest.raises(ValueError):
        populated_index.search(" ")


def test_save_empty_index_raises(empty_index, tmp_path):
    with pytest.raises(ValueError):
        empty_index.save(tmp_path / "index")


def test_save_writes_embeddings_and_metadata(populated_index, tmp_path):
    save_dir = tmp_path / "index"
    populated_index.save(save_dir)

    embeddings = np.load(save_dir / "embeddings.npy")
    assert np.array_equal(embeddings, populated_index.embeddings)

    metadata = json.loads((save_dir / "index.json").read_text(encoding="utf-8"))
    assert metadata["format_version"] == 1
    assert metadata["model_name"] == populated_index.model_name
    assert metadata["dimension"] == 384
    assert metadata["documents"] == populated_index.documents


def test_save_keeps_non_ascii_text_readable(empty_index, tmp_path):
    empty_index.add_documents(["Café au lait", "日本語のテキスト"])
    empty_index.save(tmp_path)

    raw = (tmp_path / "index.json").read_text(encoding="utf-8")
    assert "Café au lait" in raw
    assert "日本語のテキスト" in raw


@pytest.fixture
def saved_index_dir(populated_index, tmp_path):
    save_dir = tmp_path / "index"
    populated_index.save(save_dir)
    return save_dir


def rewrite_metadata(save_dir, **changes):
    """Overwrites fields in a saved index.json to simulate a corrupt file."""
    metadata_path = save_dir / "index.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(changes)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_load_round_trip(populated_index, saved_index_dir):
    loaded = VectorIndex.load(saved_index_dir)

    assert loaded.model_name == populated_index.model_name
    assert loaded.documents == populated_index.documents
    assert np.array_equal(loaded.embeddings, populated_index.embeddings)
    assert loaded.embeddings.dtype == np.float32

    query = "What language should I use to write code?"
    assert loaded.search(query) == populated_index.search(query)


@pytest.mark.parametrize("filename", ["embeddings.npy", "index.json"])
def test_load_missing_file_raises(saved_index_dir, filename):
    (saved_index_dir / filename).unlink()
    with pytest.raises(FileNotFoundError):
        VectorIndex.load(saved_index_dir)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"format_version": 999}, "format version"),
        ({"documents": "not a list"}, "must be a list"),
        ({"documents": SAMPLE_DOCS[:-1]}, "rows"),
        ({"dimension": 512}, "columns"),
    ],
)
def test_load_corrupt_metadata_raises(saved_index_dir, changes, message):
    rewrite_metadata(saved_index_dir, **changes)
    with pytest.raises(ValueError, match=message):
        VectorIndex.load(saved_index_dir)


def test_load_model_dimension_mismatch_raises(saved_index_dir):
    wrong_size = np.ones((len(SAMPLE_DOCS), 10), dtype=np.float32)
    np.save(saved_index_dir / "embeddings.npy", wrong_size)
    rewrite_metadata(saved_index_dir, dimension=10)
    with pytest.raises(ValueError, match="produces"):
        VectorIndex.load(saved_index_dir)
