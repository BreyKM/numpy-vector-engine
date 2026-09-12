# numpy Vector Engine ![Show Image](https://github.com/BreyKM/numpy-vector-engine/actions/workflows/ci.yml/badge.svg)

### A lightweight, in-memory vector search engine built from scratch in Python with NumPy.

Traditional search matches characters. Searching a product catalog for "something to keep my coffee warm" returns nothing if no listing contains those words, even when a vacuum flask is sitting right there in the data.

This engine matches meaning instead. It encodes text into 384-dimensional embeddings with a sentence-transformer model, then ranks documents by cosine similarity to the query. The similarity search, normalization, and top-k selection are implemented directly against NumPy rather than delegated to a vector database.

## Quickstart

Requires Python 3.10 or newer.
```bash
git clone https://github.com/BreyKM/numpy-vector-engine.git
cd numpy-vector-engine
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
The first run downloads the embedding model (~90 MB) and caches it locally.

## Example

main.py indexes a small product catalog and runs several queries. Note that none of the queries share vocabulary with the listing they retrieve:
```
Loading embedding model...
Loading model 'all-MiniLM-L6-v2'...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 7098.71it/s]
Encoding 5 documents...
Index updated
Indexed 5 documents.

Query: something to keep my coffee warm
 0.4902 Stainless steel vacuum flask keeps beverages hot for twelve hours.
 0.2125 Memory foam sneakers designed for nurses and retail workers.
 0.1159 Ultralight 13-inch notebook computer with all-day battery life.

Query: cheap phone with a good camera
 0.5707 Budget Android handset with a 50-megapixel main sensor.
 0.2920 Ultralight 13-inch notebook computer with all-day battery life.
 0.1222 Compact carry-on suitcase with four spinner wheels.

Query: who won the Battle of Hastings?
 0.0611 Compact carry-on suitcase with four spinner wheels.
 0.0072 Ultralight 13-inch notebook computer with all-day battery life.
 -0.0126 Budget Android handset with a 50-megapixel main sensor.
 ```

## Usage

```python
from src.engine import VectorIndex

index = VectorIndex()
index.add_documents([
    "Stainless steel vacuum flask keeps beverages hot for twelve hours.",
    "Memory foam sneakers designed for nurses and retail workers.",
])

for score, document in index.search("comfortable footwear for long shifts", top_k=2):
    print(f"{score:.4f}  {document}")
```
search returns (score, document) pairs sorted highest-first. Scores range from -1 to 1 and are meaningful relative to each other, not as absolute percentages.

## How it works

Cosine similarity measures the angle between two vectors, ignoring their magnitude:
$$
cosine(a, b) = \frac{a · b}{‖a‖ × ‖b‖}
$$
Embedding models encode meaning in a vector's direction, so magnitude should not influence the ranking. Since every stored vector is normalized to unit length at ingestion time, the denominator is always 1 and each similarity reduces to a plain dot product. One matrix multiplication then scores the entire index:

```python
embeddings (N × 384)  @  query (384,)  →  scores (N,)
```

## Design decisions

Normalize at ingestion, not at query time. The obvious implementation recomputes every document's L2 norm on every search. Normalizing once when a document is added removes N norm computations per query and reduces each similarity to a dot product. It also keeps results correct with embedding models that do not normalize their own output.

Partial sort for top-k. Fully sorting all N scores costs O(N log N) to keep only a handful of results. np.argpartition selects the top k in O(N) average time, and only those k are then sorted.

Atomic ingestion. Documents are encoded before the text is appended to the index. If encoding fails, the index is left untouched rather than holding text with no corresponding embedding row — a desynchronization that would silently return the wrong documents for every subsequent search.

float32 embeddings. Half the memory of float64 (~1.5 GB versus ~3 GB per million vectors) with no meaningful precision loss for similarity ranking.

Explicit input validation. add_documents rejects a bare string, which Python would otherwise iterate character by character, producing one "document" per letter while generating only a single embedding.

## Testing and CI

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers the normalization math against hand-computed values, ingestion and shape invariants, and edge cases including an empty index, top_k larger than the index, and blank queries. The central test compares the optimized search against a direct transcription of the cosine similarity formula, verifying that the normalization and partial-sort optimizations produce identical rankings and scores.

GitHub Actions runs Ruff (lint and format) and the full test suite on every push and pull request. Linting runs before the dependency install so style failures surface in seconds rather than minutes.

## Limitations and roadmap

The index is in-memory only, so documents are re-encoded on every run. Search is exact, comparing the query against every stored vector, which is O(N) per query and does not scale to very large collections. Documents cannot yet be updated or deleted, since removal would shift the row positions that keep text and embeddings aligned.

Planned next:

- [ ] Persistence — save and load the index from disk
- [ ] Benchmark suite with latency percentiles and memory profiling
- [ ] Approximate nearest neighbor index (IVF or HNSW), measured against this exact search for both recall and latency
- [ ] Memory-mapped vector storage for indexes larger than RAM
- [ ] FastAPI endpoint and Docker packaging

## Built with

[NumPy](https://numpy.org) · [sentence-transformers](https://www.sbert.net) · [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [pytest](https://docs.pytest.org/en/latest/) · [Ruff](https://docs.astral.sh/ruff/)

## License

MIT — see [LICENSE](./LICENSE).