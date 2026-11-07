# numpy Vector Engine 

[![CI](https://github.com/BreyKM/numpy-vector-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/BreyKM/numpy-vector-engine/actions/workflows/ci.yml)

A lightweight, in-memory vector search engine built from scratch in Python with NumPy.

Traditional search matches characters. Searching a product catalog for "something to keep my coffee warm" returns nothing if no listing contains those words, even when a vacuum flask is sitting right there in the data.

This engine matches meaning instead. It encodes text into 384-dimensional embeddings with a sentence-transformer model, then ranks documents by cosine similarity to the query. The similarity search, normalization, and top-k selection are implemented directly against NumPy rather than delegated to a vector database.

## Quickstart

Requires Python 3.12 or newer.
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
```text
Loading embedding model...
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
index.add_documents(
    [
        "Stainless steel vacuum flask keeps beverages hot for twelve hours.",
        "Memory foam sneakers designed for nurses and retail workers.",
    ]
)

for score, document in index.search("comfortable footwear for long shifts", top_k=2):
    print(f"{score:.4f}  {document}")
```
search returns (score, document) pairs sorted highest-first. Scores range from -1 to 1 and are meaningful relative to each other, not as absolute percentages.

### Saving and loading

```python
from src.engine import VectorIndex

index = VectorIndex()
index.add_documents(["Compact carry-on suitcase with four spinner wheels."])
index.save("data/my-index")

restored = VectorIndex.load("data/my-index")
print(restored.search("luggage for a short trip", top_k=1))
```

save writes the vectors to embeddings.npy and the documents and model name to index.json. load reads the model name back from the file, so a restored index always encodes queries with the model that produced its vectors.

## How it works

Cosine similarity measures the angle between two vectors, ignoring their magnitude:
$$
cosine(a, b) = \frac{a · b}{‖a‖ × ‖b‖}
$$
Embedding models encode meaning in a vector's direction, so magnitude should not influence the ranking. Since every stored vector is normalized to unit length at ingestion time, the denominator is always 1 and each similarity reduces to a plain dot product. One matrix multiplication then scores the entire index:

```
embeddings (N × 384)  @  query (384,)  →  scores (N,)
```

## Design decisions

Normalize at ingestion, not at query time. The obvious implementation recomputes every document's L2 norm on every search. Normalizing once when a document is added removes N norm computations per query and reduces each similarity to a dot product. It also keeps results correct with embedding models that do not normalize their own output.

Partial sort for top-k. Fully sorting all N scores costs O(N log N) to keep only a handful of results. np.argpartition selects the top k in O(N) average time, and only those k are then sorted.  On a million vectors this halves search time, from 51 ms to 28 ms (see Performance).

Atomic ingestion. Documents are encoded before the text is appended to the index. If encoding fails, the index is left untouched rather than holding text with no corresponding embedding row — a desynchronization that would silently return the wrong documents for every subsequent search.

float32 embeddings. Half the memory of float64 (~1.5 GB versus ~3 GB per million vectors) with no meaningful precision loss for similarity ranking.

Explicit input validation. add_documents rejects a bare string, which Python would otherwise iterate character by character, producing one "document" per letter while generating only a single embedding.

Crash-safe saves. save deletes index.json first and writes it last, so it acts as a completion marker: a save interrupted partway leaves no index.json, and load refuses the directory rather than pairing new vectors with old documents. load also treats the files as untrusted, checking the format version, the document count against the number of vector rows, and the stored dimension against the model before it builds the index.

IVF lists stored contiguously. IVFIndex.build sorts the vectors by list and records where each list starts, the same layout a CSR sparse matrix uses. A search scores each probed list as a slice of one array, which NumPy reads in place with no copying. A second array maps each sorted row back to its original position, so IVF returns the same ids as exact search.

Spherical k-means, trained on a sample. Each centroid is the average of its cluster, renormalized to unit length, so comparing a query to a centroid is still a single dot product. Starting centroids are chosen with k-means++. Training on a random sample of 100,000 vectors is much faster than training on all of them and gives nearly the same centroids. Every vector is then assigned to its nearest one.

## Performance

Measured on an Intel Core i7-12700KF (Windows 11, Python 3.14, NumPy 2.5 with OpenBLAS), returning the top 10 results. Each figure is the median (p50) of 200 queries. The index holds random unit vectors, since exact search does the same work whatever the vector values are.

| Vectors | Index size | Scoring only | argpartition (used) | Full argsort |
|---:|---:|---:|---:|---:|
| 10,000 | 15 MB | 0.18 ms | 0.21 ms | 0.38 ms |
| 100,000 | 154 MB | 2.57 ms | 2.99 ms | 4.25 ms |
| 1,000,000 | 1.5 GB | 27.6 ms | 28.2 ms | 51.0 ms |

Encoding the query with the model adds 4.8 ms to every search.

Search time grows linearly with the index. Each tenfold increase in vectors costs about ten times as much, as expected of a scan over every row.

The partial sort pays off. argpartition is 1.4 to 1.8 times faster than fully sorting the scores.

Memory bandwidth is the bottleneck. At a million vectors, scoring takes 27.6 ms and selecting the top 10 adds only 0.6 ms. Every query reads the entire 1.5 GB matrix, so the way to search faster is to read fewer vectors, which is what approximate indexes such as IVF and HNSW do.

Each search temporarily allocates about 16 bytes per stored vector: the scores, a negated copy for sorting, and the candidate indices. That is 16 MB at a million vectors.

Below roughly 150,000 vectors, encoding the query takes longer than the search itself.

To reproduce, including p95 and p99 latencies:

```bash
python -m benchmarks.bench_search
```

### Approximate search (IVF)

src/ivf.py adds an inverted file (IVF) index. k-means groups the vectors into lists of similar vectors, and a search scores only the vectors in the few lists whose centroids are closest to the query. Probing more lists (n_probe) finds more of the true neighbors but takes longer.

Measured on 1,000,000 real Wikipedia sentences encoded with all-MiniLM-L6-v2, using 1,000 held-out sentences as queries. Recall@10 is the share of the exact top 10 that IVF also returned. The index has 1,000 lists, with k-means trained on a random sample of 100,000 vectors, and took 12 s to build.

| n_probe | Vectors scored | Recall@10 | p50 latency | Speed-up |
|---:|---:|---:|---:|---:|
| exact | 100% | 1.000 | 28.3 ms | 1x |
| 1 | 0.1% | 0.452 | 0.13 ms | 225x |
| 2 | 0.2% | 0.595 | 0.20 ms | 140x |
| 4 | 0.5% | 0.709 | 0.37 ms | 77x |
| 8 | 0.9% | 0.804 | 0.70 ms | 40x |
| 16 | 1.8% | 0.876 | 1.18 ms | 24x |
| 32 | 3.5% | 0.929 | 2.28 ms | 12x |
| 64 | 6.7% | 0.963 | 4.48 ms | 6.3x |
| 128 | 13.1% | 0.981 | 9.05 ms | 3.1x |

Recall rises quickly, then flattens. Probing 32 of the 1,000 lists scores 3.5% of the vectors and finds 93% of the true top 10, twelve times faster than exact search. Beyond that, each doubling of n_probe buys fewer than four points of recall.

One list is not enough. With n_probe = 1, IVF misses more than half of the true neighbors, because many of a query's nearest neighbors fall just across the boundary in a neighboring list.

Encoding the query becomes the main cost. It takes 4.8 ms, so a full search at n_probe = 32 takes about 7 ms, against 33 ms with exact search.

IVF scores each vector less efficiently than exact search. At n_probe = 128 it scores 13% of the vectors but takes 32% of the exact search time: exact search is one large matrix-vector product, while IVF makes one small product per probed list inside a Python loop.

The costs: building the index takes 12 s, the index keeps its own copy of the vectors sorted by list (another 1.5 GB at a million vectors), and results are approximate.

These numbers need real embeddings. Random vectors have no cluster structure for k-means to find. The exact-search benchmark above can use them only because exact search does the same work whatever the values are.

To reproduce:

```bash
pip install -r requirements-bench.txt
python -m benchmarks.prepare_wikipedia
python -m benchmarks.bench_ivf
```

prepare_wikipedia downloads one file of the sentence-transformers/wikipedia-en-sentences dataset (340 MB) and encodes 1,001,000 sentences, which took 29 minutes on this CPU. The embeddings (1.5 GB) are saved to the git-ignored data/ folder.

## Testing and CI

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers the normalization math against hand-computed values, ingestion and shape invariants, and edge cases including an empty index, top_k larger than the index, and blank queries. The central test compares the optimized search against a direct transcription of the cosine similarity formula, verifying that the normalization and partial-sort optimizations produce identical rankings and scores. The IVF tests check that probing every list returns exactly the same results as exact search. They use random, unclustered vectors, so the guarantee does not depend on how good the clustering is.

GitHub Actions runs Ruff (lint and format) once, then the full test suite on Python 3.12, 3.13, and 3.14, on every push and pull request. The tests only start after linting passes, so style failures surface in seconds rather than after the multi-minute dependency install.

## Limitations and roadmap

The whole index is held in memory, including after it is loaded from disk. VectorIndex search is exact, comparing the query against every stored vector, which is O(N) per query (about 28 ms at a million vectors on a desktop CPU). The IVF index in src/ivf.py is far faster but standalone for now: VectorIndex does not use it yet, and it cannot be saved. Documents cannot yet be updated or deleted, since removal would shift the row positions that keep text and embeddings aligned.

Planned next:

- [x] Persistence — save and load the index from disk
- [x] Benchmark suite with latency percentiles and memory profiling
- [x] Approximate nearest neighbor index (IVF), measured against exact search for both recall and latency
- [ ] Search through IVF from VectorIndex, and save it with the index
- [ ] HNSW graph index, compared against IVF
- [ ] Memory-mapped vector storage for indexes larger than RAM
- [ ] FastAPI endpoint and Docker packaging

## Built with

[NumPy](https://numpy.org) · [sentence-transformers](https://www.sbert.net) · [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [pytest](https://docs.pytest.org/en/latest/) · [Ruff](https://docs.astral.sh/ruff/)

## License

MIT — see [LICENSE](./LICENSE).