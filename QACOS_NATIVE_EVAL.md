# QA-Cos Native HNSW Evaluation

This release contains two distinct groups of artifacts:

1. **Reviewer 8DVT: native scorer-replacement evidence inside official `hnswlib`**
2. **Reviewer 9crG: storage-aware end-to-end wall-clock follow-up**

The first is the main controlled experiment: native HNSW construction and search mechanics are unchanged, and only the search-time scorer is changed. The second is a narrower practical follow-up for a two-stage retrieval setting.

This release is **not** a new ANN system and **not** a redesign of HNSW.

## Patched Files

These files are the code patches used across the experiments:

- `hnswlib/hnswalg.h`
- `python_bindings/bindings.cpp`
- `run_native_hnsw_qacos_eval.py`
- `run_native_hnsw_storage_aware_eval.py`

To reproduce from source, start from the official [`nmslib/hnswlib`](https://github.com/nmslib/hnswlib) repository, copy these patched files into the matching locations, and build with:

```bash
python -m pip install -e .
```

The runners expect cached L2-normalized 768-d MPNet document/query embeddings under `--cache_dir`.

---

# Section A. Reviewer 8DVT: Native scorer-replacement inside official `hnswlib`

## Question Answered

This section addresses the request for evaluation inside an **actual native HNSW implementation**.

Runner used for this section: `run_native_hnsw_qacos_eval.py`

## Experimental Scope

- official native backend: `hnswlib`
- native HNSW graph/index construction unchanged
- insertion, pruning, neighbor selection, level assignment, `M`, `efConstruction`, `efSearch`, stopping rule, and native `top_candidates` behavior unchanged
- only the **native search-time scorer** is changed

Compared scorers:

- `simhash_baseline`: Hamming/agreement-based cosine proxy
- `qacos`: QA-Cos using the same stored document sign sketches plus full real-valued query-side projections

A narrow **logging-only extension** is also included to expose the bottom-layer distinct nodes whose scorer distance was computed. This is used only for offline candidate-efficiency evaluation and does not change native HNSW search semantics.

## Settings

- Datasets: ArguAna, NFCorpus, SciFact, FiQA
- Embeddings: original L2-normalized 768-d MPNet float embeddings
- Search settings: `M in {16,32}`, `efConstruction=200`, `efSearch=50`
- Sketch bits: `64`, `128`
- QA-Cos iterations: `T=2`
- Query cap: `80` per dataset
- Seed: `0`

Native HNSW layer structures (top-to-bottom node counts):

- ArguAna: `M=16 [5,44,554,8674]`, `M=32 [2,17,275,8674]`
- FiQA: `M=16 [2,17,248,3617,57638]`, `M=32 [5,68,1806,57638]`
- NFCorpus: `M=16 [2,17,239,3633]`, `M=32 [5,115,3633]`
- SciFact: `M=16 [3,29,346,5183]`, `M=32 [1,10,176,5183]`

## Headline Results

Averaged over matched settings across the four datasets:

- **64-bit frontier quality**
  - `Recall@10`: `0.557 -> 0.692`
  - `Recall@100`: `0.258 -> 0.335`
  - visited nodes: `1535.5 -> 1340.2`
- **128-bit frontier quality**
  - `Recall@10`: `0.770 -> 0.853`
  - `Recall@100`: `0.389 -> 0.469`
  - visited nodes: `1364.5 -> 1228.5`
- **128-bit candidate efficiency**
  - minimum exact-rerank count for `Recall@10 >= 0.9`: `177.0 -> 97.3`
  - minimum exact-rerank count for `Recall@100 >= 0.8`: `479.7 -> 352.2`

Interpretation:

- with native HNSW construction fixed, replacing only the search-time scorer improves the final returned frontier on average
- QA-Cos also reaches the same target recall with fewer exact-reranked candidates

## Files for Reviewer 8DVT

Main summaries:

- `native_hnsw_full_t2/native_hnsw_summary.csv`
- `native_hnsw_full_t2/native_hnsw_metadata.json`
- `native_hnsw_candidate_full_t2/native_hnsw_candidate_efficiency_summary.csv`
- `native_hnsw_candidate_full_t2/native_hnsw_metadata.json`

Dataset-wise tables:

- `native_hnsw_tables_4dataset/table_native_hnsw_frontier_allbits.png`
- `native_hnsw_tables_4dataset/table_native_hnsw_candidate_efficiency_allbits.png`
- `native_hnsw_tables_4dataset/table_native_hnsw_frontier_allbits.csv`
- `native_hnsw_tables_4dataset/table_native_hnsw_candidate_efficiency_allbits.csv`

Metric meanings:

- **final frontier metric**: quality after exact reranking of the final native `top_candidates` result
- **visited-pool prefix metric**: how many visited-and-scored coarse candidates must be exact-reranked to reach a target recall

---

# Section B. Reviewer 9crG: Storage-aware end-to-end wall-clock follow-up

## Question Answered

This section addresses the narrower question of whether a **slower but more accurate first-stage decoder** can still help end-to-end latency in a realistic two-stage setting once downstream full-vector accesses become more expensive.

Because QA-Cos is a **decoder-side refinement** rather than a standalone ANN method, this should be interpreted as a scope-aligned two-stage follow-up rather than as a standard ANN-benchmark-style full-system comparison.

Runner used for this section: `run_native_hnsw_storage_aware_eval.py`

## Practical Variant Used

This follow-up uses a practical gated variant:

- `qacos_gated`: apply QA-Cos only inside a same-count ambiguity band, and fall back to the SimHash baseline outside that band
- reported band: `same in [74,96]`
- QA-Cos iterations: `T=1`
- `mills_approx = True`: replace the exact Gaussian Mills-ratio evaluation used inside the QA-Cos Newton-step likelihood/derivative calculations with a standard piecewise rational/polynomial Mills-ratio approximation, reducing scorer overhead during search-time decoding at the cost of a possible small quality drop; the reported storage-aware wall-clock results reflect this practical speed/quality trade-off and still show a practical advantage over the SimHash baseline

## Storage-Aware Setup

- native HNSW search remains in memory
- exact rerank reads normalized full vectors from a file-backed store
- storage modes:
  - `warm_cache`
  - `cache_limited`
- `cache_limited` expands the file-backed store and applies cache pressure so that downstream full-vector access cost is more visible than in a warm in-memory rerank setup

For the focused FiQA `Recall@100` follow-up:

- dataset: `FiQA`
- bits: `128`
- `M=32`
- for FiQA, `cache_limited` expands the normalized full-vector store from about `177 MB` to about `1.24 GB` via `7` replicas
- cache pressure: `0.5 GB`

## Headline Results

Focused `FiQA, 128-bit, M=32` follow-up:

- **`Recall@100 >= 0.8`**
  - SimHash: `efSearch=1300`, cache-limited total `1.1753 ms`
  - gated QA-Cos: `efSearch=700`, cache-limited total `0.9973 ms`
  - visited nodes: `17634.0 -> 9846.0`
  - exact-reranked full vectors: `1300 -> 700`
- **`Recall@100 >= 0.9`**
  - SimHash: not reached in the reported sweep (best `0.8493`)
  - gated QA-Cos: reached `0.9061` at `efSearch=1500`, cache-limited total `1.6828 ms`

Interpretation:

- this follow-up is **not** intended to show that QA-Cos is a universal winner
- it is intended to show that realistic two-stage settings can exist in which a slower but more accurate first-stage scorer still yields lower end-to-end latency

## Files for Reviewer 9crG

Compact wall-clock artifacts are stored in:

- `native_hnsw_wallclock_storage_aware/`

Included files:

- `storage_aware_summary_r10.csv`
  - storage-aware `Recall@10` follow-up summary for `warm_cache` and `cache_limited`
- `storage_aware_summary_r100_fiqa_m32.csv`
  - focused FiQA `Recall@100` storage-aware summary
- `storage_aware_r100_fiqa_m32.png`
  - compact summary figure for the focused FiQA `Recall@100` follow-up

Raw per-query CSVs, smoke-test outputs, and larger intermediates are intentionally omitted to keep the release lightweight.

---

## Short Takeaway

- **Section A / Reviewer 8DVT:** in a strict native scorer-replacement experiment inside official `hnswlib`, QA-Cos improves frontier quality and candidate efficiency while leaving native HNSW construction unchanged.
- **Section B / Reviewer 9crG:** in a narrower storage-aware two-stage follow-up, a practical gated QA-Cos variant can reduce search effort and downstream reread burden enough to improve end-to-end latency in some settings.
