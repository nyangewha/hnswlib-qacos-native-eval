# QA-Cos Native HNSW Evaluation

This repository contains a **controlled scorer-replacement experiment inside the official [`nmslib/hnswlib`](https://github.com/nmslib/hnswlib) codebase**, together with compact wall-clock artifacts for a small set of practical follow-up experiments.

## Scope

This repository is still centered on a narrow methodological question:

- native HNSW graph/index construction is unchanged;
- native HNSW search mechanics are unchanged unless explicitly marked as a gated practical variant;
- the main variable is the **coarse search-time scorer** or a **decoder refinement inside the coarse stage**.

This is **not** a new ANN system and **not** a redesign of HNSW.

## What Remains Unchanged

The following native HNSW components are unchanged:

- graph/index construction
- insertion
- pruning / neighbor selection
- level assignment
- `M`, `efConstruction`, `efSearch`
- stopping rule
- final native `top_candidates` behavior

## What Changes

### A. Native scorer-replacement experiment

Compared scorers:

- `simhash_baseline`: Hamming/agreement-based cosine proxy
- `qacos`: query-aware QA-Cos using the same stored document sign sketches plus the full real-valued query-side projections

A narrow **logging-only extension** is also included to expose the bottom-layer distinct nodes whose experimental scorer distance was actually computed. This is used only for offline candidate-efficiency evaluation and does not change native HNSW search semantics.

### B. Practical wall-clock follow-up experiments

Two compact follow-up variants are also included:

- **gated QA-Cos**
  - a practical hybrid variant that applies QA-Cos only to an ambiguity band in same-count space and falls back to the SimHash baseline outside that band
- **storage-aware rerank timing**
  - native HNSW search remains in memory, while exact full-vector reranking is measured with file-backed normalized full-vector stores under:
    - `warm_cache`
    - `cache_limited`

These follow-up variants are included as practical engineering experiments and should be interpreted separately from the strict scorer-only replacement setting.

## Main Files

- `hnswlib/hnswalg.h`
  - native experimental scorer path
  - bottom-layer visited-candidate logging
  - gated QA-Cos wall-clock path
- `python_bindings/bindings.cpp`
  - Python binding exposure for experimental outputs
- `run_native_hnsw_qacos_eval.py`
  - evaluation runner for the native scorer-replacement experiment
- `run_native_hnsw_storage_aware_eval.py`
  - storage-aware two-stage rerank runner

## Reproducing From These Files

This repository is intentionally minimal. To reproduce the experiments, start from the official
[`nmslib/hnswlib`](https://github.com/nmslib/hnswlib) source tree and then replace only the patched files plus the runners provided here.

### Step 1. Get the official `hnswlib` source

```bash
git clone https://github.com/nmslib/hnswlib.git
cd hnswlib
```

### Step 2. Copy the patched files from this repository

Copy these files into the matching locations inside the official `hnswlib` checkout:

- `hnswlib/hnswalg.h` -> `./hnswlib/hnswalg.h`
- `python_bindings/bindings.cpp` -> `./python_bindings/bindings.cpp`
- `run_native_hnsw_qacos_eval.py` -> `./run_native_hnsw_qacos_eval.py`
- `run_native_hnsw_storage_aware_eval.py` -> `./run_native_hnsw_storage_aware_eval.py`

### Step 3. Build the patched Python package

```bash
python -m pip install -e .
```

### Step 4. Prepare embedding caches

The runners expect cached float embeddings for the datasets under test:

- ArguAna
- NFCorpus
- SciFact
- FiQA

They use the same cache format as our earlier experiments:

- L2-normalized 768-d MPNet embeddings
- document embeddings
- query embeddings

Pass the cache root with `--cache_dir`.

## Native scorer-replacement settings

- Datasets: ArguAna, FiQA, NFCorpus, SciFact
- Native backend: official `hnswlib`
- Embeddings: L2-normalized 768-d MPNet float embeddings
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

## Practical wall-clock follow-up settings

### A. Gated practical wall-clock setting

- Datasets: NFCorpus, FiQA
- Bits: `128`
- `M in {16,32}`
- Query cap: `80`
- Threads: `1`
- QA-Cos iterations: `T=1`
- `mills_approx = True`
- gate band: `same in [72,96]`
- the practical question is whether a slower but more accurate coarse decoder can still help at the system level once downstream full-vector accesses become more costly

### B. Storage-aware rerank timing

- native HNSW search remains unchanged and in memory
- exact rerank uses file-backed normalized full-vector stores
- the goal is not to estimate physical SSD cache-miss rates directly, but to move beyond a trivially warm fully in-memory rerank setting and make downstream full-vector access cost more visible
- two storage modes are reported:
  - `warm_cache`
  - `cache_limited`
- `cache_limited` uses a larger replicated file-backed store plus cache pressure before timed runs

Important caveat:

- this setup is intended to be **more storage-aware than a fully in-memory rerank setting**
- it does **not** directly measure physical SSD cache-miss rate

### C. FiQA `Recall@100` follow-up

To stress large downstream rerank burden, we also include a focused `FiQA, 128-bit, M=32` matched experiment for:

- `Recall@100 >= 0.8`
- `Recall@100 >= 0.9`

## Included Result Files

Only compact validation artifacts are kept in this repository.

### 1. Native scorer-replacement results

Final frontier quality:

- `native_hnsw_full_t2/native_hnsw_summary.csv`
- `native_hnsw_full_t2/native_hnsw_metadata.json`

Candidate-count efficiency:

- `native_hnsw_candidate_full_t2/native_hnsw_candidate_efficiency_summary.csv`
- `native_hnsw_candidate_full_t2/native_hnsw_metadata.json`

Combined table figures:

- `native_hnsw_tables_4dataset/table_native_hnsw_frontier_allbits.png`
- `native_hnsw_tables_4dataset/table_native_hnsw_candidate_efficiency_allbits.png`
- `native_hnsw_tables_4dataset/table_native_hnsw_frontier_allbits.csv`
- `native_hnsw_tables_4dataset/table_native_hnsw_candidate_efficiency_allbits.csv`

### 2. Practical wall-clock artifacts

All compact follow-up wall-clock files are stored in:

- `native_hnsw_wallclock_storage_aware/`

Included files:

- `storage_aware_summary_r10.csv`
  - storage-aware matched `Recall@10` summary for `warm_cache` and `cache_limited`
- `storage_aware_summary_r100_fiqa_m32.csv`
  - storage-aware FiQA `Recall@100` summary

Matched operating points such as `efSearch=1500` vs `700` in the FiQA `Recall@100` follow-up are reported directly in the rebuttal text; the release keeps only the compact storage-aware follow-up summaries.
The release keeps only the compact storage-aware follow-up summaries. Operating-point choices discussed in the rebuttal are stated directly in the text rather than exposed here as separate search-sweep artifacts.

Raw per-query CSVs, smoke-test outputs, and larger intermediate files are intentionally omitted to keep the repository lightweight.

## Metric Families

### 1. Final `top_candidates` frontier metric

This metric uses the final native HNSW `top_candidates` result and measures frontier quality after exact float-cosine reranking.

### 2. Visited candidate pool prefix metric

This metric uses the **bottom-layer visited-and-scored distinct node pool**. Candidates are sorted by the same scorer, and we measure how many coarse candidates must be exact-reranked to reach a target recall.

### 3. Storage-aware two-stage wall-clock metric

This metric asks whether a slower but more accurate coarse decoder can still improve overall query cost in a two-stage retrieval setting once downstream exact full-vector accesses become more expensive.

For the storage-aware setting, the main total is:

- native HNSW search time
- plus exact rerank time over file-backed full-vector reads

## Headline Takeaways

- In the strict native scorer-replacement setting, QA-Cos improves frontier quality and candidate efficiency while leaving native HNSW construction unchanged.
- In the practical gated wall-clock setting, a `same-count` band gate can materially reduce the cost of QA-Cos relative to full scorer replacement.
- In the storage-aware rerank setting, reducing exact full-vector rereads becomes substantially more valuable than in the fully in-memory case.
- In the focused FiQA `Recall@100` follow-up, gated QA-Cos reaches the target with much smaller `efSearch` and rerank burden; for `Recall@100 >= 0.9`, the included sweep reaches the target for gated QA-Cos while the matched SimHash baseline does not.
