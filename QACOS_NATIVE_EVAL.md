# QA-Cos Native HNSW Evaluation

This repository contains a **controlled scorer-replacement experiment inside the official [`nmslib/hnswlib`](https://github.com/nmslib/hnswlib) codebase**.

## Scope

This is **not** a new ANN system. The goal is to test whether replacing only the **search-time scorer** inside native hierarchical HNSW search improves:
- final frontier quality, and
- downstream reranking efficiency.

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

Only the **search-time scorer** is changed.

Compared scorers:
- `simhash_baseline`: Hamming/agreement-based cosine proxy
- `qacos`: query-aware QA-Cos using the same stored document sign sketches plus the full real-valued query-side projections

A narrow **logging-only extension** is also included to expose the bottom-layer distinct nodes whose experimental scorer distance was actually computed. This is used only for offline candidate-efficiency evaluation and does not change native HNSW search semantics.

## Main Files

- `hnswlib/hnswalg.h`
  - native experimental scorer path
  - bottom-layer visited-candidate logging
- `python_bindings/bindings.cpp`
  - Python binding exposure for experimental outputs
- `run_native_hnsw_qacos_eval.py`
  - evaluation runner for the native experiments

## Reproducing From These Files

This repository is intentionally minimal. To reproduce the experiment, start from the official
[`nmslib/hnswlib`](https://github.com/nmslib/hnswlib) source tree and then replace only the two
patched source files plus the runner provided here.

### Step 1. Get the official `hnswlib` source

Clone the official repository:

```bash
git clone https://github.com/nmslib/hnswlib.git
cd hnswlib
```

### Step 2. Copy the patched files from this repository

Copy these files into the matching locations inside the official `hnswlib` checkout:

- `hnswlib/hnswalg.h` -> `./hnswlib/hnswalg.h`
- `python_bindings/bindings.cpp` -> `./python_bindings/bindings.cpp`
- `run_native_hnsw_qacos_eval.py` -> `./run_native_hnsw_qacos_eval.py`

No other source files need to be modified for this experiment.

### Step 3. Build the patched Python package

From the root of the patched `hnswlib` checkout:

```bash
python -m pip install -e .
```

This rebuilds the Python extension with the experimental scorer path enabled.

### Step 4. Prepare embedding caches

The runner expects cached float embeddings for:
- ArguAna
- NFCorpus
- SciFact
- FiQA

It uses the same cache format as our earlier rebuttal experiments:
- L2-normalized 768-d MPNet embeddings
- document embeddings
- query embeddings

Pass the cache root with `--cache_dir`.

### Step 5. Run the native evaluation

Example command:

```bash
python run_native_hnsw_qacos_eval.py \
  --cache_dir /path/to/emb_cache \
  --out_dir ./native_hnsw_candidate_full_t2 \
  --datasets arguana,fiqa,nfcorpus,scifact \
  --graph_M_values 16,32 \
  --sketch_bits 64,128 \
  --ef_search_values 20,50 \
  --max_queries 80 \
  --seed 0 \
  --qacos_iters 2 \
  --num_threads 1
```

The same runner also produces the final-frontier metrics used in the native HNSW evaluation.

### Step 6. What the runner writes

The runner writes:
- summary CSVs for the final native `top_candidates` frontier metrics
- summary CSVs for visited-candidate-pool prefix efficiency metrics
- metadata JSON describing the exact run settings

### Notes

- This experiment does **not** modify graph/index construction.
- It does **not** modify insertion, pruning, level assignment, `efSearch`, or the stopping rule.
- The only intended methodological change is the **search-time scorer**.
- The candidate-efficiency metrics come from a **logging-only extension** that exposes the
  bottom-layer distinct nodes whose scorer distance was actually computed.

## Experimental Settings

- Datasets: ArguAna, FiQA, NFCorpus, SciFact
- Native backend: official `hnswlib`
- Embeddings: L2-normalized 768-d MPNet float embeddings
- Search settings: `M in {16,32}`, `efSearch in {20,50}`
- Sketch bits: `64`, `128`
- QA-Cos iterations: `T=2`
- Query cap: `80` per dataset
- Seed: `0`

## Included Result Files

Only compact validation artifacts are kept in this repository.

### Final Frontier Quality
- `native_hnsw_full_t2/native_hnsw_summary.csv`
- `native_hnsw_full_t2/native_hnsw_metadata.json`

### Candidate-Count Efficiency
- `native_hnsw_candidate_full_t2/native_hnsw_candidate_efficiency_summary.csv`
- `native_hnsw_candidate_full_t2/native_hnsw_metadata.json`

### Combined Table Figures
- `native_hnsw_tables_4dataset/table_native_hnsw_frontier_allbits.png`
- `native_hnsw_tables_4dataset/table_native_hnsw_candidate_efficiency_allbits.png`
- `native_hnsw_tables_4dataset/table_native_hnsw_frontier_allbits.csv`
- `native_hnsw_tables_4dataset/table_native_hnsw_candidate_efficiency_allbits.csv`

Raw per-query CSVs and smoke-test outputs are intentionally omitted to keep the repository lightweight.

## Metric Families

### 1. Final `top_candidates` Frontier Metric
This metric uses the final native HNSW `top_candidates` result and measures frontier quality after exact float-cosine reranking.

### 2. Visited Candidate Pool Prefix Metric
This metric uses the **bottom-layer visited-and-scored distinct node pool**. Candidates are sorted by the same scorer, and we measure how many coarse candidates must be exact-reranked to reach a target recall.

These two metric families are complementary:
- the first evaluates the quality of the final native HNSW frontier,
- the second evaluates reranking efficiency on the candidate pool surfaced during native search.

## Headline Result

Across both metric families, QA-Cos improves native HNSW search quality under matched settings while keeping the underlying HNSW build and search mechanics unchanged.
