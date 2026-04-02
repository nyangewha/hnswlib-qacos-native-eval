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

## Experimental Settings

- Datasets: ArguAna, NFCorpus, SciFact
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
- `native_hnsw_candidate_full_t2/native_hnsw_summary.csv`
- `native_hnsw_candidate_full_t2/native_hnsw_candidate_efficiency_summary.csv`
- `native_hnsw_candidate_full_t2/native_hnsw_metadata.json`

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
