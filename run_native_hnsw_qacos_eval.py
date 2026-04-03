#!/usr/bin/env python3
"""Controlled native-HNSW scorer replacement experiment.

This script evaluates a patched native hnswlib build where graph construction
remains unchanged, and only the search-time scorer is swapped between:
- simhash_baseline
- qacos

The index is built on the original float document embeddings. Document sign
sketches are attached as sidecar data after build. Native search returns:
- the final coarse frontier (unchanged from the earlier experiment)
- the bottom-layer visited-and-scored distinct node pool

Candidate-efficiency metrics are computed offline from prefixes of the visited
candidate pool. This is a logging/return-path extension only; native HNSW
search behavior is left unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

MODEL_TAG_DEFAULT = "sentence-transformers-all-mpnet-base-v2"
EPS = 1e-12
NATIVE_VISITED_BUDGET_SENTINEL = -1


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def write_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_rows(rows: List[Dict[str, object]], group_keys: List[str], mean_keys: List[str]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        bucket = buckets.setdefault(
            key,
            {"count": 0, **{k: row[k] for k in group_keys}, **{mk: 0.0 for mk in mean_keys}},
        )
        bucket["count"] = int(bucket["count"]) + 1
        for mk in mean_keys:
            val = row.get(mk, np.nan)
            if val is not None and np.isfinite(float(val)):
                bucket[mk] = float(bucket[mk]) + float(val)

    out: List[Dict[str, object]] = []
    for bucket in buckets.values():
        cnt = max(int(bucket.pop("count")), 1)
        new_row = dict(bucket)
        for mk in mean_keys:
            new_row[mk] = float(new_row[mk]) / float(cnt)
        out.append(new_row)
    out.sort(key=lambda r: tuple(r[k] for k in group_keys))
    return out


def l2_normalize(X: np.ndarray, eps: float = EPS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_before = np.linalg.norm(X, axis=1).astype(np.float32)
    X = X / np.clip(n_before[:, None], eps, None)
    n_after = np.linalg.norm(X, axis=1).astype(np.float32)
    return X.astype(np.float32), n_before, n_after


def signs_from_proj(P: np.ndarray) -> np.ndarray:
    return np.where(P >= 0.0, 1, -1).astype(np.int8)


def pack_sign_bits(signs_pm1: np.ndarray) -> np.ndarray:
    signs_pos = signs_pm1 > 0
    n, m = signs_pos.shape
    code_words = (m + 63) // 64
    out = np.zeros((n, code_words), dtype=np.uint64)
    for bit in range(m):
        out[:, bit // 64] |= signs_pos[:, bit].astype(np.uint64) << np.uint64(bit % 64)
    return out


def load_mmap_with_meta(mmap_path: Path, json_path: Path) -> np.memmap:
    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    shape = tuple(int(x) for x in meta["shape"])
    dtype = np.dtype(meta["dtype"])
    return np.memmap(mmap_path, mode="r", dtype=dtype, shape=shape)


def load_ids(ids_path: Path, expected_rows: int) -> List[str]:
    with ids_path.open("r", encoding="utf-8") as f:
        ids = [line.rstrip("\n") for line in f]
    if len(ids) != expected_rows:
        ids = [str(i) for i in range(expected_rows)]
    return ids


def dataset_paths(cache_dir: Path, dataset: str, model_tag: str) -> Dict[str, Path]:
    docs_stem = f"{dataset}_docs_{model_tag}"
    q_stem = f"{dataset}_queries_{model_tag}"
    return {
        "docs_mmap": cache_dir / f"{docs_stem}.mmap",
        "docs_json": cache_dir / f"{docs_stem}.json",
        "docs_ids": cache_dir / f"{docs_stem}_ids.txt",
        "queries_mmap": cache_dir / f"{q_stem}.mmap",
        "queries_json": cache_dir / f"{q_stem}.json",
        "queries_ids": cache_dir / f"{q_stem}_ids.txt",
    }


def min_prefix_for_target_recall(
    ordered_ids: np.ndarray,
    true_scores: np.ndarray,
    oracle_set: set[int],
    K: int,
    target_recall: float,
) -> float:
    if ordered_ids.size == 0:
        return np.nan
    import heapq

    heap: List[Tuple[float, int]] = []
    for i, d in enumerate(ordered_ids.tolist(), start=1):
        s = float(true_scores[d])
        if len(heap) < K:
            heapq.heappush(heap, (s, d))
        elif s > heap[0][0]:
            heapq.heapreplace(heap, (s, d))
        if len(heap) == K:
            cur_ids = {doc_id for _, doc_id in heap}
            rec = len(cur_ids & oracle_set) / float(K)
            if rec >= target_recall:
                return float(i)
    return np.nan


def recall_from_prefix(ordered_ids: np.ndarray, true_scores: np.ndarray, oracle_set: set[int], K: int, prefix: int) -> float:
    prefix = min(prefix, int(ordered_ids.size))
    if prefix <= 0:
        return np.nan
    cand = ordered_ids[:prefix]
    scores = true_scores[cand]
    take = min(K, cand.size)
    if take <= 0:
        return np.nan
    top_local = np.argpartition(-scores, kth=take - 1)[:take]
    reranked = cand[top_local[np.argsort(-scores[top_local])]]
    return len(set(reranked.tolist()) & oracle_set) / float(K)


def query_metrics_from_search(
    coarse_ids: np.ndarray,
    true_scores: np.ndarray,
    oracle_top10: np.ndarray,
    oracle_top100: np.ndarray,
    target_ps: Iterable[float],
) -> Dict[str, float]:
    num_coarse = int(coarse_ids.size)
    if num_coarse == 0:
        row = {
            "recall_at_10": 0.0,
            "recall_at_100": 0.0,
            "num_coarse_candidates": 0,
            "num_reranked_candidates": 0,
            "overlap_top10": 0,
            "overlap_top100": 0,
            "mean_true_cos_coarse_top10": np.nan,
            "mean_true_cos_coarse_top100": np.nan,
        }
        for p in target_ps:
            row[f"min_reranked_for_recall10_p{p:.1f}"] = np.nan
            row[f"min_reranked_for_recall100_p{p:.1f}"] = np.nan
        return row

    oracle10_set = set(oracle_top10.tolist())
    oracle100_set = set(oracle_top100.tolist())
    coarse_top10 = coarse_ids[: min(10, num_coarse)]
    coarse_top100 = coarse_ids[: min(100, num_coarse)]
    overlap_top10 = int(len(set(coarse_top10.tolist()) & oracle10_set))
    overlap_top100 = int(len(set(coarse_top100.tolist()) & oracle100_set))
    mean_true_cos_coarse_top10 = float(np.mean(true_scores[coarse_top10])) if coarse_top10.size else np.nan
    mean_true_cos_coarse_top100 = float(np.mean(true_scores[coarse_top100])) if coarse_top100.size else np.nan

    cand_scores = true_scores[coarse_ids]
    order = np.argsort(-cand_scores, kind="mergesort")
    reranked_ids = coarse_ids[order]
    top10 = set(reranked_ids[: min(10, reranked_ids.size)].tolist())
    top100 = set(reranked_ids[: min(100, reranked_ids.size)].tolist())
    recall_at_10 = len(top10 & oracle10_set) / 10.0
    recall_at_100 = len(top100 & oracle100_set) / 100.0

    row = {
        "recall_at_10": float(recall_at_10),
        "recall_at_100": float(recall_at_100),
        "num_coarse_candidates": num_coarse,
        "num_reranked_candidates": num_coarse,
        "overlap_top10": overlap_top10,
        "overlap_top100": overlap_top100,
        "mean_true_cos_coarse_top10": mean_true_cos_coarse_top10,
        "mean_true_cos_coarse_top100": mean_true_cos_coarse_top100,
    }
    for p in target_ps:
        row[f"min_reranked_for_recall10_p{p:.1f}"] = min_prefix_for_target_recall(coarse_ids, true_scores, oracle10_set, 10, float(p))
        row[f"min_reranked_for_recall100_p{p:.1f}"] = min_prefix_for_target_recall(coarse_ids, true_scores, oracle100_set, 100, float(p))
    return row


def candidate_efficiency_metrics(
    visited_candidate_ids: np.ndarray,
    visited_candidate_dists: np.ndarray,
    true_scores: np.ndarray,
    oracle_top10: np.ndarray,
    oracle_top100: np.ndarray,
    target_ps: Sequence[float],
    prefix_grid: Sequence[int],
) -> Dict[str, float]:
    row: Dict[str, float] = {
        "visited_candidate_pool_size": float(visited_candidate_ids.size),
    }
    oracle10_set = set(oracle_top10.tolist())
    oracle100_set = set(oracle_top100.tolist())

    if visited_candidate_ids.size == 0:
        for p in target_ps:
            row[f"pool_min_reranked_for_recall10_p{p:.1f}"] = np.nan
            row[f"pool_min_reranked_for_recall100_p{p:.1f}"] = np.nan
        for prefix in prefix_grid:
            row[f"pool_recall10_at_prefix_{prefix}"] = np.nan
            row[f"pool_recall100_at_prefix_{prefix}"] = np.nan
        return row

    order = np.argsort(visited_candidate_dists, kind="mergesort")
    ordered_ids = visited_candidate_ids[order]

    for p in target_ps:
        row[f"pool_min_reranked_for_recall10_p{p:.1f}"] = min_prefix_for_target_recall(ordered_ids, true_scores, oracle10_set, 10, float(p))
        row[f"pool_min_reranked_for_recall100_p{p:.1f}"] = min_prefix_for_target_recall(ordered_ids, true_scores, oracle100_set, 100, float(p))
    for prefix in prefix_grid:
        row[f"pool_recall10_at_prefix_{prefix}"] = recall_from_prefix(ordered_ids, true_scores, oracle10_set, 10, int(prefix))
        row[f"pool_recall100_at_prefix_{prefix}"] = recall_from_prefix(ordered_ids, true_scores, oracle100_set, 100, int(prefix))
    return row


def build_efficiency_summary(rows: List[Dict[str, object]], target_ps: Sequence[float]) -> List[Dict[str, object]]:
    group_keys = ["dataset", "graph_M", "method", "scorer_type", "sketch_bits", "visited_budget", "ef_search"]
    metric_specs: List[Tuple[str, str]] = []
    for p in target_ps:
        for base in (f"pool_min_reranked_for_recall10_p{p:.1f}", f"pool_min_reranked_for_recall100_p{p:.1f}"):
            metric_specs.extend([
                (base, f"avg_{base}"),
                (base, f"median_{base}"),
                (base, f"reachable_frac_{base}"),
            ])
    buckets: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        buckets.setdefault(key, []).append(row)
    out: List[Dict[str, object]] = []
    for key, group in buckets.items():
        summary = {k: v for k, v in zip(group_keys, key)}
        summary["num_queries"] = len(group)
        for raw_key, out_key in metric_specs:
            vals = [float(r[raw_key]) for r in group if r.get(raw_key) is not None and np.isfinite(float(r[raw_key]))]
            if out_key.startswith("avg_"):
                summary[out_key] = float(np.mean(vals)) if vals else np.nan
            elif out_key.startswith("median_"):
                summary[out_key] = float(statistics.median(vals)) if vals else np.nan
            else:
                summary[out_key] = float(len(vals)) / float(len(group)) if group else np.nan
        out.append(summary)
    out.sort(key=lambda r: tuple(r[k] for k in group_keys))
    return out


def build_pairwise_win_summary(rows: List[Dict[str, object]], target_ps: Sequence[float]) -> List[Dict[str, object]]:
    pair_group_keys = ["dataset", "graph_M", "sketch_bits", "visited_budget", "ef_search"]
    query_key_names = pair_group_keys + ["query_id"]
    per_query: Dict[Tuple[object, ...], Dict[str, Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in query_key_names)
        per_query.setdefault(key, {})[str(row["method"])] = row

    agg: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for key, methods in per_query.items():
        if "simhash" not in methods or "qacos" not in methods:
            continue
        group_key = key[:-1]
        bucket = agg.setdefault(group_key, {k: v for k, v in zip(pair_group_keys, group_key)})
        bucket["num_queries"] = int(bucket.get("num_queries", 0)) + 1
        s = methods["simhash"]
        q = methods["qacos"]
        for p in target_ps:
            for raw in (f"pool_min_reranked_for_recall10_p{p:.1f}", f"pool_min_reranked_for_recall100_p{p:.1f}"):
                prefix = raw.replace("pool_", "")
                s_val = float(s[raw]) if np.isfinite(float(s[raw])) else np.nan
                q_val = float(q[raw]) if np.isfinite(float(q[raw])) else np.nan
                bucket[f"reachable_both_{prefix}"] = int(bucket.get(f"reachable_both_{prefix}", 0)) + int(np.isfinite(s_val) and np.isfinite(q_val))
                bucket[f"qacos_wins_{prefix}"] = int(bucket.get(f"qacos_wins_{prefix}", 0)) + int(np.isfinite(s_val) and np.isfinite(q_val) and q_val < s_val)
                bucket[f"simhash_wins_{prefix}"] = int(bucket.get(f"simhash_wins_{prefix}", 0)) + int(np.isfinite(s_val) and np.isfinite(q_val) and s_val < q_val)
                bucket[f"ties_{prefix}"] = int(bucket.get(f"ties_{prefix}", 0)) + int(np.isfinite(s_val) and np.isfinite(q_val) and s_val == q_val)
    out = list(agg.values())
    out.sort(key=lambda r: tuple(r[k] for k in pair_group_keys))
    return out


def import_hnswlib():
    import hnswlib  # type: ignore
    return hnswlib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="../emb_cache")
    parser.add_argument("--out_dir", type=str, default="native_hnsw_qacos_eval_out")
    parser.add_argument("--model_tag", type=str, default=MODEL_TAG_DEFAULT)
    parser.add_argument("--datasets", type=str, default="arguana,nfcorpus,scifact")
    parser.add_argument("--graph_M_values", type=str, default="16,32")
    parser.add_argument("--sketch_bits", type=str, default="64,128")
    parser.add_argument("--ef_search_values", type=str, default="50")
    parser.add_argument("--target_recalls", type=str, default="0.8,0.9")
    parser.add_argument("--prefix_grid", type=str, default="10,20,50,100,150,200,300,500")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qacos_iters", type=int, default=2)
    parser.add_argument("--max_queries", type=int, default=80)
    parser.add_argument("--num_threads", type=int, default=1)
    args = parser.parse_args()

    hnswlib = import_hnswlib()

    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    graph_M_values = sorted(parse_int_list(args.graph_M_values))
    sketch_bits_values = sorted(parse_int_list(args.sketch_bits))
    ef_search_values = sorted(parse_int_list(args.ef_search_values))
    target_ps = sorted(parse_float_list(args.target_recalls))
    prefix_grid = sorted(parse_int_list(args.prefix_grid))

    metadata = {
        "cache_dir": str(cache_dir),
        "datasets": datasets,
        "graph_M_values": graph_M_values,
        "sketch_bits": sketch_bits_values,
        "ef_search_values": ef_search_values,
        "visited_budget": "not_exposed_natively",
        "seed": int(args.seed),
        "qacos_iters": int(args.qacos_iters),
        "max_queries": int(args.max_queries),
        "num_threads": int(args.num_threads),
        "prefix_grid": prefix_grid,
        "candidate_pool_definition": "bottom-layer distinct nodes whose experimental scorer distance was actually computed",
    }

    rng_global = np.random.default_rng(args.seed)
    all_rows: List[Dict[str, object]] = []

    for ds in datasets:
        print(f"\n=== Dataset: {ds} ===", flush=True)
        p = dataset_paths(cache_dir, ds, args.model_tag)
        if any(not v.exists() for v in p.values()):
            print(f"[skip] missing cache files for {ds}", flush=True)
            continue

        docs = np.asarray(load_mmap_with_meta(p["docs_mmap"], p["docs_json"]), dtype=np.float32)
        queries = np.asarray(load_mmap_with_meta(p["queries_mmap"], p["queries_json"]), dtype=np.float32)
        query_ids = load_ids(p["queries_ids"], queries.shape[0])
        docs, _, _ = l2_normalize(docs)
        queries, _, _ = l2_normalize(queries)

        q_idx_all = np.arange(queries.shape[0], dtype=np.int32)
        if args.max_queries > 0 and args.max_queries < q_idx_all.size:
            q_idx_all = rng_global.choice(q_idx_all, size=args.max_queries, replace=False)
            q_idx_all.sort()
        Q = queries[q_idx_all]
        q_ids_sel = [query_ids[i] for i in q_idx_all.tolist()]

        true_scores = Q @ docs.T
        oracle_top100 = np.argpartition(-true_scores, kth=99, axis=1)[:, :100]
        oracle_top10 = np.argpartition(-true_scores, kth=9, axis=1)[:, :10]
        for i in range(true_scores.shape[0]):
            o10 = oracle_top10[i]
            oracle_top10[i] = o10[np.argsort(-true_scores[i, o10])]
            o100 = oracle_top100[i]
            oracle_top100[i] = o100[np.argsort(-true_scores[i, o100])]

        doc_labels = np.arange(docs.shape[0], dtype=np.int64)
        ds_rows_start = len(all_rows)
        for graph_M in graph_M_values:
            print(f"building native HNSW ds={ds} M={graph_M}", flush=True)
            index = hnswlib.Index(space="cosine", dim=docs.shape[1])
            index.init_index(max_elements=docs.shape[0], M=graph_M, ef_construction=200, random_seed=args.seed, allow_replace_deleted=False)
            index.add_items(docs.astype(np.float32), doc_labels, num_threads=args.num_threads, replace_deleted=False)
            index.set_num_threads(args.num_threads)

            for m_bits in sketch_bits_values:
                rng_h = np.random.default_rng(args.seed + 1009 * m_bits + len(ds))
                H = rng_h.standard_normal((m_bits, docs.shape[1])).astype(np.float32)
                doc_proj = docs @ H.T
                doc_sign = signs_from_proj(doc_proj)
                doc_sign_packed = pack_sign_bits(doc_sign)
                q_proj_all = (Q @ H.T).astype(np.float32)
                q_sign_all = signs_from_proj(q_proj_all)
                q_sign_packed_all = pack_sign_bits(q_sign_all)

                index.set_doc_sign_sketches(doc_labels, doc_sign_packed, sketch_bits=m_bits)

                for ef_search in ef_search_values:
                    index.set_ef(int(ef_search))
                    for method in ("simhash_baseline", "qacos"):
                        labels, dists, info = index.knn_query_experimental(
                            q_sign_packed_all,
                            q_proj_all if method == "qacos" else None,
                            k=min(max(int(ef_search), 100), docs.shape[0]),
                            num_threads=args.num_threads,
                            scorer=method,
                            qacos_iters=args.qacos_iters,
                        )
                        _ = labels
                        _ = dists
                        coarse_ids_lists = info["coarse_ids"]
                        visited_candidate_ids_lists = info["visited_candidate_ids"]
                        visited_candidate_dists_lists = info["visited_candidate_dists"]
                        visited_candidate_sizes = np.asarray(info["visited_candidate_sizes"])
                        visited_nodes = np.asarray(info["visited_nodes"])
                        upper_hops = np.asarray(info["upper_hops"])
                        lower_pops = np.asarray(info["lower_pops"])
                        lower_pushes = np.asarray(info["lower_pushes"])

                        for qi_local, qid in enumerate(q_ids_sel):
                            coarse_ids = np.array(list(coarse_ids_lists[qi_local]), dtype=np.int64)
                            qrow = query_metrics_from_search(coarse_ids, true_scores[qi_local], oracle_top10[qi_local], oracle_top100[qi_local], target_ps)
                            visited_candidate_ids = np.array(list(visited_candidate_ids_lists[qi_local]), dtype=np.int64)
                            visited_candidate_dists = np.array(list(visited_candidate_dists_lists[qi_local]), dtype=np.float32)
                            prow = candidate_efficiency_metrics(
                                visited_candidate_ids,
                                visited_candidate_dists,
                                true_scores[qi_local],
                                oracle_top10[qi_local],
                                oracle_top100[qi_local],
                                target_ps,
                                prefix_grid,
                            )
                            row: Dict[str, object] = {
                                "dataset": ds,
                                "graph_M": int(graph_M),
                                "method": "simhash" if method == "simhash_baseline" else "qacos",
                                "scorer_type": method,
                                "sketch_bits": int(m_bits),
                                "visited_budget": NATIVE_VISITED_BUDGET_SENTINEL,
                                "ef_search": int(ef_search),
                                "query_id": qid,
                                "visited_nodes": int(visited_nodes[qi_local]),
                                "upper_hops": int(upper_hops[qi_local]),
                                "lower_pops": int(lower_pops[qi_local]),
                                "lower_pushes": int(lower_pushes[qi_local]),
                                "visited_candidate_pool_size_raw": int(visited_candidate_sizes[qi_local]),
                            }
                            row.update(qrow)
                            row.update(prow)
                            all_rows.append(row)
                    print(f"[progress] ds={ds} M={graph_M} bits={m_bits} ef={ef_search} rows={len(all_rows)-ds_rows_start}", flush=True)

        ds_csv = out_dir / f"{ds}_native_hnsw_per_query.csv"
        write_rows_csv(ds_csv, all_rows[ds_rows_start:])
        print(f"saved per-query: {ds_csv}", flush=True)

    if not all_rows:
        raise RuntimeError("No results produced")

    per_query_csv = out_dir / "native_hnsw_per_query_all.csv"
    write_rows_csv(per_query_csv, all_rows)

    agg_cols = [
        "recall_at_10",
        "recall_at_100",
        "visited_nodes",
        "upper_hops",
        "lower_pops",
        "lower_pushes",
        "num_reranked_candidates",
        "overlap_top10",
        "overlap_top100",
        "num_coarse_candidates",
        "mean_true_cos_coarse_top10",
        "mean_true_cos_coarse_top100",
        "visited_candidate_pool_size",
        "visited_candidate_pool_size_raw",
    ] + [c for c in all_rows[0].keys() if str(c).startswith("pool_min_reranked_for_") or str(c).startswith("pool_recall")]

    summary_rows = summarize_rows(all_rows, ["dataset", "graph_M", "method", "scorer_type", "sketch_bits", "visited_budget", "ef_search"], agg_cols)
    rename_map = {
        "recall_at_10": "mean_recall_at_10",
        "recall_at_100": "mean_recall_at_100",
        "visited_nodes": "mean_visited_nodes",
        "upper_hops": "mean_upper_hops",
        "lower_pops": "mean_lower_pops",
        "lower_pushes": "mean_lower_pushes",
        "num_reranked_candidates": "mean_num_reranked_candidates",
        "overlap_top10": "mean_overlap_top10",
        "overlap_top100": "mean_overlap_top100",
        "num_coarse_candidates": "mean_num_coarse_candidates",
        "visited_candidate_pool_size": "mean_visited_candidate_pool_size",
        "visited_candidate_pool_size_raw": "mean_visited_candidate_pool_size_raw",
    }
    for row in summary_rows:
        for old, new in rename_map.items():
            row[new] = row.pop(old)

    summary_csv = out_dir / "native_hnsw_summary.csv"
    write_rows_csv(summary_csv, summary_rows)

    efficiency_summary_rows = build_efficiency_summary(all_rows, target_ps)
    efficiency_summary_csv = out_dir / "native_hnsw_candidate_efficiency_summary.csv"
    write_rows_csv(efficiency_summary_csv, efficiency_summary_rows)

    pairwise_summary_rows = build_pairwise_win_summary(all_rows, target_ps)
    pairwise_summary_csv = out_dir / "native_hnsw_candidate_efficiency_pairwise.csv"
    write_rows_csv(pairwise_summary_csv, pairwise_summary_rows)

    metadata["num_rows"] = int(len(all_rows))
    metadata["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with (out_dir / "native_hnsw_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(per_query_csv)
    print(summary_csv)
    print(efficiency_summary_csv)
    print(pairwise_summary_csv)
    print(out_dir / "native_hnsw_metadata.json")


if __name__ == "__main__":
    main()
