#!/usr/bin/env python3
"""Wall-clock evaluation for the patched native-HNSW scorer path.

This script keeps native HNSW graph construction unchanged and compares
SimHash vs QA-Cos at identical search settings, while measuring:
- native search latency reported inside the patched binding
- exact rerank latency over the final returned coarse frontier
- total latency = search + rerank

The current focus is same-setting timing at fixed efSearch (default: 100).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from run_native_hnsw_qacos_eval import (
    MODEL_TAG_DEFAULT,
    dataset_paths,
    l2_normalize,
    load_ids,
    load_mmap_with_meta,
    pack_sign_bits,
    query_metrics_from_search,
    signs_from_proj,
    write_rows_csv,
)


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def import_hnswlib():
    import hnswlib  # type: ignore

    return hnswlib


def percentile_ms(values_ms: Sequence[float], q: float) -> float:
    if not values_ms:
        return math.nan
    return float(np.percentile(np.asarray(values_ms, dtype=np.float64), q))


def exact_rerank_metrics_and_time_ns(
    query_vec: np.ndarray,
    docs: np.ndarray,
    coarse_ids: np.ndarray,
    oracle_top10: np.ndarray,
    oracle_top100: np.ndarray,
) -> tuple[Dict[str, float], int]:
    if coarse_ids.size == 0:
        return (
            {
                "recall_at_10": 0.0,
                "recall_at_100": 0.0,
                "num_coarse_candidates": 0,
                "num_reranked_candidates": 0,
                "overlap_top10": 0,
                "overlap_top100": 0,
                "mean_true_cos_coarse_top10": math.nan,
                "mean_true_cos_coarse_top100": math.nan,
            },
            0,
        )

    oracle10_set = set(oracle_top10.tolist())
    oracle100_set = set(oracle_top100.tolist())
    coarse_top10 = coarse_ids[: min(10, coarse_ids.size)]
    coarse_top100 = coarse_ids[: min(100, coarse_ids.size)]

    t0 = time.perf_counter_ns()
    cand_scores = docs[coarse_ids] @ query_vec
    order = np.argsort(-cand_scores, kind="mergesort")
    reranked_ids = coarse_ids[order]
    t1 = time.perf_counter_ns()

    top10 = set(reranked_ids[: min(10, reranked_ids.size)].tolist())
    top100 = set(reranked_ids[: min(100, reranked_ids.size)].tolist())
    mean_top10 = float(np.mean(cand_scores[: min(10, cand_scores.size)])) if cand_scores.size else math.nan
    mean_top100 = float(np.mean(cand_scores[: min(100, cand_scores.size)])) if cand_scores.size else math.nan

    metrics = {
        "recall_at_10": len(top10 & oracle10_set) / 10.0,
        "recall_at_100": len(top100 & oracle100_set) / 100.0,
        "num_coarse_candidates": int(coarse_ids.size),
        "num_reranked_candidates": int(coarse_ids.size),
        "overlap_top10": int(len(set(coarse_top10.tolist()) & oracle10_set)),
        "overlap_top100": int(len(set(coarse_top100.tolist()) & oracle100_set)),
        "mean_true_cos_coarse_top10": mean_top10,
        "mean_true_cos_coarse_top100": mean_top100,
    }
    return metrics, int(t1 - t0)


def safe_cpu_info() -> Dict[str, str]:
    out = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    for key, cmd in (
        ("cpu_brand", ["sysctl", "-n", "machdep.cpu.brand_string"]),
        ("cpu_cores_logical", ["sysctl", "-n", "hw.logicalcpu"]),
        ("cpu_cores_physical", ["sysctl", "-n", "hw.physicalcpu"]),
    ):
        try:
            out[key] = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            continue
    return out


def summarize_setting(rows: List[Dict[str, object]]) -> Dict[str, object]:
    first = rows[0]
    search_vals = [float(r["search_ms"]) for r in rows]
    rerank_vals = [float(r["rerank_ms"]) for r in rows]
    total_vals = [float(r["total_ms"]) for r in rows]
    out: Dict[str, object] = {
        "dataset": first["dataset"],
        "bits": first["sketch_bits"],
        "M": first["graph_M"],
        "efSearch": first["ef_search"],
        "scorer": first["method"],
        "mean_search_ms": float(np.mean(search_vals)),
        "mean_rerank_ms": float(np.mean(rerank_vals)),
        "mean_total_ms": float(np.mean(total_vals)),
        "p50_total_ms": percentile_ms(total_vals, 50.0),
        "p95_total_ms": percentile_ms(total_vals, 95.0),
        "Recall@10": float(np.mean([float(r["recall_at_10"]) for r in rows])),
        "Recall@100": float(np.mean([float(r["recall_at_100"]) for r in rows])),
        "top10_overlap": float(np.mean([float(r["overlap_top10"]) for r in rows])),
        "top100_overlap": float(np.mean([float(r["overlap_top100"]) for r in rows])),
        "mean_visited_nodes": float(np.mean([float(r["visited_nodes"]) for r in rows])),
        "mean_num_reranked": float(np.mean([float(r["num_reranked_candidates"]) for r in rows])),
        "num_points": len(rows),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="/Users/nyang/Library/CloudStorage/OneDrive-이화여자대학교/Code/simhash/emb_cache")
    parser.add_argument("--out_dir", type=str, default="native_hnsw_wallclock_same_setting")
    parser.add_argument("--model_tag", type=str, default=MODEL_TAG_DEFAULT)
    parser.add_argument("--datasets", type=str, default="nfcorpus,fiqa")
    parser.add_argument("--graph_M_values", type=str, default="16,32")
    parser.add_argument("--sketch_bits", type=str, default="128")
    parser.add_argument("--ef_search_values", type=str, default="100")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qacos_iters", type=int, default=2)
    parser.add_argument("--max_queries", type=int, default=80)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--warmup_runs", type=int, default=2)
    parser.add_argument("--timed_runs", type=int, default=5)
    parser.add_argument("--methods", type=str, default="simhash_baseline,qacos")
    parser.add_argument("--mills_approx", action="store_true")
    parser.add_argument("--qacos_gate_min_same", type=int, default=-1)
    parser.add_argument("--qacos_gate_max_same", type=int, default=-1)
    parser.add_argument("--return_k", type=int, default=100)
    args = parser.parse_args()

    hnswlib = import_hnswlib()

    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    graph_M_values = sorted(parse_int_list(args.graph_M_values))
    sketch_bits_values = sorted(parse_int_list(args.sketch_bits))
    ef_search_values = sorted(parse_int_list(args.ef_search_values))
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]

    metadata: Dict[str, object] = {
        "cache_dir": str(cache_dir),
        "datasets": datasets,
        "graph_M_values": graph_M_values,
        "sketch_bits": sketch_bits_values,
        "ef_search_values": ef_search_values,
        "seed": int(args.seed),
        "qacos_iters": int(args.qacos_iters),
        "max_queries": int(args.max_queries),
        "num_threads": int(args.num_threads),
        "warmup_runs": int(args.warmup_runs),
        "timed_runs": int(args.timed_runs),
        "methods": methods,
        "mills_approx": bool(args.mills_approx),
        "qacos_gate_min_same": int(args.qacos_gate_min_same),
        "qacos_gate_max_same": int(args.qacos_gate_max_same),
        "return_k": int(args.return_k),
        "timing_definition": {
            "search_ms": "Native search time measured inside knn_query_experimental; includes scorer evaluation and heap/result packing, excludes Python list/dict marshalling.",
            "rerank_ms": "Python-side exact float-cosine rerank over the final returned coarse frontier only.",
            "total_ms": "search_ms + rerank_ms.",
        },
        "hardware": safe_cpu_info(),
    }

    rng_global = np.random.default_rng(args.seed)
    per_query_rows: List[Dict[str, object]] = []

    for ds in datasets:
        print(f"\\n=== Dataset: {ds} ===", flush=True)
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

        for graph_M in graph_M_values:
            print(f"building native HNSW ds={ds} M={graph_M}", flush=True)
            index = hnswlib.Index(space="cosine", dim=docs.shape[1])
            index.init_index(
                max_elements=docs.shape[0],
                M=graph_M,
                ef_construction=200,
                random_seed=args.seed,
                allow_replace_deleted=False,
            )
            index.add_items(docs.astype(np.float32), doc_labels, num_threads=args.num_threads, replace_deleted=False)
            index.set_num_threads(args.num_threads)

            for m_bits in sketch_bits_values:
                rng_h = np.random.default_rng(args.seed + 1009 * m_bits + len(ds))
                H = rng_h.standard_normal((m_bits, docs.shape[1])).astype(np.float32)
                doc_proj = docs @ H.T
                doc_sign_packed = pack_sign_bits(signs_from_proj(doc_proj))
                q_proj_all = (Q @ H.T).astype(np.float32)
                q_sign_packed_all = pack_sign_bits(signs_from_proj(q_proj_all))

                index.set_doc_sign_sketches(doc_labels, doc_sign_packed, sketch_bits=m_bits)

                for ef_search in ef_search_values:
                    index.set_ef(int(ef_search))
                    return_k = min(max(int(ef_search), int(args.return_k)), docs.shape[0])
                    for method in methods:
                        for warm_idx in range(args.warmup_runs):
                            index.knn_query_experimental(
                                q_sign_packed_all,
                                q_proj_all if method != "simhash_baseline" else None,
                                k=return_k,
                                num_threads=args.num_threads,
                                scorer=method,
                                qacos_iters=args.qacos_iters,
                                mills_approx=args.mills_approx,
                                qacos_gate_min_same=args.qacos_gate_min_same,
                                qacos_gate_max_same=args.qacos_gate_max_same,
                            )
                            print(
                                f"[warmup] ds={ds} M={graph_M} bits={m_bits} ef={ef_search} scorer={method} run={warm_idx + 1}/{args.warmup_runs}",
                                flush=True,
                            )

                        for repeat_idx in range(args.timed_runs):
                            _, _, info = index.knn_query_experimental(
                                q_sign_packed_all,
                                q_proj_all if method != "simhash_baseline" else None,
                                k=return_k,
                                num_threads=args.num_threads,
                                scorer=method,
                                qacos_iters=args.qacos_iters,
                                mills_approx=args.mills_approx,
                                qacos_gate_min_same=args.qacos_gate_min_same,
                                qacos_gate_max_same=args.qacos_gate_max_same,
                            )
                            coarse_ids_lists = info["coarse_ids"]
                            visited_nodes = np.asarray(info["visited_nodes"])
                            search_ns = np.asarray(info["search_ns"], dtype=np.uint64)

                            for qi_local, qid in enumerate(q_ids_sel):
                                coarse_ids = np.array(list(coarse_ids_lists[qi_local]), dtype=np.int64)
                                metrics, rerank_ns = exact_rerank_metrics_and_time_ns(
                                    Q[qi_local],
                                    docs,
                                    coarse_ids,
                                    oracle_top10[qi_local],
                                    oracle_top100[qi_local],
                                )
                                row: Dict[str, object] = {
                                    "dataset": ds,
                                    "graph_M": int(graph_M),
                                    "method": "simhash" if method == "simhash_baseline" else method,
                                    "scorer_type": method,
                                    "sketch_bits": int(m_bits),
                                    "ef_search": int(ef_search),
                                    "query_id": qid,
                                    "repeat_idx": int(repeat_idx),
                                    "search_ms": float(search_ns[qi_local]) / 1e6,
                                    "rerank_ms": float(rerank_ns) / 1e6,
                                    "total_ms": (float(search_ns[qi_local]) + float(rerank_ns)) / 1e6,
                                    "visited_nodes": int(visited_nodes[qi_local]),
                                }
                                row.update(metrics)
                                per_query_rows.append(row)

                            print(
                                f"[timed] ds={ds} M={graph_M} bits={m_bits} ef={ef_search} scorer={method} run={repeat_idx + 1}/{args.timed_runs}",
                                flush=True,
                            )

    if not per_query_rows:
        raise RuntimeError("No results produced")

    per_query_csv = out_dir / "native_hnsw_wallclock_per_query.csv"
    write_rows_csv(per_query_csv, per_query_rows)

    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in per_query_rows:
        key = (
            row["dataset"],
            row["sketch_bits"],
            row["graph_M"],
            row["ef_search"],
            row["method"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = [summarize_setting(rows) for _, rows in sorted(grouped.items())]
    summary_csv = out_dir / "native_hnsw_wallclock_summary.csv"
    write_rows_csv(summary_csv, summary_rows)

    metadata["num_rows"] = int(len(per_query_rows))
    metadata["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with (out_dir / "native_hnsw_wallclock_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\\nSaved:")
    print(per_query_csv)
    print(summary_csv)
    print(out_dir / "native_hnsw_wallclock_metadata.json")


if __name__ == "__main__":
    main()
