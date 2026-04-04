#!/usr/bin/env python3
"""Storage-aware matched-recall timing for native HNSW experiments.

This runner keeps the native HNSW search path unchanged and reevaluates only
the exact full-vector rerank stage under file-backed storage modes:

- warm_cache: normalized full vectors stored in a file-backed memmap
- cache_limited: a larger replicated file-backed store with query-dependent
  replica selection, plus optional cache-pressure before each timed run

The matched efSearch settings are read from a previously generated matched
Recall@10 table. Since rerank storage changes do not affect search results,
the chosen efSearch values remain valid and only wall-clock timing changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from run_native_hnsw_qacos_eval import (
    MODEL_TAG_DEFAULT,
    dataset_paths,
    l2_normalize,
    load_ids,
    load_mmap_with_meta,
    pack_sign_bits,
    signs_from_proj,
    write_rows_csv,
)


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def import_hnswlib():
    import hnswlib  # type: ignore

    return hnswlib


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
        ("mem_bytes", ["sysctl", "-n", "hw.memsize"]),
    ):
        try:
            out[key] = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            continue
    return out


def percentile_ms(values_ms: Sequence[float], q: float) -> float:
    if not values_ms:
        return math.nan
    return float(np.percentile(np.asarray(values_ms, dtype=np.float64), q))


def read_matched_rows(path: Path, target_metric: str, target_values: Sequence[float]) -> List[Dict[str, object]]:
    wanted = {float(v) for v in target_values}
    out: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            if raw.get("target_metric") != target_metric:
                continue
            try:
                target_value = float(raw["target_value"])
            except Exception:
                continue
            if target_value not in wanted:
                continue
            chosen = str(raw.get("chosen_efSearch", "")).strip()
            if not chosen:
                continue
            out.append(
                {
                    "dataset": str(raw["dataset"]),
                    "bits": int(float(raw["bits"])),
                    "M": int(float(raw["M"])),
                    "target_metric": target_metric,
                    "target_value": float(target_value),
                    "scorer": str(raw["scorer"]),
                    "chosen_efSearch": int(float(chosen)),
                }
            )
    return out


def stable_hash_u64(parts: Sequence[object]) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "little", signed=False)


def ensure_memmap_store(path_stem: Path, docs: np.ndarray) -> Tuple[np.memmap, Path, Path]:
    mmap_path = path_stem.with_suffix(".mmap")
    json_path = path_stem.with_suffix(".json")
    meta = {
        "shape": [int(docs.shape[0]), int(docs.shape[1])],
        "dtype": str(docs.dtype),
    }
    need_write = True
    if mmap_path.exists() and json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                old_meta = json.load(f)
            if old_meta == meta and mmap_path.stat().st_size == docs.nbytes:
                need_write = False
        except Exception:
            need_write = True
    if need_write:
        mm = np.memmap(mmap_path, mode="w+", dtype=docs.dtype, shape=docs.shape)
        mm[:] = docs[:]
        mm.flush()
        del mm
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    return np.memmap(mmap_path, mode="r", dtype=docs.dtype, shape=docs.shape), mmap_path, json_path


def ensure_replicated_store(path_stem: Path, docs: np.ndarray, target_bytes: int) -> Tuple[np.memmap, int, Path, Path]:
    base_bytes = int(docs.nbytes)
    num_replicas = max(1, int(math.ceil(float(target_bytes) / float(max(base_bytes, 1)))))
    full_shape = (num_replicas * docs.shape[0], docs.shape[1])
    mmap_path = path_stem.with_suffix(".mmap")
    json_path = path_stem.with_suffix(".json")
    meta = {
        "shape": [int(full_shape[0]), int(full_shape[1])],
        "dtype": str(docs.dtype),
        "num_replicas": int(num_replicas),
        "base_docs": int(docs.shape[0]),
    }
    need_write = True
    if mmap_path.exists() and json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                old_meta = json.load(f)
            if old_meta == meta and mmap_path.stat().st_size == int(np.prod(full_shape)) * docs.dtype.itemsize:
                need_write = False
        except Exception:
            need_write = True
    if need_write:
        mm = np.memmap(mmap_path, mode="w+", dtype=docs.dtype, shape=full_shape)
        rows = docs.shape[0]
        for rep in range(num_replicas):
            start = rep * rows
            mm[start : start + rows] = docs
        mm.flush()
        del mm
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    return np.memmap(mmap_path, mode="r", dtype=docs.dtype, shape=full_shape), num_replicas, mmap_path, json_path


def ensure_cache_buster(path: Path, size_bytes: int) -> np.memmap:
    dtype = np.uint8
    need_write = not path.exists() or path.stat().st_size != int(size_bytes)
    if need_write:
        mm = np.memmap(path, mode="w+", dtype=dtype, shape=(size_bytes,))
        block = np.arange(1 << 20, dtype=dtype)
        for start in range(0, size_bytes, block.size):
            end = min(size_bytes, start + block.size)
            mm[start:end] = block[: end - start]
        mm.flush()
        del mm
    return np.memmap(path, mode="r", dtype=dtype, shape=(size_bytes,))


def apply_cache_pressure(mm: np.memmap, page_bytes: int = 4096) -> int:
    arr = np.asarray(mm, dtype=np.uint8)
    if arr.size == 0:
        return 0
    step = max(1, int(page_bytes))
    total = 0
    for idx in range(0, arr.size, step):
        total += int(arr[idx])
    return total


def physical_rows_for_query(
    coarse_ids: np.ndarray,
    query_id: str,
    num_docs: int,
    num_replicas: int,
    storage_mode: str,
) -> np.ndarray:
    if storage_mode == "warm_cache" or num_replicas <= 1:
        return coarse_ids
    qsalt = stable_hash_u64((query_id, "replica"))
    doc_u = coarse_ids.astype(np.uint64, copy=False)
    reps = ((doc_u * np.uint64(11400714819323198485)) ^ np.uint64(qsalt)) % np.uint64(num_replicas)
    return reps.astype(np.int64) * int(num_docs) + coarse_ids.astype(np.int64, copy=False)


def exact_rerank_storage_metrics_and_time_ns(
    query_vec: np.ndarray,
    store: np.memmap,
    coarse_ids: np.ndarray,
    physical_rows: np.ndarray,
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
            },
            0,
        )

    oracle10_set = set(oracle_top10.tolist())
    oracle100_set = set(oracle_top100.tolist())
    coarse_top10 = coarse_ids[: min(10, coarse_ids.size)]
    coarse_top100 = coarse_ids[: min(100, coarse_ids.size)]

    t0 = time.perf_counter_ns()
    cand_vecs = np.asarray(store[physical_rows], dtype=np.float32)
    cand_scores = cand_vecs @ query_vec
    order = np.argsort(-cand_scores, kind="mergesort")
    reranked_ids = coarse_ids[order]
    t1 = time.perf_counter_ns()

    top10 = set(reranked_ids[: min(10, reranked_ids.size)].tolist())
    top100 = set(reranked_ids[: min(100, reranked_ids.size)].tolist())
    metrics = {
        "recall_at_10": len(top10 & oracle10_set) / 10.0,
        "recall_at_100": len(top100 & oracle100_set) / 100.0,
        "num_coarse_candidates": int(coarse_ids.size),
        "num_reranked_candidates": int(coarse_ids.size),
        "overlap_top10": int(len(set(coarse_top10.tolist()) & oracle10_set)),
        "overlap_top100": int(len(set(coarse_top100.tolist()) & oracle100_set)),
    }
    return metrics, int(t1 - t0)


def summarize_setting(rows: List[Dict[str, object]]) -> Dict[str, object]:
    first = rows[0]
    search_vals = [float(r["search_ms"]) for r in rows]
    rerank_vals = [float(r["rerank_ms"]) for r in rows]
    total_vals = [float(r["total_ms"]) for r in rows]
    target_metric = str(first["target_metric"])
    if target_metric == "Recall@100":
        achieved_key = "recall_at_100"
    else:
        achieved_key = "recall_at_10"
    return {
        "dataset": first["dataset"],
        "bits": first["sketch_bits"],
        "M": first["graph_M"],
        "target_metric": first["target_metric"],
        "target_value": first["target_value"],
        "scorer": first["method"],
        "chosen_efSearch": first["ef_search"],
        "storage_mode": first["storage_mode"],
        "mean_search_ms": float(np.mean(search_vals)),
        "mean_rerank_ms": float(np.mean(rerank_vals)),
        "mean_total_ms": float(np.mean(total_vals)),
        "p50_total_ms": percentile_ms(total_vals, 50.0),
        "p95_total_ms": percentile_ms(total_vals, 95.0),
        "achieved_metric": float(np.mean([float(r[achieved_key]) for r in rows])),
        "Recall@100": float(np.mean([float(r["recall_at_100"]) for r in rows])),
        "top10_overlap": float(np.mean([float(r["overlap_top10"]) for r in rows])),
        "top100_overlap": float(np.mean([float(r["overlap_top100"]) for r in rows])),
        "mean_visited_nodes": float(np.mean([float(r["visited_nodes"]) for r in rows])),
        "mean_num_reranked": float(np.mean([float(r["num_reranked_candidates"]) for r in rows])),
        "num_points": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="/Users/nyang/Library/CloudStorage/OneDrive-이화여자대학교/Code/simhash/emb_cache")
    parser.add_argument("--matched_table", type=str, default="/tmp/hnswlib-wallclock-matched-r10-gated74_96/matched_recall_r10_table.csv")
    parser.add_argument("--out_dir", type=str, default="/tmp/hnswlib-storage-aware-matched")
    parser.add_argument("--storage_root", type=str, default="/tmp/hnswlib-storage-aware")
    parser.add_argument("--model_tag", type=str, default=MODEL_TAG_DEFAULT)
    parser.add_argument("--datasets", type=str, default="nfcorpus,fiqa")
    parser.add_argument("--target_values", type=str, default="0.8,0.9")
    parser.add_argument("--target_metric", type=str, default="Recall@10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_queries", type=int, default=80)
    parser.add_argument("--qacos_iters", type=int, default=1)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--timed_runs", type=int, default=3)
    parser.add_argument("--storage_modes", type=str, default="warm_cache,cache_limited")
    parser.add_argument("--storage_target_gb", type=float, default=1.5)
    parser.add_argument("--cache_buster_gb", type=float, default=1.0)
    parser.add_argument("--mills_approx", action="store_true")
    parser.add_argument("--qacos_gate_min_same", type=int, default=74)
    parser.add_argument("--qacos_gate_max_same", type=int, default=96)
    args = parser.parse_args()

    hnswlib = import_hnswlib()
    cache_dir = Path(args.cache_dir).resolve()
    matched_table = Path(args.matched_table).resolve()
    out_dir = Path(args.out_dir).resolve()
    storage_root = Path(args.storage_root).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_root.mkdir(parents=True, exist_ok=True)

    dataset_list = [x.strip() for x in args.datasets.split(",") if x.strip()]
    datasets = set(dataset_list)
    target_values = parse_float_list(args.target_values)
    storage_modes = [x.strip() for x in args.storage_modes.split(",") if x.strip()]
    matched_rows = [
        row
        for row in read_matched_rows(matched_table, args.target_metric, target_values)
        if row["dataset"] in datasets
    ]
    if not matched_rows:
        raise RuntimeError("No matched rows selected")

    rows_by_dataset: Dict[str, List[Dict[str, object]]] = {}
    for row in matched_rows:
        rows_by_dataset.setdefault(str(row["dataset"]), []).append(row)

    metadata: Dict[str, object] = {
        "cache_dir": str(cache_dir),
        "matched_table": str(matched_table),
        "storage_root": str(storage_root),
        "datasets": dataset_list,
        "target_metric": args.target_metric,
        "target_values": target_values,
        "max_queries": int(args.max_queries),
        "num_threads": int(args.num_threads),
        "warmup_runs": int(args.warmup_runs),
        "timed_runs": int(args.timed_runs),
        "storage_modes": storage_modes,
        "storage_target_gb": float(args.storage_target_gb),
        "cache_buster_gb": float(args.cache_buster_gb),
        "qacos_iters": int(args.qacos_iters),
        "mills_approx": bool(args.mills_approx),
        "qacos_gate_min_same": int(args.qacos_gate_min_same),
        "qacos_gate_max_same": int(args.qacos_gate_max_same),
        "timing_definition": {
            "search_ms": "Native search time measured inside knn_query_experimental; unchanged from the in-memory matched-recall setup.",
            "rerank_ms": "Exact rerank over file-backed normalized full vectors. warm_cache uses direct memmap reads; cache_limited uses a replicated file-backed store with query-dependent replica selection.",
            "total_ms": "search_ms + rerank_ms.",
            "cache_pressure": "For cache_limited, a separate file-backed cache-buster buffer is touched before each timed run to reduce trivially warm repeated reads.",
        },
        "hardware": safe_cpu_info(),
    }

    cache_buster = None
    cache_buster_checksum = 0
    rng_global = np.random.default_rng(args.seed)
    if "cache_limited" in storage_modes and args.cache_buster_gb > 0:
        cache_buster_path = storage_root / f"cache_buster_{args.cache_buster_gb:.1f}gb.bin"
        cache_buster = ensure_cache_buster(cache_buster_path, int(args.cache_buster_gb * (1024**3)))
        metadata["cache_buster_path"] = str(cache_buster_path)

    all_rows: List[Dict[str, object]] = []

    for ds in dataset_list:
        ds_rows = rows_by_dataset.get(ds, [])
        if not ds_rows:
            continue
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

        warm_store, warm_mmap_path, warm_json_path = ensure_memmap_store(storage_root / f"{ds}_normalized_store", docs)
        metadata.setdefault("stores", {})[f"{ds}_warm"] = {
            "mmap": str(warm_mmap_path),
            "json": str(warm_json_path),
            "bytes": int(warm_store.size * warm_store.dtype.itemsize),
        }
        cache_limited_store = None
        cache_limited_reps = 1
        if "cache_limited" in storage_modes:
            cache_limited_store, cache_limited_reps, repl_mmap_path, repl_json_path = ensure_replicated_store(
                storage_root / f"{ds}_replicated_store_{args.storage_target_gb:.1f}gb",
                docs,
                int(args.storage_target_gb * (1024**3)),
            )
            metadata.setdefault("stores", {})[f"{ds}_cache_limited"] = {
                "mmap": str(repl_mmap_path),
                "json": str(repl_json_path),
                "num_replicas": int(cache_limited_reps),
                "bytes": int(cache_limited_store.size * cache_limited_store.dtype.itemsize),
            }

        doc_labels = np.arange(docs.shape[0], dtype=np.int64)
        rows_by_group: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
        for row in ds_rows:
            rows_by_group.setdefault((int(row["M"]), int(row["bits"])), []).append(row)

        for (graph_M, m_bits), group_rows in sorted(rows_by_group.items()):
            print(f"building native HNSW ds={ds} M={graph_M} bits={m_bits}", flush=True)
            index = hnswlib.Index(space="cosine", dim=docs.shape[1])
            index.init_index(
                max_elements=docs.shape[0],
                M=int(graph_M),
                ef_construction=200,
                random_seed=args.seed,
                allow_replace_deleted=False,
            )
            index.add_items(docs.astype(np.float32), doc_labels, num_threads=args.num_threads, replace_deleted=False)
            index.set_num_threads(args.num_threads)

            rng_h = np.random.default_rng(args.seed + 1009 * int(m_bits) + len(ds))
            H = rng_h.standard_normal((int(m_bits), docs.shape[1])).astype(np.float32)
            doc_proj = docs @ H.T
            doc_sign_packed = pack_sign_bits(signs_from_proj(doc_proj))
            q_proj_all = (Q @ H.T).astype(np.float32)
            q_sign_packed_all = pack_sign_bits(signs_from_proj(q_proj_all))
            index.set_doc_sign_sketches(doc_labels, doc_sign_packed, sketch_bits=int(m_bits))

            for spec in sorted(group_rows, key=lambda r: (float(r["target_value"]), str(r["scorer"]))):
                scorer_name = str(spec["scorer"])
                method = "simhash_baseline" if scorer_name == "simhash" else scorer_name
                ef_search = int(spec["chosen_efSearch"])
                index.set_ef(ef_search)
                min_return_k = 100 if str(spec["target_metric"]) == "Recall@100" else 10
                return_k = min(max(ef_search, min_return_k), docs.shape[0])

                for storage_mode in storage_modes:
                    store = warm_store if storage_mode == "warm_cache" else cache_limited_store
                    num_replicas = 1 if storage_mode == "warm_cache" else int(cache_limited_reps)
                    if store is None:
                        continue
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
                            f"[warmup] ds={ds} M={graph_M} bits={m_bits} target={spec['target_value']} scorer={method} storage={storage_mode} run={warm_idx + 1}/{args.warmup_runs}",
                            flush=True,
                        )

                    for repeat_idx in range(args.timed_runs):
                        if storage_mode == "cache_limited" and cache_buster is not None:
                            cache_buster_checksum += apply_cache_pressure(cache_buster)
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
                            phys_rows = physical_rows_for_query(
                                coarse_ids,
                                qid,
                                num_docs=docs.shape[0],
                                num_replicas=num_replicas,
                                storage_mode=storage_mode,
                            )
                            metrics, rerank_ns = exact_rerank_storage_metrics_and_time_ns(
                                Q[qi_local],
                                store,
                                coarse_ids,
                                phys_rows,
                                oracle_top10[qi_local],
                                oracle_top100[qi_local],
                            )
                            row: Dict[str, object] = {
                                "dataset": ds,
                                "graph_M": int(graph_M),
                                "method": scorer_name,
                                "scorer_type": method,
                                "sketch_bits": int(m_bits),
                                "target_metric": str(spec["target_metric"]),
                                "target_value": float(spec["target_value"]),
                                "ef_search": int(ef_search),
                                "storage_mode": storage_mode,
                                "query_id": qid,
                                "repeat_idx": int(repeat_idx),
                                "search_ms": float(search_ns[qi_local]) / 1e6,
                                "rerank_ms": float(rerank_ns) / 1e6,
                                "total_ms": (float(search_ns[qi_local]) + float(rerank_ns)) / 1e6,
                                "visited_nodes": int(visited_nodes[qi_local]),
                            }
                            row.update(metrics)
                            all_rows.append(row)

                        print(
                            f"[timed] ds={ds} M={graph_M} bits={m_bits} target={spec['target_value']} scorer={method} storage={storage_mode} run={repeat_idx + 1}/{args.timed_runs}",
                            flush=True,
                        )

    if not all_rows:
        raise RuntimeError("No rows produced")

    per_query_csv = out_dir / "storage_aware_per_query.csv"
    write_rows_csv(per_query_csv, all_rows)
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in all_rows:
        key = (
            row["dataset"],
            row["sketch_bits"],
            row["graph_M"],
            row["target_metric"],
            row["target_value"],
            row["method"],
            row["ef_search"],
            row["storage_mode"],
        )
        grouped.setdefault(key, []).append(row)
    summary_rows = [summarize_setting(rows) for _, rows in sorted(grouped.items())]
    summary_csv = out_dir / "storage_aware_summary.csv"
    write_rows_csv(summary_csv, summary_rows)
    metadata["num_rows"] = int(len(all_rows))
    metadata["cache_buster_checksum"] = int(cache_buster_checksum)
    metadata["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with (out_dir / "storage_aware_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(per_query_csv)
    print(summary_csv)
    print(out_dir / "storage_aware_metadata.json")


if __name__ == "__main__":
    main()
