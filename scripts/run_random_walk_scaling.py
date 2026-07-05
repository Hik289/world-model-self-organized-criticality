"""
Random-walk scaling study — 6 graph families × 3 sizes × 3 seeds.

设计:
- 6 图 = 5 主图 (uniform_degree / exponential_degree / scale_free / modular / mixed)
        + baseline (uniform k-regular + fully symmetric payload)
- 3 |V| ∈ {100, 500, 1000}
- 3 seeds ∈ {42, 43, 44}
- N=100,000
- policy: random-walk
- memory: v2 StateAwareReservoirMemory (sort-by-freq + top_k=3), M=200
- 输出:
  - results/<graph_type>_v<n_nodes>_s<seed>_meta.json (每 run stats)
  - results/<graph_type>_v<n_nodes>_s<seed>_mem_freqs.json (完整 memory-access 频次分布, per memory_id counts)
  - results/<graph_type>_v<n_nodes>_s<seed>_mem_time.jsonl (memory-access 时序: per event 的 (wallclock_ms, agent_step, memory_id) 三元组, 为 PSD 铺路)
  - results/<graph_type>_v<n_nodes>_s<seed>_state_freqs.json
  - results/<graph_type>_v<n_nodes>_s<seed>_trans_freqs.json
  - summary.json: 54 runs 的 raw stats table

不写完整 events.jsonl (太大, 100k × 5 events = 500k 事件/run, 54 runs 会几十 GB).
仅写下游 SOC 分析需要的三样: (a) 频次 counts, (b) memory 访问时序 (agent_step + memory_id), (c) summary stats.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
from scipy import stats as spstats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldmodelsoc.memory.reservoir import StateAwareReservoirMemory  # noqa: E402
from worldmodelsoc.env.synthetic_graph_world import (  # noqa: E402
    build_graph, build_state_payloads, GRAPH_TYPES as MAIN_GRAPH_TYPES,
)


# ==============================================================================
# baseline 图 + 对称 payload (复用 baseline_hub_effect 逻辑)
# ==============================================================================


@dataclass
class SymmetricPayload:
    entities: List[str] = field(default_factory=lambda: ["e1", "e2", "e3"])
    relations: List[Tuple[str, str, str]] = field(default_factory=lambda: [("e1", "r1", "e2")])
    constraints: List[str] = field(default_factory=lambda: ["c1"])
    actions: List[str] = field(default_factory=lambda: ["a1", "a2", "a3"])


def build_baseline_graph(n_nodes: int, seed: int, k_deg: int = 6) -> nx.DiGraph:
    if (k_deg * n_nodes) % 2 != 0:
        k_deg += 1
    g_und = nx.random_regular_graph(k_deg, n_nodes, seed=seed)
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    for u, v in g_und.edges():
        g.add_edge(u, v)
        g.add_edge(v, u)
    if not nx.is_strongly_connected(g):
        rng = random.Random(seed)
        sccs = list(nx.strongly_connected_components(g))
        reps = [rng.choice(list(scc)) for scc in sccs]
        for i in range(len(reps)):
            u = reps[i]; v = reps[(i + 1) % len(reps)]
            if u != v and not g.has_edge(u, v):
                g.add_edge(u, v)
    return g


ALL_GRAPH_TYPES = MAIN_GRAPH_TYPES + ["baseline_symmetric"]


def build_graph_and_payloads(graph_type: str, n_nodes: int, seed: int):
    if graph_type == "baseline_symmetric":
        g = build_baseline_graph(n_nodes, seed=seed)
        proto = SymmetricPayload()
        payloads = {i: proto for i in range(n_nodes)}
    else:
        g = build_graph(graph_type, n_nodes, seed=seed)
        payloads = build_state_payloads(g, seed=seed)
    return g, payloads


# ==============================================================================
# 统计工具
# ==============================================================================


def gini(x: np.ndarray) -> float:
    if x.size == 0: return 0.0
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if x.sum() == 0: return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def summary_stats(freqs: List[int]) -> Dict[str, float]:
    arr = np.array(sorted(freqs, reverse=True), dtype=np.int64)
    if arr.size == 0:
        return {"n_unique": 0, "top1": 0, "median": 0.0, "max_over_median": 0.0,
                "skew": 0.0, "gini": 0.0, "top10pct_share": 0.0, "singleton_fraction": 0.0,
                "mean": 0.0, "std": 0.0, "total": 0}
    total = int(arr.sum())
    med = float(np.median(arr))
    top1 = int(arr[0])
    mx_med = (float(arr[0]) / med) if med > 0 else float("inf")
    skew = float(spstats.skew(arr)) if arr.size > 1 else 0.0
    g = float(gini(arr))
    top10n = max(1, int(np.ceil(arr.size * 0.1)))
    top10_share = float(arr[:top10n].sum()) / max(1, total)
    singletons = float((arr == 1).sum()) / arr.size
    return {"n_unique": int(arr.size), "top1": top1, "median": med, "max_over_median": mx_med,
            "skew": skew, "gini": g, "top10pct_share": top10_share,
            "singleton_fraction": singletons, "mean": float(arr.mean()),
            "std": float(arr.std()), "total": total}


# ==============================================================================
# 单 run
# ==============================================================================


def run_one(graph_type: str, n_nodes: int, seed: int, n_steps: int,
            reservoir_capacity: int, top_k_retrieve: int,
            results_dir: str, write_mem_timeseries: bool) -> Dict[str, Any]:
    """
    单 run: 生成图, 跑 random-walk, 写下游需要的产物 (per-id counts + mem timeseries + summary).

    write_mem_timeseries: 若 True 写 memory-access 时序 (agent_step, memory_id, access_kind, retrieval_rank).
                          若 False 只写频次 counts (节省 IO).
    """
    t0 = time.time()
    g, payloads = build_graph_and_payloads(graph_type, n_nodes, seed=seed)

    rng_walk = random.Random(seed + 100)
    rng_mem = random.Random(seed + 200)

    neighbors_cache = {n: list(g.successors(n)) for n in g.nodes()}
    action_cache = {n: payloads[n].actions for n in g.nodes()}

    mem = StateAwareReservoirMemory(capacity=reservoir_capacity, rng=rng_mem)

    state_counter: Counter = Counter()
    trans_counter: Counter = Counter()

    # 每半段计数, 用于时间稳定性
    half = n_steps // 2
    state_first: Counter = Counter()
    state_second: Counter = Counter()
    mem_first: Counter = Counter()
    mem_second: Counter = Counter()

    # memory 访问时序 (仅 access_kind=read, 用于 PSD)
    # 我们记 (agent_step, memory_id_index) 二元组; memory_id_index = 一个全局递增 id, 用于 PSD 输入
    mem_time_records: List[Tuple[int, str, str, int]] = []  # (step, mid, kind, rank_or_-1)

    current = rng_walk.choice(list(g.nodes()))
    for step in range(n_steps):
        sid = f"v_{current:04d}"
        state_counter[sid] += 1
        if step < half:
            state_first[sid] += 1
        else:
            state_second[sid] += 1

        if step + 1 < n_steps:
            neighbors = neighbors_cache[current]
            actions = action_cache[current]
            action = rng_walk.choice(actions)
            nxt_node = rng_walk.choice(neighbors)
            nxt_sid = f"v_{nxt_node:04d}"
            tid = f"{sid}::{action}::{nxt_sid}"

            mid = f"tx_{tid}"
            content = f"transition {sid}--{action}-->{nxt_sid}"
            for ev in mem.write(mid, content, prev=sid, action=action, nxt=nxt_sid, step=step):
                if step < half:
                    mem_first[ev["memory_id"]] += 1
                else:
                    mem_second[ev["memory_id"]] += 1
                if write_mem_timeseries:
                    mem_time_records.append((step, ev["memory_id"], ev["access_kind"], -1))

            for ev in mem.retrieve(current_state=sid, k=top_k_retrieve, step=step):
                if step < half:
                    mem_first[ev["memory_id"]] += 1
                else:
                    mem_second[ev["memory_id"]] += 1
                if write_mem_timeseries:
                    mem_time_records.append((step, ev["memory_id"], ev["access_kind"], ev["retrieval_rank"]))

            trans_counter[tid] += 1
            current = nxt_node

    state_freqs = list(state_counter.values())
    trans_freqs = list(trans_counter.values())
    mem_freqs = list(mem.access_counter.values())

    # ============ 落盘 ============
    os.makedirs(results_dir, exist_ok=True)
    tag = f"{graph_type}_v{n_nodes}_s{seed}"

    # per-id count files (下游 SOC 分析)
    with open(os.path.join(results_dir, f"{tag}_state_counts.json"), "w") as f:
        json.dump(dict(state_counter), f)
    with open(os.path.join(results_dir, f"{tag}_trans_counts.json"), "w") as f:
        json.dump(dict(trans_counter), f)
    with open(os.path.join(results_dir, f"{tag}_mem_counts.json"), "w") as f:
        json.dump(dict(mem.access_counter), f)

    # temporal half counts (时间稳定性用)
    with open(os.path.join(results_dir, f"{tag}_state_first_half.json"), "w") as f:
        json.dump(dict(state_first), f)
    with open(os.path.join(results_dir, f"{tag}_state_second_half.json"), "w") as f:
        json.dump(dict(state_second), f)
    with open(os.path.join(results_dir, f"{tag}_mem_first_half.json"), "w") as f:
        json.dump(dict(mem_first), f)
    with open(os.path.join(results_dir, f"{tag}_mem_second_half.json"), "w") as f:
        json.dump(dict(mem_second), f)

    # memory 时序 (可选, 只在 gt=scale_free 上启用, 全跑太大)
    if write_mem_timeseries:
        with open(os.path.join(results_dir, f"{tag}_mem_time.jsonl"), "w") as f:
            for (st, mid, kind, rank) in mem_time_records:
                f.write(json.dumps({"step": st, "mid": mid, "kind": kind, "rank": rank}) + "\n")

    # summary stats
    st = summary_stats(state_freqs)
    tr = summary_stats(trans_freqs)
    me = summary_stats(mem_freqs)
    st_first = summary_stats(list(state_first.values()))
    st_second = summary_stats(list(state_second.values()))
    me_first = summary_stats(list(mem_first.values()))
    me_second = summary_stats(list(mem_second.values()))

    elapsed = time.time() - t0

    meta = {
        "graph_type": graph_type, "n_nodes": n_nodes, "seed": seed, "n_steps": n_steps,
        "reservoir_capacity": reservoir_capacity, "top_k_retrieve": top_k_retrieve,
        "elapsed_sec": elapsed,
        "state_stats": st, "trans_stats": tr, "mem_stats": me,
        "temporal_stability": {
            "state_first_half": st_first, "state_second_half": st_second,
            "mem_first_half": me_first, "mem_second_half": me_second,
        },
        "memory_bookkeeping": {
            "n_write_events": mem.write_events,
            "n_read_events": mem.read_events,
            "n_evict_events": mem.evict_events,
            "n_conflicts": mem.conflicts,
            "n_slots_used": len(mem.slots),
        },
        "artifacts": {
            "state_counts": f"{tag}_state_counts.json",
            "trans_counts": f"{tag}_trans_counts.json",
            "mem_counts": f"{tag}_mem_counts.json",
            "state_first_half": f"{tag}_state_first_half.json",
            "state_second_half": f"{tag}_state_second_half.json",
            "mem_first_half": f"{tag}_mem_first_half.json",
            "mem_second_half": f"{tag}_mem_second_half.json",
            "mem_time": f"{tag}_mem_time.jsonl" if write_mem_timeseries else None,
        },
        "graph_edges": g.number_of_edges(),
    }
    with open(os.path.join(results_dir, f"{tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


# ==============================================================================
# 全局主入口
# ==============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=100_000)
    parser.add_argument("--reservoir_capacity", type=int, default=200)
    parser.add_argument("--top_k_retrieve", type=int, default=3)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--graph_types", type=str, nargs="+", default=None,
                        help="限制图类型 (subset of ALL_GRAPH_TYPES)")
    parser.add_argument("--n_nodes", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--write_mem_timeseries_for", type=str, nargs="+",
                        default=["scale_free"],
                        help="哪些图类型写 mem_time.jsonl (为 PSD 用). 全写会几十 GB")
    args = parser.parse_args()

    results_dir = os.path.join(args.out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    gts = args.graph_types if args.graph_types else ALL_GRAPH_TYPES
    print(f"[SETUP] graph_types={gts} | n_nodes={args.n_nodes} | seeds={args.seeds}", flush=True)
    print(f"[SETUP] N={args.n_steps} M={args.reservoir_capacity} k={args.top_k_retrieve}", flush=True)
    print(f"[SETUP] mem_time_for={args.write_mem_timeseries_for}", flush=True)

    all_meta: List[Dict[str, Any]] = []
    combos = [(gt, nn, s) for gt in gts for nn in args.n_nodes for s in args.seeds]
    t_all_start = time.time()
    for i, (gt, nn, s) in enumerate(combos):
        write_ts = gt in args.write_mem_timeseries_for
        print(f"[{i+1}/{len(combos)}] {gt} v{nn} s{s} (mem_ts={write_ts})", flush=True)
        try:
            meta = run_one(gt, nn, s, args.n_steps,
                           args.reservoir_capacity, args.top_k_retrieve,
                           results_dir=results_dir, write_mem_timeseries=write_ts)
            all_meta.append(meta)
            m = meta["mem_stats"]
            print(f"    mem: gini={m['gini']:.3f} skew={m['skew']:.2f} max/med={m['max_over_median']:.1f} n_uniq={m['n_unique']}   [{meta['elapsed_sec']:.1f}s]", flush=True)
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            all_meta.append({"error": str(e), "graph_type": gt, "n_nodes": nn, "seed": s})

    # summary
    summary = {
        "study": "random_walk_scaling",
        "config": {
            "n_steps": args.n_steps, "graph_types": gts, "n_nodes": args.n_nodes,
            "seeds": args.seeds, "reservoir_capacity": args.reservoir_capacity,
            "top_k_retrieve": args.top_k_retrieve,
            "write_mem_timeseries_for": args.write_mem_timeseries_for,
        },
        "n_runs": len(combos), "elapsed_total_sec": time.time() - t_all_start,
        "runs": all_meta,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] {len(combos)} runs, total {summary['elapsed_total_sec']:.1f}s")


if __name__ == "__main__":
    main()
