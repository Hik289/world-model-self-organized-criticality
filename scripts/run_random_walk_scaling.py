from __future__ import annotations

import argparse
import itertools
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
from worldmodelsoc.memory.reservoir import StateAwareReservoirMemory, ControlledReservoir, audit_counts, psd_audit
from worldmodelsoc.memory.reservoir import summary_stats as memory_summary_stats
from worldmodelsoc.env.synthetic_graph_world import (
    action_index_for_neighbor,
    build_graph,
    build_state_payloads,
    run_random_walk,
    GRAPH_TYPES as MAIN_GRAPH_TYPES,
)
from worldmodelsoc.pipeline.modules import provenance, read_records, write_json


@dataclass
class SymmetricPayload:
    entities: List[str] = field(default_factory=lambda: ["e1", "e2", "e3"])
    relations: List[Tuple[str, str, str]] = field(default_factory=lambda: [("e1", "r1", "e2")])
    constraints: List[str] = field(default_factory=lambda: ["c1"])
    actions: List[str] = field(default_factory=lambda: ["a1", "a2", "a3"])


def build_baseline_graph(n_nodes: int, seed: int, k_deg: int = 6) -> nx.DiGraph:
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    if not 0 < k_deg < n_nodes:
        raise ValueError("k_deg must be between 1 and n_nodes - 1")
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
        payloads = dict.fromkeys(range(n_nodes), proto)
    else:
        g = build_graph(graph_type, n_nodes, seed=seed)
        payloads = build_state_payloads(g, seed=seed)
    return g, payloads


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
    skew = (
        float(spstats.skew(arr))
        if arr.size > 1 and np.ptp(arr) > 0
        else 0.0
    )
    g = float(gini(arr))
    top10n = max(1, int(np.ceil(arr.size * 0.1)))
    top10_share = float(arr[:top10n].sum()) / max(1, total)
    singletons = float((arr == 1).sum()) / arr.size
    return {"n_unique": int(arr.size), "top1": top1, "median": med, "max_over_median": mx_med,
            "skew": skew, "gini": g, "top10pct_share": top10_share,
            "singleton_fraction": singletons, "mean": float(arr.mean()),
            "std": float(arr.std()), "total": total}


def run_one(graph_type: str, n_nodes: int, seed: int, n_steps: int,
            reservoir_capacity: int, top_k_retrieve: int,
            results_dir: str, write_mem_timeseries: bool) -> Dict[str, Any]:
    t0 = time.time()
    g, payloads = build_graph_and_payloads(graph_type, n_nodes, seed=seed)

    rng_walk = random.Random(seed + 100)
    rng_mem = random.Random(seed + 200)

    neighbors_cache = {n: list(g.successors(n)) for n in g.nodes()}
    action_cache = {n: payloads[n].actions for n in g.nodes()}

    mem = StateAwareReservoirMemory(capacity=reservoir_capacity, rng=rng_mem)

    state_counter: Counter = Counter()
    trans_counter: Counter = Counter()

    half = n_steps // 2
    state_first: Counter = Counter()
    state_second: Counter = Counter()
    mem_first: Counter = Counter()
    mem_second: Counter = Counter()

    mem_time_records: List[Tuple[int, str, str, int]] = []

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
            nxt_node = rng_walk.choice(neighbors)
            action_idx = action_index_for_neighbor(
                neighbors,
                len(actions),
                nxt_node,
            )
            action = actions[action_idx]
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

    os.makedirs(results_dir, exist_ok=True)
    tag = f"{graph_type}_v{n_nodes}_s{seed}"

    with open(os.path.join(results_dir, f"{tag}_state_counts.json"), "w") as f:
        json.dump(dict(state_counter), f)
    with open(os.path.join(results_dir, f"{tag}_trans_counts.json"), "w") as f:
        json.dump(dict(trans_counter), f)
    with open(os.path.join(results_dir, f"{tag}_mem_counts.json"), "w") as f:
        json.dump(dict(mem.access_counter), f)

    with open(os.path.join(results_dir, f"{tag}_state_first_half.json"), "w") as f:
        json.dump(dict(state_first), f)
    with open(os.path.join(results_dir, f"{tag}_state_second_half.json"), "w") as f:
        json.dump(dict(state_second), f)
    with open(os.path.join(results_dir, f"{tag}_mem_first_half.json"), "w") as f:
        json.dump(dict(mem_first), f)
    with open(os.path.join(results_dir, f"{tag}_mem_second_half.json"), "w") as f:
        json.dump(dict(mem_second), f)

    if write_mem_timeseries:
        with open(os.path.join(results_dir, f"{tag}_mem_time.jsonl"), "w") as f:
            for (st, mid, kind, rank) in mem_time_records:
                f.write(json.dumps({"step": st, "mid": mid, "kind": kind, "rank": rank}) + "\n")

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


def legacy_main(argv=None):
    parser = argparse.ArgumentParser(epilog='Additional modes: --mode retriever-ladder|audit. Append --help for the selected mode.')
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
    args = parser.parse_args(argv)

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


def retriever_ladder_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_types", nargs="+", default=["uniform_degree", "exponential_degree", "scale_free", "modular", "mixed", "baseline_symmetric"])
    parser.add_argument("--n_nodes", nargs="+", type=int, default=[100, 500, 1000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n_steps", type=int, default=100000)
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=["uniform_frequency", "state_frequency", "no_retrieval", "state_uniform"], default=["uniform_frequency", "state_frequency", "no_retrieval", "state_uniform"])
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.n_steps < 2 or args.capacity < 1 or args.top_k < 1:
        parser.error("n_steps must be at least 2; capacity and top_k must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir / "config.json", provenance({**vars(args), "out_dir": str(args.out_dir)}))
    summaries = []
    for graph_type, size, seed in itertools.product(args.graph_types, args.n_nodes, args.seeds):
        graph, payloads = build_graph_and_payloads(graph_type, size, seed)
        states, transitions = run_random_walk(graph, payloads, args.n_steps, seed)
        for mode in args.modes:
            path = args.out_dir / f"{graph_type}_n{size}_s{seed}_{mode}"
            path.mkdir()
            memory = ControlledReservoir(args.capacity, random.Random(seed + 200), mode)
            reads = Counter()
            writes = Counter()
            with (path / "accesses.jsonl").open("w", encoding="utf-8") as handle:
                for step, (prev, action, nxt) in enumerate(transitions):
                    previous, next_state = f"v_{prev:04d}", f"v_{nxt:04d}"
                    events = memory.retrieve(previous, args.top_k, step)
                    reads.update(event["memory_id"] for event in events)
                    key = f"{previous}::{action}::{next_state}"
                    memory.write(key, key, previous, action, next_state, step)
                    writes.update([key])
                    handle.write(json.dumps({"agent_step": step, "state": previous, "retrieved": events, "access_count": len(events) + 1, "write_memory_id": key}) + "\n")
            counts = reads + writes
            write_json(path / "state_counts.json", dict(Counter(f"v_{state:04d}" for state in states)))
            write_json(path / "read_counts.json", dict(reads))
            write_json(path / "write_counts.json", dict(writes))
            write_json(path / "memory_access_counts.json", dict(counts))
            row = {"graph_type": graph_type, "n_nodes": size, "seed": seed, "mode": mode, "paired_trajectory": True, "retrieval_precedes_update": True, "memory_stats": memory_summary_stats(list(counts.values())), "read_stats": memory_summary_stats(list(reads.values()))}
            summaries.append(row)
            write_json(path / "summary.json", row)
    write_json(args.out_dir / "summary.json", summaries)


def audit_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=Path, nargs="+", required=True)
    parser.add_argument("--count_kind", choices=["state_visits", "memory_reads", "memory_accesses", "transition_visits"], required=True)
    parser.add_argument("--field")
    parser.add_argument("--min_tail", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xmin", type=float)
    parser.add_argument("--timeseries", type=Path)
    parser.add_argument("--series_field", default="access_count")
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--min_frequency", type=float)
    parser.add_argument("--max_frequency", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists")
    output = {"provenance": provenance({key: str(value) for key, value in vars(args).items()}), "count_kind": args.count_kind, "audits": []}
    for path in args.counts:
        values = json.loads(path.read_text(encoding="utf-8"))
        if args.field:
            for part in args.field.split("."):
                values = values[part]
        values = list(values.values()) if isinstance(values, dict) else values
        output["audits"].append({"source": str(path), **audit_counts(values, args.min_tail, args.bootstrap, args.seed, args.xmin)})
        write_json(args.output, output)
    if args.timeseries:
        series = read_records(args.timeseries)
        output["temporal_audit"] = {
            "source": str(args.timeseries), "observable": args.series_field,
            **psd_audit([record[args.series_field] for record in series], args.sample_rate, args.min_frequency, args.max_frequency),
        }
    write_json(args.output, output)


def main(argv=None):
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument('--mode', choices=['scaling', 'retriever-ladder', 'audit'], default='scaling')
    selected, remaining = selector.parse_known_args(argv)
    if selected.mode == "retriever-ladder":
        return retriever_ladder_main(remaining)
    if selected.mode == "audit":
        return audit_main(remaining)
    return legacy_main(remaining)


if __name__ == "__main__":
    main()
