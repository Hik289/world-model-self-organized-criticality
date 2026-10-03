from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any

import networkx as nx
import numpy as np


GRAPH_TYPES = [
    "uniform_degree",
    "exponential_degree",
    "scale_free",
    "modular",
    "mixed",
]


def _ensure_strongly_connected_di(g: nx.DiGraph, rng: random.Random) -> nx.DiGraph:
    if nx.is_strongly_connected(g):
        return g
    sccs = list(nx.strongly_connected_components(g))
    scc_reps = []
    for scc in sccs:
        scc_list = list(scc)
        scc_reps.append(rng.choice(scc_list))
    for i in range(len(scc_reps)):
        u = scc_reps[i]
        v = scc_reps[(i + 1) % len(scc_reps)]
        if u == v:
            continue
        if not g.has_edge(u, v):
            g.add_edge(u, v)
    return g


def build_graph(graph_type: str, n_nodes: int, seed: int = 42) -> nx.DiGraph:
    if graph_type not in GRAPH_TYPES:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    if n_nodes < 6:
        raise ValueError("n_nodes must be at least 6")
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if graph_type == "uniform_degree":
        k = 6 if n_nodes >= 20 else max(2, n_nodes // 4 * 2)
        if (k * n_nodes) % 2 != 0:
            k += 1
        g_und = nx.random_regular_graph(k, n_nodes, seed=seed)
        g = nx.DiGraph()
        g.add_nodes_from(range(n_nodes))
        for u, v in g_und.edges():
            g.add_edge(u, v)
            g.add_edge(v, u)

    elif graph_type == "exponential_degree":
        target_mean = 6.0
        while True:
            degs = np_rng.exponential(scale=target_mean, size=n_nodes).astype(int) + 1
            if degs.sum() % 2 == 0:
                break
        g_und = nx.configuration_model(degs.tolist(), seed=seed)
        g_und = nx.Graph(g_und)
        g_und.remove_edges_from(nx.selfloop_edges(g_und))
        g = nx.DiGraph()
        g.add_nodes_from(range(n_nodes))
        for u, v in g_und.edges():
            g.add_edge(u, v)
            g.add_edge(v, u)

    elif graph_type == "scale_free":
        m = min(3, max(1, n_nodes // 30))
        g_und = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
        g = nx.DiGraph()
        g.add_nodes_from(range(n_nodes))
        for u, v in g_und.edges():
            g.add_edge(u, v)
            g.add_edge(v, u)

    elif graph_type == "modular":
        n_blocks = 5 if n_nodes >= 50 else 2
        block_size = n_nodes // n_blocks
        sizes = [block_size] * n_blocks
        sizes[-1] += n_nodes - sum(sizes)
        p_in = 0.15
        p_out = 0.01
        p_matrix = [[p_in if i == j else p_out for j in range(n_blocks)] for i in range(n_blocks)]
        g_und = nx.stochastic_block_model(sizes, p_matrix, seed=seed)
        g = nx.DiGraph()
        g.add_nodes_from(range(n_nodes))
        for u, v in g_und.edges():
            g.add_edge(u, v)
            g.add_edge(v, u)

    elif graph_type == "mixed":
        half = n_nodes // 2
        m = min(3, max(1, half // 30))
        g_ba = nx.barabasi_albert_graph(half, m, seed=seed)
        n_blocks = 3
        block_size = (n_nodes - half) // n_blocks
        sizes = [block_size] * n_blocks
        sizes[-1] += (n_nodes - half) - sum(sizes)
        p_matrix = [[0.15 if i == j else 0.01 for j in range(n_blocks)] for i in range(n_blocks)]
        g_sbm = nx.stochastic_block_model(sizes, p_matrix, seed=seed + 1)
        g = nx.DiGraph()
        g.add_nodes_from(range(n_nodes))
        for u, v in g_ba.edges():
            g.add_edge(u, v)
            g.add_edge(v, u)
        for u, v in g_sbm.edges():
            uu = u + half
            vv = v + half
            g.add_edge(uu, vv)
            g.add_edge(vv, uu)
        for _ in range(5):
            a = rng.randrange(half)
            b = rng.randrange(half, n_nodes)
            g.add_edge(a, b)
            g.add_edge(b, a)

    g = _ensure_strongly_connected_di(g, rng)

    for node in list(g.nodes()):
        if g.out_degree(node) == 0:
            other = rng.choice([n for n in g.nodes() if n != node])
            g.add_edge(node, other)

    return g


@dataclass
class StatePayload:
    entities: List[str] = field(default_factory=list)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)


def build_state_payloads(g: nx.DiGraph, seed: int = 42) -> Dict[int, StatePayload]:
    rng = random.Random(seed + 1)
    payloads: Dict[int, StatePayload] = {}
    entity_pool = [f"ent_{i}" for i in range(200)]
    rel_pool = ["hasPart", "connectedTo", "adjacentTo", "similarTo", "controls", "contains"]
    constraint_pool = [
        "temperature<50",
        "battery>=0.2",
        "count<=10",
        "unlocked==True",
        "safe_mode==False",
        "queue_empty==True",
        "cooldown<3",
    ]
    for node in g.nodes():
        n_ent = rng.randint(3, 8)
        entities = rng.sample(entity_pool, n_ent)
        n_rel = rng.randint(2, 6)
        relations: List[Tuple[str, str, str]] = []
        for _ in range(n_rel):
            h = rng.choice(entities)
            t = rng.choice(entities)
            r = rng.choice(rel_pool)
            relations.append((h, r, t))
        n_cons = rng.randint(1, 4)
        constraints = rng.sample(constraint_pool, min(n_cons, len(constraint_pool)))
        out_deg = g.out_degree(node)
        n_act = rng.randint(1, 3)
        n_act = min(n_act, max(1, out_deg))
        actions = [f"act_{i}" for i in range(n_act)]
        payloads[node] = StatePayload(
            entities=entities, relations=relations,
            constraints=constraints, actions=actions,
        )
    return payloads


def neighbor_segment(neighbors: List[int], n_actions: int, action_idx: int) -> List[int]:
    if n_actions < 1:
        raise ValueError("n_actions must be at least 1")
    if not 0 <= action_idx < n_actions:
        raise ValueError("action_idx is out of range")
    if not neighbors:
        return []
    if n_actions > len(neighbors):
        raise ValueError("n_actions cannot exceed the number of neighbors")
    segment_size = max(1, len(neighbors) // n_actions)
    start = action_idx * segment_size
    end = start + segment_size if action_idx < n_actions - 1 else len(neighbors)
    segment = neighbors[start:end] if start < len(neighbors) else neighbors
    return segment or neighbors


def action_index_for_neighbor(
    neighbors: List[int],
    n_actions: int,
    neighbor: int,
) -> int:
    if neighbor not in neighbors:
        raise ValueError("neighbor is not in the supplied neighbor list")
    for action_idx in range(n_actions):
        if neighbor in neighbor_segment(neighbors, n_actions, action_idx):
            return action_idx
    raise RuntimeError("no action segment contains the selected neighbor")


def describe_action_options(
    g: nx.DiGraph,
    payloads: Dict[int, StatePayload],
    node: int,
) -> List[str]:
    neighbors = list(g.successors(node))
    actions = payloads[node].actions
    descriptions = []
    for action_idx, action in enumerate(actions):
        targets = neighbor_segment(neighbors, len(actions), action_idx)
        summaries = []
        for target in targets[:3]:
            payload = payloads[target]
            summaries.append(
                f"{target}: entities={payload.entities[:2]}, "
                f"constraints={payload.constraints[:2]}"
            )
        if len(targets) > 3:
            summaries.append(f"+{len(targets) - 3} more")
        descriptions.append(f"{action} -> " + "; ".join(summaries))
    return descriptions


def run_random_walk(
    g: nx.DiGraph,
    payloads: Dict[int, StatePayload],
    n_steps: int,
    seed: int = 42,
    start_node: int | None = None,
) -> Tuple[List[int], List[Tuple[int, str, int]]]:
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if not g:
        raise ValueError("g must contain at least one node")
    if start_node is not None and start_node not in g:
        raise ValueError("start_node is not present in g")
    missing_payloads = set(g) - set(payloads)
    if missing_payloads:
        raise ValueError(f"missing payloads for {len(missing_payloads)} nodes")
    rng = random.Random(seed + 2)
    if start_node is None:
        start_node = rng.choice(list(g.nodes()))
    states = [start_node]
    transitions: List[Tuple[int, str, int]] = []
    current = start_node
    neighbors_cache: Dict[int, List[int]] = {n: list(g.successors(n)) for n in g.nodes()}
    action_cache: Dict[int, List[str]] = {n: payloads[n].actions for n in g.nodes()}

    for _step in range(n_steps - 1):
        neighbors = neighbors_cache[current]
        actions = action_cache[current]
        nxt = rng.choice(neighbors)
        action_idx = action_index_for_neighbor(neighbors, len(actions), nxt)
        action = actions[action_idx]
        transitions.append((current, action, nxt))
        states.append(nxt)
        current = nxt

    return states, transitions


@dataclass
class AssertionReport:
    graph_type: str
    n_nodes: int
    n_steps: int
    seed: int
    A1_nonzero_state_coverage: float
    A1_pass: bool
    A2_max_state_freq_ratio: float
    A2_pass: bool
    A3_unique_transitions_count: int
    A3_threshold: int
    A3_pass: bool
    A4_state_freq_variance: float
    A4_transition_freq_variance: float
    A4_pass: bool
    A5_generation_time_sec: float
    A5_pass: bool
    all_pass: bool
    n_singleton_states: int
    n_singleton_transitions: int
    top10_state_freq: List[int]
    top10_transition_freq: List[int]
    n_unique_states_visited: int
    n_unique_transitions: int
    graph_edge_count: int


def evaluate_run(
    graph_type: str,
    n_nodes: int,
    n_steps: int,
    seed: int,
    states: List[int],
    transitions: List[Tuple[int, str, int]],
    graph_edge_count: int,
    gen_time: float,
) -> AssertionReport:
    unique_states = set(states)
    coverage = len(unique_states) / n_nodes
    A1_pass = coverage >= 0.2

    state_counter = Counter(states)
    max_freq = max(state_counter.values()) if state_counter else 0
    max_ratio = max_freq / n_steps
    A2_pass = max_ratio <= 0.5

    trans_ids = [f"{u}::{a}::{v}" for (u, a, v) in transitions]
    trans_counter = Counter(trans_ids)
    n_unique_trans = len(trans_counter)
    A3_threshold = 3 * n_nodes
    A3_pass = n_unique_trans >= A3_threshold

    state_freqs = np.array(list(state_counter.values()), dtype=np.int64)
    trans_freqs = np.array(list(trans_counter.values()), dtype=np.int64)
    state_var = float(state_freqs.var())
    trans_var = float(trans_freqs.var())
    A4_pass = (state_var > 0) and (trans_var > 0)

    A5_pass = gen_time <= 120.0

    all_pass = all([A1_pass, A2_pass, A3_pass, A4_pass, A5_pass])

    top10_state = [c for _, c in state_counter.most_common(10)]
    top10_trans = [c for _, c in trans_counter.most_common(10)]

    n_singleton_states = int(sum(1 for c in state_counter.values() if c == 1))
    n_singleton_trans = int(sum(1 for c in trans_counter.values() if c == 1))

    return AssertionReport(
        graph_type=graph_type,
        n_nodes=n_nodes,
        n_steps=n_steps,
        seed=seed,
        A1_nonzero_state_coverage=coverage,
        A1_pass=A1_pass,
        A2_max_state_freq_ratio=max_ratio,
        A2_pass=A2_pass,
        A3_unique_transitions_count=n_unique_trans,
        A3_threshold=A3_threshold,
        A3_pass=A3_pass,
        A4_state_freq_variance=state_var,
        A4_transition_freq_variance=trans_var,
        A4_pass=A4_pass,
        A5_generation_time_sec=gen_time,
        A5_pass=A5_pass,
        all_pass=all_pass,
        n_singleton_states=n_singleton_states,
        n_singleton_transitions=n_singleton_trans,
        top10_state_freq=top10_state,
        top10_transition_freq=top10_trans,
        n_unique_states_visited=len(unique_states),
        n_unique_transitions=n_unique_trans,
        graph_edge_count=graph_edge_count,
    )


def emit_sample_jsonl(
    graph_type: str,
    n_nodes: int,
    seed: int,
    payloads: Dict[int, StatePayload],
    states: List[int],
    transitions: List[Tuple[int, str, int]],
    out_path: str,
    short_len: int = 500,
    run_id: str | None = None,
    start_time_utc: str | None = None,
    start_wallclock_ms: int = 0,
) -> str:
    import datetime as dt
    if run_id is None:
        run_id = f"sgw_{graph_type}_v{n_nodes}_seed{seed}_sample"
    if start_time_utc is None:
        start_time_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    state_seq = 0
    trans_seq = 0
    token_seq = 0
    meta_seq = 0

    state_freq_running: Dict[int, int] = {}
    trans_freq_running: Dict[str, int] = {}

    with open(out_path, "w", encoding="utf-8") as f:
        rec = {
            "schema_version": "0.1.0",
            "run_id": run_id,
            "benchmark": "synthetic_graph_world",
            "task_id": "stream_0",
            "agent_step": 0,
            "event_type": "meta",
            "event_seq": meta_seq,
            "timestamp_utc": start_time_utc,
            "wallclock_ms_since_run_start": start_wallclock_ms,
            "note": f"sample graph-world JSONL, graph_type={graph_type}, n_nodes={n_nodes}",
            "kind": "run_start",
        }
        f.write(json.dumps(rec) + "\n")
        meta_seq += 1

        L = min(short_len, len(states))
        for step in range(L):
            wc_ms = start_wallclock_ms + step * 10

            s = states[step]
            state_freq_running[s] = state_freq_running.get(s, 0) + 1
            novelty = (state_freq_running[s] == 1)
            state_rec = {
                "schema_version": "0.1.0",
                "run_id": run_id,
                "benchmark": "synthetic_graph_world",
                "task_id": "stream_0",
                "agent_step": step,
                "event_type": "state",
                "event_seq": state_seq,
                "timestamp_utc": start_time_utc,
                "wallclock_ms_since_run_start": wc_ms,
                "state_id": f"v_{s:04d}",
                "state_source": "env_ground_truth",
                "state_novelty_flag": novelty,
                "state_freq_running": state_freq_running[s],
                "state_context_tokens": len(payloads[s].entities) * 4 + len(payloads[s].relations) * 8,
            }
            f.write(json.dumps(state_rec) + "\n")
            state_seq += 1

            if step < len(transitions):
                u, a, v = transitions[step]
                tid = f"v_{u:04d}::{a}::v_{v:04d}"
                trans_freq_running[tid] = trans_freq_running.get(tid, 0) + 1
                tnov = (trans_freq_running[tid] == 1)
                trans_rec = {
                    "schema_version": "0.1.0",
                    "run_id": run_id,
                    "benchmark": "synthetic_graph_world",
                    "task_id": "stream_0",
                    "agent_step": step,
                    "event_type": "transition",
                    "event_seq": trans_seq,
                    "timestamp_utc": start_time_utc,
                    "wallclock_ms_since_run_start": wc_ms + 1,
                    "transition_id": tid,
                    "state_id_prev": f"v_{u:04d}",
                    "action": a,
                    "state_id_next": f"v_{v:04d}",
                    "transition_source": "env_ground_truth",
                    "transition_freq_running": trans_freq_running[tid],
                    "transition_novelty_flag": tnov,
                }
                f.write(json.dumps(trans_rec) + "\n")
                trans_seq += 1

            if step % 100 == 0:
                tok_rec = {
                    "schema_version": "0.1.0",
                    "run_id": run_id,
                    "benchmark": "synthetic_graph_world",
                    "task_id": "stream_0",
                    "agent_step": step,
                    "event_type": "token_profile",
                    "event_seq": token_seq,
                    "timestamp_utc": start_time_utc,
                    "wallclock_ms_since_run_start": wc_ms + 2,
                    "tokens_prompt": 0,
                    "tokens_completion": 0,
                    "tokens_memory_in_context": 0,
                    "api_calls": 0,
                    "budget_remaining": None,
                }
                f.write(json.dumps(tok_rec) + "\n")
                token_seq += 1

        rec = {
            "schema_version": "0.1.0",
            "run_id": run_id,
            "benchmark": "synthetic_graph_world",
            "task_id": "stream_0",
            "agent_step": L - 1,
            "event_type": "meta",
            "event_seq": meta_seq,
            "timestamp_utc": start_time_utc,
            "wallclock_ms_since_run_start": start_wallclock_ms + L * 10 + 5,
            "note": f"end of sample JSONL, {L} steps",
            "kind": "run_end",
        }
        f.write(json.dumps(rec) + "\n")

    return run_id


def run_one_combo(graph_type: str, n_nodes: int, n_steps: int, seed: int,
                  sample_out_path: str | None = None) -> AssertionReport:
    t0 = time.time()
    g = build_graph(graph_type, n_nodes, seed=seed)
    payloads = build_state_payloads(g, seed=seed)
    states, transitions = run_random_walk(g, payloads, n_steps, seed=seed)
    gen_time = time.time() - t0

    if sample_out_path:
        emit_sample_jsonl(
            graph_type, n_nodes, seed, payloads, states, transitions,
            out_path=sample_out_path, short_len=500,
        )

    report = evaluate_run(
        graph_type=graph_type,
        n_nodes=n_nodes,
        n_steps=n_steps,
        seed=seed,
        states=states,
        transitions=transitions,
        graph_edge_count=g.number_of_edges(),
        gen_time=gen_time,
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--emit_samples", action="store_true",
                        help="每 graph_type 生成一个 500 步 sample JSONL (只在 |V|=100 组合).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logs_dir = os.path.join(args.out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    combos: List[Tuple[str, int]] = []
    for gt in GRAPH_TYPES:
        for n in [100, 500, 1000]:
            combos.append((gt, n))

    all_reports: List[Dict[str, Any]] = []
    for (gt, n) in combos:
        sample_path = None
        if args.emit_samples and n == 100:
            sample_path = os.path.join(logs_dir, f"sample_{gt}_v{n}_seed{args.seed}.jsonl")
        print(f"[RUN] graph_type={gt} n_nodes={n} n_steps={args.n_steps} seed={args.seed}", flush=True)
        rep = run_one_combo(gt, n, args.n_steps, args.seed, sample_path)
        rec = asdict(rep)
        rec["sample_jsonl_path"] = sample_path
        all_reports.append(rec)
        print(f"  A1={rep.A1_pass}({rep.A1_nonzero_state_coverage:.3f})  "
              f"A2={rep.A2_pass}({rep.A2_max_state_freq_ratio:.4f})  "
              f"A3={rep.A3_pass}({rep.A3_unique_transitions_count}/{rep.A3_threshold})  "
              f"A4={rep.A4_pass}(sv={rep.A4_state_freq_variance:.2f}, tv={rep.A4_transition_freq_variance:.2f})  "
              f"A5={rep.A5_pass}({rep.A5_generation_time_sec:.1f}s)  "
              f"all_pass={rep.all_pass}",
              flush=True)

    results_path = os.path.join(args.out_dir, "results.json")
    out_obj = {
        "study": "synthetic_graph_world_checks",
        "n_steps_per_run": args.n_steps,
        "seed": args.seed,
        "n_combos": len(combos),
        "n_all_pass": sum(1 for r in all_reports if r["all_pass"]),
        "reports": all_reports,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] wrote {results_path}", flush=True)
    print(f"       {out_obj['n_all_pass']} / {out_obj['n_combos']} combos passed all assertions", flush=True)


if __name__ == "__main__":
    main()
