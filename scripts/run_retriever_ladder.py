from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_random_walk_scaling import build_graph_and_payloads
from worldmodelsoc.env.synthetic_graph_world import run_random_walk
from worldmodelsoc.experiment_io import provenance, write_json
from worldmodelsoc.memory.reservoir import StateAwareReservoirMemory, summary_stats


class ControlledReservoir(StateAwareReservoirMemory):
    def __init__(self, capacity, rng, mode):
        super().__init__(capacity, rng)
        self.mode = mode

    def _evict_if_needed(self):
        if self.mode in ("uniform_frequency", "state_uniform"):
            if len(self.slots) >= self.capacity:
                del self.slots[self.rng.choice(sorted(self.slots))]
                self.evict_events += 1
        else:
            super()._evict_if_needed()

    def retrieve(self, current_state, k=3, step=0):
        if self.mode == "no_retrieval":
            return []
        candidates = list(self.slots.items())
        if self.mode != "uniform_frequency":
            local = [(key, value) for key, value in candidates if current_state in (value["prev"], value["next"])]
            candidates = local or candidates
        if self.mode == "state_uniform":
            candidates = self.rng.sample(candidates, min(k, len(candidates)))
        else:
            candidates.sort(key=lambda item: (-item[1]["freq"], item[0]))
            candidates = candidates[:k]
        events = []
        for rank, (key, value) in enumerate(candidates, 1):
            self.access_counter[key] += 1
            self.read_events += 1
            events.append({"memory_id": key, "rank": rank, "content": value["content"], "step": step})
        return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_types", nargs="+", default=["uniform_degree", "exponential_degree", "scale_free", "modular", "mixed", "baseline_symmetric"])
    parser.add_argument("--n_nodes", nargs="+", type=int, default=[100, 500, 1000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n_steps", type=int, default=100000)
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=["uniform_frequency", "state_frequency", "no_retrieval", "state_uniform"], default=["uniform_frequency", "state_frequency", "no_retrieval", "state_uniform"])
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
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
            row = {"graph_type": graph_type, "n_nodes": size, "seed": seed, "mode": mode, "paired_trajectory": True, "retrieval_precedes_update": True, "memory_stats": summary_stats(list(counts.values())), "read_stats": summary_stats(list(reads.values()))}
            summaries.append(row)
            write_json(path / "summary.json", row)
    write_json(args.out_dir / "summary.json", summaries)


if __name__ == "__main__":
    main()
