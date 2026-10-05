from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_random_walk_scaling import build_graph_and_payloads
from worldmodelsoc.experiment_io import read_records, write_json
from worldmodelsoc.synthetic_experiments import run_synthetic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["synthetic", "ablation", "tau", "replay"], default="synthetic")
    parser.add_argument("--methods", nargs="+", choices=["full_history", "flat_retrieval", "graph_memory", "ctmc"], default=["full_history", "flat_retrieval", "graph_memory", "ctmc"])
    parser.add_argument("--serializations", nargs="+", choices=["verbose", "compact"], default=["compact"])
    parser.add_argument("--graph_types", nargs="+", default=["scale_free"])
    parser.add_argument("--n_nodes", nargs="+", type=int, default=[100])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--taus", nargs="+", type=float, default=[1.0])
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--core_size", type=int, default=3)
    parser.add_argument("--memory_budget", type=int, default=512)
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--policy", choices=["random", "semantic"], default="semantic")
    parser.add_argument("--model")
    parser.add_argument("--max_calls", type=int, default=10000)
    parser.add_argument("--replay_actions", type=Path)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.n_steps < 1 or any(size < 6 for size in args.n_nodes):
        parser.error("n_steps must be positive and graph sizes must be at least 6")
    if args.experiment == "ablation":
        args.methods = ["graph_memory", "ctmc"]
        args.serializations = ["verbose", "compact"]
    if args.experiment == "tau":
        args.methods = ["ctmc"]
        if args.taus == [1.0]:
            args.taus = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    replay = None
    if args.experiment == "replay":
        if args.replay_actions is None:
            parser.error("replay requires --replay_actions")
        replay = read_records(args.replay_actions)
        args.methods = ["ctmc"]
        args.serializations = ["verbose", "compact"]
        args.seeds = args.seeds[:1]
        args.graph_types = ["frozen"]
        args.n_nodes = [0]
    args.out_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for graph_type, size, seed, tau, method, serialization in itertools.product(
        args.graph_types, args.n_nodes, args.seeds, args.taus, args.methods, args.serializations,
    ):
        tag = f"{graph_type}_n{size}_s{seed}_tau{tau:g}_{method}_{serialization}"
        graph, payloads = (None, None) if replay is not None else build_graph_and_payloads(graph_type, size, seed)
        config = {
            "experiment": args.experiment, "graph_type": graph_type, "n_nodes": size,
            "seed": seed, "n_steps": args.n_steps, "tau": tau, "method": method,
            "serialization": serialization, "capacity": args.capacity, "core_size": args.core_size,
            "memory_budget": args.memory_budget, "tokenizer": args.tokenizer, "policy": args.policy,
            "model": args.model, "max_calls": args.max_calls, "out_dir": str(args.out_dir / tag),
            "replay_actions": str(args.replay_actions) if args.replay_actions else None,
        }
        result = run_synthetic(config, graph, payloads, replay)
        results.append(result)
        write_json(args.out_dir / "runs.json", results)


if __name__ == "__main__":
    main()
