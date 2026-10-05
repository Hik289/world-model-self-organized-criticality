from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldmodelsoc.experiment_io import paired_summary, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", default="graph_memory")
    parser.add_argument("--method", default="ctmc")
    parser.add_argument("--metrics", nargs="+", default=["prompt_tokens_per_step", "tail_prediction_error", "retained_state_coverage", "retained_transition_coverage"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--compare_serializations", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    rows = []
    for path in args.runs:
        data = json.loads(path.read_text(encoding="utf-8"))
        for run in data if isinstance(data, list) else [data]:
            config = run["config"]
            keys = ("graph_type", "n_nodes", "seed", "n_steps", "tau", "memory_budget", "core_size", "capacity", "policy", "model", "tokenizer", "replay_actions")
            pair = {key: config.get(key) for key in keys}
            if args.compare_serializations:
                pair["method"] = config["method"]
                name = config["serialization"]
            else:
                pair["serialization"] = config["serialization"]
                name = config["method"]
            rows.append({"pair_id": json.dumps(pair, sort_keys=True), "method": name, **run["metrics"]})
    write_json(args.output, paired_summary(rows, args.baseline, args.method, args.metrics, args.seed, args.resamples))


if __name__ == "__main__":
    main()
