from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_ctwm_comparison import run_method
from worldmodelsoc.pipeline.modules import paired_summary, write_json


def legacy_main(argv=None):
    p = argparse.ArgumentParser(epilog='Summarize completed paired runs: --mode summarize --help.')
    p.add_argument("--method", required=True, choices=["B7_GraphMemory", "B8_CTWM"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n_nodes", type=int, default=100)
    p.add_argument("--n_steps", type=int, default=2000)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--budget_usd", type=float, default=0.15)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    run_method(
        method_name=args.method, n_nodes=args.n_nodes, n_steps=args.n_steps,
        seed=args.seed, budget_usd=args.budget_usd, tau=args.tau,
        out_dir=args.out_dir,
    )


def summarize_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", default="graph_memory")
    parser.add_argument("--method", default="ctmc")
    parser.add_argument("--metrics", nargs="+", default=["prompt_tokens_per_step", "tail_prediction_error", "retained_state_coverage", "retained_transition_coverage"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--compare_serializations", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
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


def main(argv=None):
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument('--mode', choices=['run', 'summarize'], default='run')
    selected, remaining = selector.parse_known_args(argv)
    if selected.mode == "summarize":
        return summarize_main(remaining)
    return legacy_main(remaining)


if __name__ == "__main__":
    main()
