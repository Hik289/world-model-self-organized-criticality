"""Replay frozen trajectories under one retrieval implementation.

The replay measures trajectory-specific differences without making API calls.
Optional live metrics may be supplied explicitly for comparison; no empirical
values are embedded in the script.
"""

from __future__ import annotations
import argparse, json, os, sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldmodelsoc.memory.backends_ctwm import B8_CTWM  # noqa: E402


def load_actions(path: str) -> List[Dict[str, Any]]:
    records = []
    required = {"prev", "action", "next"}
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            r = json.loads(line)
            missing = required - set(r)
            if missing:
                raise ValueError(
                    f"{path}:{line_number} is missing fields: {sorted(missing)}"
                )
            records.append(r)
    if not records:
        raise ValueError(f"no action records found in {path}")
    return records


def replay_on_trajectory(
    actions: List[Dict[str, Any]],
    tag: str,
    tau: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Given a frozen list of (prev, action, next) records, replay B8_CTWM memory:
    - Write each transition
    - After each write, retrieve top-1 hint for the current step
    - Compare top-1 hint's next-state against actual next-state -> tail metric

    Returns tail_err (bottom-50% state), plus stats.
    """
    mem = B8_CTWM(
        tau=tau,
        core_pct=0.30,
        core_slots=3,
        tail_slots=2,
        seed=seed,
    )
    per_step_state: List[str] = []
    per_step_correct: List[bool] = []

    for i, r in enumerate(actions):
        prev = r["prev"]
        action = r["action"]
        nxt = r["next"]
        step = r.get("agent_step", i)

        # Retrieve first (before write, matches live run order)
        hints = mem.retrieve_hints(prev, step)
        # Write the transition
        mem.write_transition(prev, action, nxt, step)

        # Prediction: top-1 hint memory_id matches actual next?
        pred_correct = False
        if hints:
            top1_mid = str(hints[0].get("memory_id", ""))
            if "::" in top1_mid:
                parts = top1_mid.replace("tx_", "").split("::")
                if len(parts) == 3 and parts[2] == nxt:
                    pred_correct = True
        per_step_correct.append(pred_correct)
        per_step_state.append(prev)

    # Tail = bottom-50% states by visit count
    state_visits = Counter(per_step_state)
    sorted_by_v = sorted(state_visits.items(), key=lambda x: x[1])
    n_tail = max(1, int(len(sorted_by_v) * 0.5))
    tail_states = {state for state, _count in sorted_by_v[:n_tail]}

    tail_correct = []
    for i, r in enumerate(actions):
        nxt = r["next"]
        if nxt in tail_states and i < len(per_step_correct):
            tail_correct.append(per_step_correct[i])
    tail_acc = sum(tail_correct) / max(1, len(tail_correct))
    tail_err = 1.0 - tail_acc

    return {
        "tag": tag,
        "n_actions": len(actions),
        "n_tail_states": len(tail_states),
        "tail_correct_count": sum(tail_correct),
        "tail_total_count": len(tail_correct),
        "tail_pred_accuracy": tail_acc,
        "tail_pred_error": tail_err,
        "n_predictions": sum(per_step_correct),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v2_actions",
                    default=None,
                    help="Path to v2 actions.jsonl (default: auto-detect)")
    p.add_argument("--v2c_actions",
                    default=None,
                    help="Path to v2c actions.jsonl (default: auto-detect)")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--live_v2_tail_err", type=float)
    p.add_argument("--live_v2c_tail_err", type=float)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    if args.v2_actions is None:
        args.v2_actions = os.path.join(
            ROOT, "results", "method_comparison",
            "results", "B8_CTWM_v2_actions.jsonl")
    if args.v2c_actions is None:
        args.v2c_actions = os.path.join(
            ROOT, "results", "ctwm_comparison",
            "results", "B8_CTWM_v2c_actions.jsonl")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[LOAD] v2 actions: {args.v2_actions}")
    v2_actions = load_actions(args.v2_actions)
    print(f"       v2c actions: {args.v2c_actions}")
    v2c_actions = load_actions(args.v2c_actions)
    print(f"[LOAD] v2 N={len(v2_actions)}, v2c N={len(v2c_actions)}")

    # Replay: 2 backends × 2 trajectories = 4 combinations
    # But since retrieve_hints is identical between v2 and v2c backends
    # (only context_string differs, which doesn't affect tail_err), we
    # only need 2 replays: one per trajectory
    print("\n[REPLAY] on v2 trajectory (backend logic identical to live v2c backend since retrieve_hints unchanged)")
    v2_result = replay_on_trajectory(
        v2_actions,
        tag="v2_trajectory",
        tau=args.tau,
        seed=args.seed,
    )
    print(f"  → tail_err = {v2_result['tail_pred_error']:.4f}")

    print("\n[REPLAY] on v2c trajectory (same backend)")
    v2c_result = replay_on_trajectory(
        v2c_actions,
        tag="v2c_trajectory",
        tau=args.tau,
        seed=args.seed,
    )
    print(f"  → tail_err = {v2c_result['tail_pred_error']:.4f}")

    live_metrics_supplied = (
        args.live_v2_tail_err is not None
        and args.live_v2c_tail_err is not None
    )
    if (args.live_v2_tail_err is None) != (args.live_v2c_tail_err is None):
        p.error("supply both live tail-error values or neither")
    if live_metrics_supplied and not (
        0 <= args.live_v2_tail_err <= 1
        and 0 <= args.live_v2c_tail_err <= 1
    ):
        p.error("live tail-error values must be in [0, 1]")

    offline_delta = (
        v2c_result["tail_pred_error"] - v2_result["tail_pred_error"]
    )
    live_delta = (
        args.live_v2c_tail_err - args.live_v2_tail_err
        if live_metrics_supplied
        else None
    )

    isolate = {
        "study": "frozen_replay_encoding_vs_path_effect",
        "backend_retrieve_hints_identical_v2_vs_v2c": True,
        "backend_context_string_differs_v2_vs_v2c": True,
        "tail_err_offline_replay_on_v2_traj": v2_result["tail_pred_error"],
        "tail_err_offline_replay_on_v2c_traj": v2c_result["tail_pred_error"],
        "tail_err_live_v2": args.live_v2_tail_err,
        "tail_err_live_v2c": args.live_v2c_tail_err,
        "delta_live_v2c_v2": live_delta,
        "delta_offline_v2c_v2": offline_delta,
        "notes": (
            "With a fixed seed, retrieval is deterministic for a frozen trajectory. "
            "The offline delta therefore measures the difference between the two "
            "recorded trajectories under the same retrieval implementation. It "
            "does not by itself establish a causal percentage for the live-run gap."
        ),
    }

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({
            "v2_replay": v2_result,
            "v2c_replay": v2c_result,
            "isolate_analysis": isolate,
        }, f, indent=2)

    if live_metrics_supplied:
        live_section = f"""
## Supplied live-run comparison

| Run | tail_pred_err |
|:---:|:-------------:|
| Live v2 | {args.live_v2_tail_err:.4f} |
| Live v2c | {args.live_v2c_tail_err:.4f} |
| Delta (v2c - v2) | {live_delta:+.4f} |

These values were supplied on the command line; the script does not embed or
select live-run results.
"""
    else:
        live_section = """
## Live-run comparison

No live metrics were supplied. Pass both `--live_v2_tail_err` and
`--live_v2c_tail_err` to add a comparison.
"""

    md = f"""# Frozen Replay Analysis: Encoding vs Walker-Path Effect on tail_err

## Design

Offline frozen-trajectory replay:
- Load v2 actions.jsonl (frozen trajectory of B8_v2 live run)
- Load v2c actions.jsonl (frozen trajectory of B8_v2c live run)
- Replay each trajectory through B8_CTWM backend (which has identical retrieve_hints() in v2 and v2c)
- Compute tail_pred_err (bottom-50% states) for each

## Key backend fact

- `retrieve_hints()`: identical between v2 and v2c backend (5-feature Core/Tail scoring + b(r;τ) sampling)
- `context_string()`: differs (v2: verbose c=... embedded; v2c: rank-order + count-only)
- `tail_pred_err` computation depends only on top-1 hint's memory_id vs actual next state → depends only on `retrieve_hints()`, NOT on `context_string()`

For a fixed trajectory and seed, replay is deterministic under this backend.

## Results

| Trajectory | Replay backend | tail_pred_err |
|:----------:|:-------------:|:-------------:|
| v2 (live)  | B8_CTWM       | {v2_result['tail_pred_error']:.4f} |
| v2c (live) | B8_CTWM       | {v2c_result['tail_pred_error']:.4f} |

| Delta offline (v2c trajectory - v2 trajectory) | | **{offline_delta:+.4f}** |

{live_section}

## Interpretation

The offline delta compares two trajectories under identical retrieval code. It
quantifies trajectory sensitivity, but it does not alone prove what fraction of
any live-run difference was caused by the encoding change.
"""
    with open(os.path.join(args.out_dir, "isolate_analysis.md"), "w") as f:
        f.write(md)
    print("\n[DONE] wrote results.json and isolate_analysis.md")


if __name__ == "__main__":
    main()
