from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldmodelsoc.pipeline.modules import (
    make_client, TokenAccumulator,
    state_extractor, transition_extractor,
    MemoryStore, next_state_predictor, prediction_evaluator,
    token_profiler_snapshot,
)
from worldmodelsoc.llm_config import LLM_API_BASE_URL, LLM_MODEL


def generate_gt_trajectory(toy_graph: Dict[str, Any], n_steps: int, seed: int) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    rng = random.Random(seed)
    edges: List[Tuple[str, str, str]] = [tuple(e) for e in toy_graph["edges_ground_truth"]]
    adj: Dict[str, List[Tuple[str, str]]] = {}
    for (s, a, t) in edges:
        adj.setdefault(s, []).append((a, t))

    start = rng.choice(sorted(adj.keys()))
    states = [start]
    transitions: List[Tuple[str, str, str]] = []
    current = start
    for _ in range(n_steps - 1):
        options = adj.get(current, [])
        if not options:
            current = "hallway"
            options = adj.get(current, [])
        a, nxt = rng.choice(options)
        transitions.append((current, a, nxt))
        states.append(nxt)
        current = nxt
    return states, transitions


def observation_of(toy_graph: Dict[str, Any], state_id: str, rng: random.Random) -> str:
    state_defs = {s["state_id"]: s for s in toy_graph["states"]}
    st = state_defs[state_id]
    base = st["observation_template"]
    if rng.random() < 0.3:
        alias = rng.choice(st["canonical_labels"])
        base = base.replace(f"in the {state_id.replace('_',' ')}", f"in the {alias}")
    if rng.random() < 0.3:
        distractors = [
            "You hear a distant clock tick.",
            "The air feels a bit humid.",
            "A faint smell of coffee lingers.",
            "You notice a dust bunny under a corner.",
        ]
        base = base + " " + rng.choice(distractors)
    return base


def run_pipeline(toy_graph: Dict[str, Any], n_steps: int, seed: int,
                 out_dir: str) -> Dict[str, Any]:
    canonical_ids = toy_graph["canonical_state_ids"]

    adjacency_lookup: Dict[str, List[str]] = {}
    for e in toy_graph["edges_ground_truth"]:
        s, a, t = e
        adjacency_lookup.setdefault(s, []).append(t)

    rng = random.Random(seed + 100)
    gt_states, gt_transitions = generate_gt_trajectory(toy_graph, n_steps, seed=seed)

    os.makedirs(out_dir, exist_ok=True)
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    run_id = f"sanity_toy5_seed{seed}"
    events_path = os.path.join(logs_dir, f"{run_id}_events.jsonl")
    manifest_path = os.path.join(logs_dir, f"{run_id}_manifest.json")

    client = make_client()
    acc = TokenAccumulator()
    mem = MemoryStore()

    seq = dict.fromkeys(["state", "transition", "memory_access", "prediction_error", "token_profile", "meta"], 0)

    start_time_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    start_wall = time.time()

    extracted_states: List[str] = []
    extracted_transitions: List[str] = []
    prediction_events: List[Dict[str, Any]] = []
    per_step_records: List[Dict[str, Any]] = []

    def wc_ms() -> int:
        return int((time.time() - start_wall) * 1000)

    def emit(f, event_type: str, agent_step: int, extra: Dict[str, Any]) -> None:
        rec = {
            "schema_version": "0.1.0",
            "run_id": run_id,
            "benchmark": "synthetic_graph_world",
            "task_id": "toy5_star",
            "agent_step": agent_step,
            "event_type": event_type,
            "event_seq": seq[event_type],
            "timestamp_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "wallclock_ms_since_run_start": wc_ms(),
        }
        rec.update(extra)
        f.write(json.dumps(rec) + "\n")
        seq[event_type] += 1

    with open(events_path, "w", encoding="utf-8") as f:
        emit(f, "meta", 0, {"note": "toy 5-state sanity run start", "kind": "run_start"})

        for sd in toy_graph["states"]:
            mid = f"kb_{sd['state_id']}"
            content = f"{sd['state_id']}: entities={sd['entities']}, actions={sd['actions_available']}"
            info = mem.write(mid, content, step=0)
            emit(f, "memory_access", 0, {
                "memory_id": mid,
                "access_kind": info["access_kind"],
                "retrieval_rank": None,
                "retrieval_score": None,
                "access_freq_running": info["freq"],
                "memory_tokens": max(1, len(content) // 4),
                "tau_active": None,
            })

        state_freq_running: Dict[str, int] = {}
        trans_freq_running: Dict[str, int] = {}

        for step in range(n_steps):
            gt_state = gt_states[step]
            obs = observation_of(toy_graph, gt_state, rng)

            ext_state = state_extractor(client, obs, canonical_ids, acc)
            extracted_states.append(ext_state)
            state_freq_running[ext_state] = state_freq_running.get(ext_state, 0) + 1
            emit(f, "state", step, {
                "state_id": ext_state,
                "state_source": "llm_extractor",
                "state_novelty_flag": state_freq_running[ext_state] == 1,
                "state_freq_running": state_freq_running[ext_state],
                "state_context_tokens": len(obs) // 4,
            })

            mem_key = f"state_obs_{ext_state}_{step}"
            info = mem.write(mem_key, obs, step=step)
            emit(f, "memory_access", step, {
                "memory_id": mem_key,
                "access_kind": info["access_kind"],
                "retrieval_rank": None,
                "retrieval_score": None,
                "access_freq_running": info["freq"],
                "memory_tokens": max(1, len(obs) // 4),
                "tau_active": None,
            })

            hits = mem.retrieve(obs, top_k=3, step=step)
            for h in hits:
                emit(f, "memory_access", step, {
                    "memory_id": h["memory_id"],
                    "access_kind": h["access_kind"],
                    "retrieval_rank": h["retrieval_rank"],
                    "retrieval_score": h["retrieval_score"],
                    "access_freq_running": h["access_freq_running"],
                    "memory_tokens": h["memory_tokens"],
                    "tau_active": None,
                })

            if step + 1 < n_steps:
                gt_next_state = gt_states[step + 1]
                gt_action = gt_transitions[step][1] if step < len(gt_transitions) else "unknown"

                pred_state, pred_conf = next_state_predictor(
                    client, ext_state, gt_action, canonical_ids, hits, acc,
                )
                eval_out = prediction_evaluator(pred_state, gt_next_state, canonical_ids, adjacency_lookup)

                if eval_out["prediction_correct"]:
                    avalanche = 1
                else:
                    avalanche = 2 if eval_out["error_magnitude"] == 0.5 else 3

                emit(f, "prediction_error", step, {
                    "predicted_next_state_id": pred_state,
                    "actual_next_state_id": gt_next_state,
                    "prediction_correct": eval_out["prediction_correct"],
                    "error_magnitude": eval_out["error_magnitude"],
                    "prediction_confidence": pred_conf,
                    "tail_or_core": "unknown",
                    "avalanche_size": avalanche,
                })
                prediction_events.append({
                    "step": step, "pred": pred_state, "actual": gt_next_state,
                    "correct": eval_out["prediction_correct"], "err": eval_out["error_magnitude"],
                })

            if step + 1 < n_steps:
                (u, a, v) = gt_transitions[step]
                tid = transition_extractor(client, ext_state, a,
                                            gt_states[step + 1], acc)
                extracted_transitions.append(tid)
                trans_freq_running[tid] = trans_freq_running.get(tid, 0) + 1
                emit(f, "transition", step, {
                    "transition_id": tid,
                    "state_id_prev": tid.split("::")[0] if "::" in tid else ext_state,
                    "action": a,
                    "state_id_next": tid.split("::")[-1] if "::" in tid else gt_states[step + 1],
                    "transition_source": "llm_extractor",
                    "transition_freq_running": trans_freq_running[tid],
                    "transition_novelty_flag": trans_freq_running[tid] == 1,
                })

            if step % 5 == 0:
                snap = token_profiler_snapshot(acc, memory_token_estimate=sum(
                    max(1, len(r["content"]) // 4) for r in mem.entries.values()
                ))
                emit(f, "token_profile", step, snap)

            per_step_records.append({
                "step": step, "gt_state": gt_state, "ext_state": ext_state, "obs": obs,
            })

        snap = token_profiler_snapshot(acc, memory_token_estimate=sum(
            max(1, len(r["content"]) // 4) for r in mem.entries.values()
        ))
        emit(f, "token_profile", n_steps - 1, snap)

        emit(f, "meta", n_steps - 1, {"note": "toy 5-state sanity run end", "kind": "run_end"})

    end_time_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    manifest = {
        "run_id": run_id,
        "benchmark": "synthetic_graph_world",
        "task_ids": ["toy5_star"],
        "n_agent_steps": n_steps,
        "started_at": start_time_utc,
        "finished_at": end_time_utc,
        "agent_config": {
            "memory_backend": "simple_kv_overlap_retrieval",
            "top_k": 3,
            "seed": seed,
        },
        "llm": {
            "provider": "OpenAI-compatible",
            "model": LLM_MODEL,
            "endpoint": LLM_API_BASE_URL,
        },
        "seed": seed,
        "code_version": "sanity_v0.1",
        "counts": {
            "state_events": seq["state"],
            "transition_events": seq["transition"],
            "memory_access_events": seq["memory_access"],
            "prediction_error_events": seq["prediction_error"],
            "token_profile_events": seq["token_profile"],
            "meta_events": seq["meta"],
        },
        "success_flag": mem.conflicts == 0,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    n_correct_states = sum(1 for i in range(n_steps) if extracted_states[i] == gt_states[i])
    B1_rate = n_correct_states / n_steps
    B1_pass = B1_rate >= 0.9

    gt_trans_ids = {f"{u}::{a}::{v}" for (u, a, v) in gt_transitions}
    ext_trans_ids = set(extracted_transitions)
    coverage = len(gt_trans_ids & ext_trans_ids) / max(1, len(gt_trans_ids))
    B2_pass = coverage >= 0.9

    B3_pass = (mem.conflicts == 0) and (mem.write_events > 0) and (mem.read_events > 0)

    warmup = 5
    later = [p for p in prediction_events if p["step"] >= warmup]
    later_acc = sum(1 for p in later if p["correct"]) / max(1, len(later))
    B4_pass = later_acc >= 0.5

    B5_pass = seq["token_profile"] >= 5

    B6_pass = all(seq[t] >= 1 for t in ["state", "transition", "memory_access", "prediction_error", "token_profile", "meta"])

    all_pass = all([B1_pass, B2_pass, B3_pass, B4_pass, B5_pass, B6_pass])

    metrics = {
        "study": "toy_pipeline_sanity",
        "run_id": run_id,
        "n_steps": n_steps,
        "seed": seed,
        "B1_state_extract_rate": B1_rate,
        "B1_pass": B1_pass,
        "B2_transition_coverage": coverage,
        "B2_pass": B2_pass,
        "B3_memory_conflicts": mem.conflicts,
        "B3_writer_events": mem.write_events,
        "B3_reader_events": mem.read_events,
        "B3_pass": B3_pass,
        "B4_late_prediction_accuracy_after_warmup": later_acc,
        "B4_n_predictions_after_warmup": len(later),
        "B4_pass": B4_pass,
        "B5_n_token_profile_events": seq["token_profile"],
        "B5_pass": B5_pass,
        "B6_event_type_counts": dict(seq),
        "B6_pass": B6_pass,
        "all_pass": all_pass,
        "token_usage": {
            "tokens_prompt": acc.tokens_prompt,
            "tokens_completion": acc.tokens_completion,
            "api_calls": acc.api_calls,
        },
        "events_path": events_path,
        "manifest_path": manifest_path,
        "gt_state_seq": gt_states,
        "extracted_state_seq": extracted_states,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--toy_graph",
        type=str,
        default=str(ROOT / "data" / "toy_graph.json"),
    )
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "results" / "sanity"),
    )
    args = parser.parse_args()

    with open(args.toy_graph, "r", encoding="utf-8") as f:
        toy_graph = json.load(f)

    metrics = run_pipeline(toy_graph, args.n_steps, args.seed, args.out_dir)

    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[DONE] wrote {results_path}")
    print(f"  B1_state_accuracy: {metrics['B1_state_extract_rate']:.3f}   pass={metrics['B1_pass']}")
    print(f"  B2_transition_cov: {metrics['B2_transition_coverage']:.3f}   pass={metrics['B2_pass']}")
    print(f"  B3_mem: conflicts={metrics['B3_memory_conflicts']} writes={metrics['B3_writer_events']} reads={metrics['B3_reader_events']}  pass={metrics['B3_pass']}")
    print(f"  B4_predict_acc_after_warmup: {metrics['B4_late_prediction_accuracy_after_warmup']:.3f}  pass={metrics['B4_pass']}")
    print(f"  B5_token_profile_events: {metrics['B5_n_token_profile_events']}  pass={metrics['B5_pass']}")
    print(f"  B6_event_types: {metrics['B6_event_type_counts']}  pass={metrics['B6_pass']}")
    print(f"  ALL PASS: {metrics['all_pass']}")
    print(f"  token usage: prompt={metrics['token_usage']['tokens_prompt']}  completion={metrics['token_usage']['tokens_completion']}  api_calls={metrics['token_usage']['api_calls']}")


if __name__ == "__main__":
    main()
