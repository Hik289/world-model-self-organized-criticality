"""Compare compact CTWM encoding with seven memory baselines.

Memory context is included in every policy prompt. Outputs contain both
provider-reported prompt tokens and the legacy token estimator.
"""

from __future__ import annotations
import argparse, datetime as dt, json, os, random, re, sys, time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import powerlaw
from scipy import stats as spstats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldmodelsoc.env.synthetic_graph_world import (  # noqa: E402
    build_graph,
    build_state_payloads,
    describe_action_options,
)
from worldmodelsoc.memory.backends_ctwm import (  # noqa: E402
    B1_FullHistory, B2_SlidingWindow, B3_FlatRetrieval, B4_FrequencyCache,
    B5_RecencyCache, B6_HierarchicalSummary, B7_GraphMemory, B8_CTWM,
)
from worldmodelsoc.llm_config import LLM_MODEL, make_openai_client  # noqa: E402


PRICE_PROMPT_PER_1M = 0.15
PRICE_COMPL_PER_1M = 0.60


def make_client():
    return make_openai_client()


class CostAccountant:
    def __init__(self, budget_usd=0.5):
        if budget_usd <= 0:
            raise ValueError("budget_usd must be positive")
        self.tokens_prompt = 0; self.tokens_completion = 0; self.api_calls = 0
        self.fallback_count = 0; self.budget_usd = budget_usd
        self.error_count = 0; self.last_error = None
        self.per_call_prompt_tokens: List[int] = []  # for tokens_actual analysis
    def add(self, p, c):
        self.tokens_prompt += p; self.tokens_completion += c; self.api_calls += 1
        self.per_call_prompt_tokens.append(p)
    def estimated_usd(self):
        return (self.tokens_prompt/1e6)*PRICE_PROMPT_PER_1M + (self.tokens_completion/1e6)*PRICE_COMPL_PER_1M
    def exceeded(self):
        return self.estimated_usd() >= self.budget_usd


def _extract_action_idx(text, n_actions):
    if not text: return None
    m = re.search(r"\{[^{}]*\}", text)
    candidate = m.group(0) if m else text
    try:
        obj = json.loads(candidate)
        idx = int(obj.get("action_idx", -1))
        if 0 <= idx < n_actions: return idx
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    nums = re.findall(r"\d+", text)
    for n in nums:
        i = int(n)
        if 0 <= i < n_actions: return i
    return None


def llm_pick_action(client, current_sid, actions, neighbors, recent_states,
                    memory_context_str, accountant, fallback_rng,
                    state_context="", action_options=None,
                    max_completion_tokens=60, retries=2):
    n_actions = len(actions)
    recent_text = ", ".join(recent_states[-5:]) if recent_states else "(none)"
    system_msg = ("You are a walker choosing the next action in a text world. "
                  "Reply with ONLY: {\"action_idx\": <int>} where int is 0..n_actions-1. "
                  "No explanation.")
    # Real memory context concat here
    options = action_options or actions
    user_msg = (f"current_state: {current_sid}\n"
                f"state_payload: {state_context or '(not provided)'}\n"
                f"action_options: {options}\n"
                f"n_neighbors: {len(neighbors)}\n"
                f"recent_states: {recent_text}\n"
                f"memory: {memory_context_str}\n"
                f"Choose one action index (0..{n_actions-1}).")

    last_err = None
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role":"system","content":system_msg},
                          {"role":"user","content":user_msg}],
                max_completion_tokens=max_completion_tokens,
            )
            if resp.usage:
                accountant.add(resp.usage.prompt_tokens or 0,
                                resp.usage.completion_tokens or 0)
            content = (resp.choices[0].message.content or "").strip()
            idx = _extract_action_idx(content, n_actions)
            if idx is not None: return idx, True
            last_err = ValueError(f"parse: {content[:80]!r}")
        except Exception as e:
            last_err = e; time.sleep(1.0)
    accountant.fallback_count += 1
    accountant.error_count += 1
    accountant.last_error = str(last_err)[:200] if last_err else None
    return fallback_rng.randrange(n_actions), False


def build_backend(name: str, tau: float = 1.0, seed: int = 42):
    if name == "B1_FullHistory":       return B1_FullHistory()
    if name == "B2_SlidingWindow":     return B2_SlidingWindow(K=100)
    if name == "B3_FlatRetrieval":     return B3_FlatRetrieval(top_k=3, seed=seed)
    if name == "B4_FrequencyCache":    return B4_FrequencyCache(capacity=100, top_k=3)
    if name == "B5_RecencyCache":      return B5_RecencyCache(capacity=100, top_k=3)
    if name == "B6_HierarchicalSummary": return B6_HierarchicalSummary(chunk=100)
    if name == "B7_GraphMemory":       return B7_GraphMemory(episode_length=100, top_k=3)
    if name == "B8_CTWM":
        return B8_CTWM(
            tau=tau,
            core_pct=0.30,
            core_slots=3,
            tail_slots=2,
            seed=seed,
        )
    raise ValueError(f"unknown: {name}")


def run_method(method_name, n_nodes, n_steps, seed, budget_usd, tau, out_dir):
    t0 = time.time()
    print(f"\n===== {method_name} (N={n_steps}) =====", flush=True)

    g = build_graph("scale_free", n_nodes, seed=seed)
    payloads = build_state_payloads(g, seed=seed)

    rng_walk = random.Random(seed + 100)
    rng_fallback = random.Random(seed + 300)
    neighbors_cache = {n: list(g.successors(n)) for n in g.nodes()}
    action_cache = {n: payloads[n].actions for n in g.nodes()}

    client = make_client()
    accountant = CostAccountant(budget_usd=budget_usd)
    mem = build_backend(method_name, tau=tau, seed=seed)
    uses_entities = isinstance(mem, B7_GraphMemory)

    walker_states: set = set()
    walker_trans: set = set()
    per_step_tokens_est: List[int] = []
    per_step_action_correct: List[bool] = []
    per_step_state: List[str] = []
    per_step_prompt_tokens: List[int] = []  # tokens_actual per step
    per_step_ctx_chars: List[int] = []
    per_step_b8_core_fill: List[int] = []
    per_step_b8_tail_fill: List[int] = []
    per_step_b8_n_clusters: List[int] = []
    recent_states: List[str] = []

    action_records: List[Dict[str, Any]] = []
    mem_time_records: List[Dict[str, Any]] = []

    current = rng_walk.choice(list(g.nodes()))
    stopped_reason = "completed"
    actual_steps = 0
    run_id = f"{method_name}_v2c_scale_free_v{n_nodes}_s{seed}"
    run_started_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    for step in range(n_steps):
        sid = f"v_{current:04d}"
        walker_states.add(sid)
        per_step_state.append(sid)
        recent_states.append(sid)

        if step + 1 >= n_steps:
            actual_steps = step + 1; break

        # Retrieve
        hints = mem.retrieve_hints(sid, step)
        ctx_est = mem.context_tokens_estimator()
        per_step_tokens_est.append(ctx_est)
        ctx_str = mem.context_string(sid)
        per_step_ctx_chars.append(len(ctx_str))
        if method_name == "B8_CTWM":
            per_step_b8_core_fill.append(len(getattr(mem, "_last_core", []) or []))
            per_step_b8_tail_fill.append(len(getattr(mem, "_last_tail", []) or []))
            # clusters = unique prev states in _last_tail
            tail_prevs = set()
            for t in (getattr(mem, "_last_tail", []) or []):
                if t in mem.entries:
                    tail_prevs.add(mem.entries[t]["prev"])
            per_step_b8_n_clusters.append(len(tail_prevs))
        for h in hints:
            mem_time_records.append({
                "schema_version": "0.1.1", "run_id": run_id,
                "event_type": "memory_access", "event_seq": len(mem_time_records),
                "agent_step": step,
                "memory_id": h.get("memory_id"), "access_kind": "read",
                "retrieval_rank": h.get("rank"),
                "layer": h.get("layer"),
            })

        # LLM pick
        actions = action_cache[current]
        neighbors = neighbors_cache[current]
        n_prompt_before = accountant.tokens_prompt
        picked_idx, is_llm = llm_pick_action(
            client, sid, actions, neighbors, recent_states, ctx_str,
            accountant, rng_fallback,
            state_context=(
                f"entities={payloads[current].entities}; "
                f"constraints={payloads[current].constraints}"
            ),
            action_options=describe_action_options(g, payloads, current),
        )
        n_prompt_delta = accountant.tokens_prompt - n_prompt_before
        per_step_prompt_tokens.append(n_prompt_delta)

        n_acts = len(actions)
        if len(neighbors) == 0:
            nxt_node = current
        else:
            seg_size = max(1, len(neighbors) // n_acts)
            seg_start = picked_idx * seg_size
            seg_end = seg_start + seg_size if picked_idx < n_acts - 1 else len(neighbors)
            seg = neighbors[seg_start:seg_end] if seg_start < len(neighbors) else neighbors
            nxt_node = rng_walk.choice(seg) if seg else rng_walk.choice(neighbors)
        action = actions[picked_idx]
        nxt_sid = f"v_{nxt_node:04d}"
        tid = f"{sid}::{action}::{nxt_sid}"
        walker_trans.add(tid)

        # Write
        if uses_entities:
            mem.write_transition_with_entities(
                sid, action, nxt_sid, step,
                entities_prev=payloads[current].entities,
                entities_next=payloads[nxt_node].entities,
            )
        else:
            mem.write_transition(sid, action, nxt_sid, step)
        mem_time_records.append({
            "schema_version": "0.1.1", "run_id": run_id,
            "event_type": "memory_access", "event_seq": len(mem_time_records),
            "agent_step": step,
            "memory_id": tid, "access_kind": "write",
        })

        # Prediction check: top-1 hint next matches actual?
        pred_correct = False
        if hints:
            top1_mid = str(hints[0].get("memory_id", ""))
            if "::" in top1_mid:
                parts = top1_mid.replace("tx_", "").split("::")
                if len(parts) == 3 and parts[2] == nxt_sid:
                    pred_correct = True
        per_step_action_correct.append(pred_correct)

        action_records.append({
            "schema_version": "0.1.1", "run_id": run_id,
            "event_type": "action", "event_seq": step,
            "agent_step": step,
            "prev": sid, "action_idx": picked_idx, "action": action,
            "next": nxt_sid, "is_llm_picked": is_llm,
            "prompt_tokens_actual": n_prompt_delta,
        })

        current = nxt_node
        actual_steps = step + 1

        if step % 100 == 99:
            used = accountant.estimated_usd()
            if step % 500 == 499:
                actual_avg = np.mean(per_step_prompt_tokens) if per_step_prompt_tokens else 0
                print(f"  step {step+1}: used=${used:.3f} tokens_actual/step={actual_avg:.0f} tokens_est/step={np.mean(per_step_tokens_est):.0f}", flush=True)
            if accountant.exceeded():
                stopped_reason = "budget_exceeded"
                print(f"  [BUDGET] ${used:.3f}>=${budget_usd:.2f} stop step {step+1}", flush=True)
                break

    # Metrics
    avg_tokens_est = float(np.mean(per_step_tokens_est)) if per_step_tokens_est else 0.0
    avg_tokens_actual = float(np.mean(per_step_prompt_tokens)) if per_step_prompt_tokens else 0.0
    coverage_state = mem.coverage_state(walker_states)
    coverage_trans = mem.coverage_trans(walker_trans)

    # Tail err: bottom-50% state visit → prediction accuracy
    state_visits = Counter(per_step_state)
    sorted_by_v = sorted(state_visits.items(), key=lambda x: x[1])
    n_tail = max(1, int(len(sorted_by_v) * 0.5))
    tail_states = {state for state, _count in sorted_by_v[:n_tail]}
    tail_correct = []
    for i in range(len(per_step_state) - 1):
        if per_step_state[i + 1] in tail_states and i < len(per_step_action_correct):
            tail_correct.append(per_step_action_correct[i])
    tail_pred_acc = float(np.mean(tail_correct)) if tail_correct else 0.0
    tail_pred_err = 1.0 - tail_pred_acc

    mem_arr = np.array(list(mem.access_counter.values()), dtype=float)

    def _g(x):
        if x.size == 0: return 0.0
        x = np.sort(x); n = x.size
        if x.sum() == 0: return 0.0
        cum = np.cumsum(x)
        return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    gi = _g(mem_arr) if mem_arr.size else 0.0
    skew = (
        float(spstats.skew(mem_arr))
        if mem_arr.size > 1 and np.ptp(mem_arr) > 0
        else 0.0
    )

    # Bucket analysis (5 buckets of ~400 steps) for tokens_actual + ctx_chars
    n_avail = len(per_step_prompt_tokens)
    n_buckets = 5
    buckets = []
    if n_avail >= n_buckets:
        bs = n_avail // n_buckets
        for i in range(n_buckets):
            lo, hi = i * bs, (i + 1) * bs if i < n_buckets - 1 else n_avail
            bucket = {
                "step_range": [lo, hi],
                "avg_tokens_actual": float(np.mean(per_step_prompt_tokens[lo:hi])),
                "avg_ctx_chars": float(np.mean(per_step_ctx_chars[lo:hi])) if per_step_ctx_chars[lo:hi] else 0.0,
            }
            if method_name == "B8_CTWM" and per_step_b8_core_fill[lo:hi]:
                bucket["b8_avg_core_fill"] = float(np.mean(per_step_b8_core_fill[lo:hi]))
                bucket["b8_avg_tail_fill"] = float(np.mean(per_step_b8_tail_fill[lo:hi]))
                bucket["b8_avg_n_clusters"] = float(np.mean(per_step_b8_n_clusters[lo:hi]))
            buckets.append(bucket)

    fit_info = {}
    if mem_arr.size >= 30:
        try:
            fit = powerlaw.Fit(mem_arr, discrete=True, verbose=False)
            fit_info = {"alpha_hat": float(fit.alpha), "xmin": float(fit.xmin),
                         "alpha_sigma": float(fit.sigma)}
        except Exception as exc:
            fit_info = {
                "error": type(exc).__name__,
                "message": str(exc)[:200],
            }

    result = {
        "method": method_name,
        "config": {"graph_type": "scale_free", "n_nodes": n_nodes,
                    "n_steps_target": n_steps, "n_steps_actual": actual_steps,
                    "seed": seed, "tau": tau if method_name == "B8_CTWM" else None},
        "cost_usd": accountant.estimated_usd(),
        "api_calls": accountant.api_calls,
        "fallback_count": accountant.fallback_count,
        "error_count": accountant.error_count,
        "elapsed_sec": time.time() - t0,
        "stopped_reason": stopped_reason,
        "run_started_utc": run_started_utc,
        "run_finished_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "metrics": {
            "avg_tokens_actual_per_step": avg_tokens_actual,
            "avg_tokens_estimator_per_step": avg_tokens_est,
            "coverage_state": coverage_state,
            "coverage_trans": coverage_trans,
            "tail_pred_accuracy": tail_pred_acc,
            "tail_pred_error": tail_pred_err,
            "tail_state_count": len(tail_states),
            "mem_gini": gi, "mem_skew": skew,
            "mem_n_unique": int(mem_arr.size),
            "walker_unique_states": len(walker_states),
            "walker_unique_trans": len(walker_trans),
        },
        "fit": fit_info,
        "diagnostics": {
            "buckets_5x400": buckets,
        },
    }

    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    tag = f"{method_name}_v2c"
    with open(os.path.join(results_dir, f"{tag}_meta.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(results_dir, f"{tag}_mem_counts.json"), "w") as f:
        json.dump(dict(mem.access_counter), f)
    with open(os.path.join(results_dir, f"{tag}_mem_time.jsonl"), "w") as f:
        for e in mem_time_records:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(results_dir, f"{tag}_actions.jsonl"), "w") as f:
        for e in action_records:
            f.write(json.dumps(e) + "\n")

    m = result["metrics"]
    print(f"  [DONE] elapsed={result['elapsed_sec']:.1f}s cost=${result['cost_usd']:.3f}", flush=True)
    print(f"    tokens_actual/step={m['avg_tokens_actual_per_step']:.0f} tokens_est/step={m['avg_tokens_estimator_per_step']:.0f}", flush=True)
    print(f"    cov_state={m['coverage_state']:.3f} cov_trans={m['coverage_trans']:.3f}", flush=True)
    print(f"    tail_pred_err={m['tail_pred_error']:.3f} mem_gini={m['mem_gini']:.3f} n_uniq={m['mem_n_unique']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--n_steps_b1", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--budget_total", type=float, default=3.0)
    parser.add_argument("--methods", type=str, nargs="+",
                         default=["B1_FullHistory", "B2_SlidingWindow",
                                  "B3_FlatRetrieval", "B4_FrequencyCache",
                                  "B5_RecencyCache", "B6_HierarchicalSummary",
                                  "B7_GraphMemory", "B8_CTWM"])
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    if args.budget_total <= 0:
        parser.error("--budget_total must be positive")

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = []
    total_cost = 0.0

    # Per-method caps reflect the different context lengths. The defaults sum
    # to $2.95 under the configured pricing estimate.
    method_budgets = {
        "B1_FullHistory": 0.35,      # N=500 with linear growth: ~$0.27 empirical
        "B2_SlidingWindow": 0.65,    # N=2000 with K=100 window ~1500 tokens/step: ~$0.55
        "B3_FlatRetrieval": 0.20,
        "B4_FrequencyCache": 0.20,
        "B5_RecencyCache": 0.20,
        "B6_HierarchicalSummary": 0.65,   # similar to B2, grows with summaries
        "B7_GraphMemory": 0.35,      # KG may grow, allow more
        "B8_CTWM": 0.35,             # 5 features + Core/Tail slots, allow more
    }
    for m in args.methods:
        n_steps = args.n_steps_b1 if m == "B1_FullHistory" else args.n_steps
        method_budget = method_budgets.get(m, max(0.15, (args.budget_total - total_cost) / max(1, len(args.methods) - len(all_results))))
        remaining = args.budget_total - total_cost
        if remaining <= 0:
            all_results.append({
                "method": m,
                "error": "skipped because the total budget was exhausted",
            })
            continue
        method_budget = min(method_budget, remaining)
        try:
            rep = run_method(m, args.n_nodes, n_steps, args.seed,
                              method_budget, args.tau, args.out_dir)
            all_results.append(rep)
            total_cost += rep["cost_usd"]
        except Exception as e:
            print(f"  ERROR {m}: {e}", flush=True)
            import traceback; traceback.print_exc()
            all_results.append({"method": m, "error": str(e)})

    # Verdicts
    by_m = {r["method"]: r for r in all_results if "metrics" in r}
    def get(n, k): return by_m.get(n, {}).get("metrics", {}).get(k)

    b3_a = get("B3_FlatRetrieval", "avg_tokens_actual_per_step")
    b7_a = get("B7_GraphMemory", "avg_tokens_actual_per_step")
    b8_a = get("B8_CTWM", "avg_tokens_actual_per_step")

    G6 = (b8_a / b7_a <= 0.75) if b8_a and b7_a else None
    G7 = (b8_a / b3_a <= 0.65) if b8_a and b3_a else None
    b7_tail = get("B7_GraphMemory", "tail_pred_error")
    b8_tail = get("B8_CTWM", "tail_pred_error")
    G8 = (b8_tail <= b7_tail * 1.1) if b7_tail is not None and b8_tail is not None else None
    b7_cs = get("B7_GraphMemory", "coverage_state")
    b8_cs = get("B8_CTWM", "coverage_state")
    b7_ct = get("B7_GraphMemory", "coverage_trans")
    b8_ct = get("B8_CTWM", "coverage_trans")
    G9_s = (
        b8_cs >= 0.95 * b7_cs
        if b8_cs is not None and b7_cs is not None
        else None
    )
    G9_t = (
        b8_ct >= 0.95 * b7_ct
        if b8_ct is not None and b7_ct is not None
        else None
    )

    verdict = {
        "G6": {"pass": G6, "b8_actual": b8_a, "b7_actual": b7_a,
                "ratio": (b8_a / b7_a) if b8_a and b7_a else None},
        "G7": {"pass": G7, "b8_actual": b8_a, "b3_actual": b3_a,
                "ratio": (b8_a / b3_a) if b8_a and b3_a else None},
        "G8": {"pass": G8, "b8_tail_err": b8_tail, "b7_tail_err": b7_tail},
        "G9_state": {"pass": G9_s, "b8": b8_cs, "b7": b7_cs},
        "G9_trans": {"pass": G9_t, "b8": b8_ct, "b7": b7_ct},
    }

    summary = {
        "study": "compact_ctwm_comparison",
        "config": {"graph_type": "scale_free", "n_nodes": args.n_nodes,
                    "n_steps_default": args.n_steps, "n_steps_b1": args.n_steps_b1,
                    "seed": args.seed, "tau": args.tau,
                    "budget_total_usd": args.budget_total,
                    "policy": "llm_policy",
                    "model": LLM_MODEL,
                    "backend_version": "v2c",
                    "notes": "Memory context is included in the policy prompt; token counts come from API usage.",
                    "ctwm_v1_limitations": "The uncertainty feature is fixed at 0.5; dynamic tail expansion is not implemented."},
        "total_cost_usd": total_cost,
        "results": all_results,
        "G6_G7_G8_G9_verdict": verdict,
    }
    with open(os.path.join(args.out_dir, "summary_v2c.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== METHODS COMPARISON (v2) =====")
    print(f"{'method':<24}{'N':<6}{'cost$':<9}{'tok_act':<10}{'tok_est':<10}{'cov_s':<8}{'cov_t':<8}{'tail_err':<10}{'gini':<8}")
    for r in all_results:
        if "metrics" not in r: continue
        m = r["metrics"]; c = r["config"]
        print(f"{r['method']:<24}{c['n_steps_actual']:<6}{r['cost_usd']:<9.3f}"
              f"{m['avg_tokens_actual_per_step']:<10.1f}{m['avg_tokens_estimator_per_step']:<10.1f}"
              f"{m['coverage_state']:<8.3f}{m['coverage_trans']:<8.3f}"
              f"{m['tail_pred_error']:<10.3f}{m['mem_gini']:<8.3f}")
    print("\n===== G6-G9 VERDICT (v2, tokens_actual based) =====")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print(f"\nTotal cost: ${total_cost:.3f} / ${args.budget_total:.2f}")


if __name__ == "__main__":
    main()
