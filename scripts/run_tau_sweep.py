"""
Exp6 τ scan — LLM policy version (P3).
Cleanroom vs seed=42 LLM policy pilot; single τ per run for parallel launch.
"""

from __future__ import annotations
import argparse, datetime as dt, json, os, random, re, sys, time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import powerlaw
from openai import OpenAI
from scipy import stats as spstats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldmodelsoc.memory.reservoir import TauReservoirMemory, summary_stats, pl_fit  # noqa: E402
from worldmodelsoc.env.synthetic_graph_world import build_graph, build_state_payloads  # noqa: E402
from worldmodelsoc.llm_config import LLM_MODEL, make_openai_client  # noqa: E402

PRICE_PROMPT_PER_1M = 0.15
PRICE_COMPL_PER_1M = 0.60


def make_client():
    return make_openai_client()


class CostAccountant:
    def __init__(self, budget_usd=1.0):
        self.tokens_prompt = 0; self.tokens_completion = 0; self.api_calls = 0
        self.fallback_count = 0; self.budget_usd = budget_usd
        self.error_count = 0; self.last_error = None
    def add(self, p, c):
        self.tokens_prompt += p; self.tokens_completion += c; self.api_calls += 1
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
    except Exception: pass
    nums = re.findall(r"\d+", text)
    for n in nums:
        i = int(n)
        if 0 <= i < n_actions: return i
    return None


def llm_pick_action(client, current_sid, entities, actions, neighbors,
                    recent_states, memory_hints, accountant,
                    max_completion_tokens=60, retries=2):
    n_actions = len(actions)
    hints_text = ""
    if memory_hints:
        hh = [f"{h.get('memory_id','')[:40]} (freq={h.get('access_freq_running','?')})"
              for h in memory_hints[:3]]
        hints_text = "\n  memory_hints: " + "; ".join(hh)
    recent_text = ", ".join(recent_states[-5:]) if recent_states else "(none)"
    system_msg = ("You are a walker choosing the next action in a text world. "
                  "Reply with ONLY: {\"action_idx\": <int>} where int is 0..n_actions-1. "
                  "No explanation.")
    user_msg = (f"current_state: {current_sid}\nactions: {actions}\n"
                f"n_neighbors: {len(neighbors)}\nrecent_states: {recent_text}"
                f"{hints_text}\nChoose one action index (0..{n_actions-1}).")

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
    return random.Random().randrange(n_actions), False


def run_one_tau_llm(tau, n_nodes, n_steps, seed, reservoir_capacity, K_pool, M_pass,
                    out_dir, budget_usd, graph_type="scale_free", checkpoint_every=500):
    t0 = time.time()
    g = build_graph(graph_type, n_nodes, seed=seed)
    payloads = build_state_payloads(g, seed=seed)
    rng_walk = random.Random(seed + 100)
    rng_mem = random.Random(seed + 200 + int(tau * 1000))
    neighbors_cache = {n: list(g.successors(n)) for n in g.nodes()}
    action_cache = {n: payloads[n].actions for n in g.nodes()}

    client = make_client()
    accountant = CostAccountant(budget_usd=budget_usd)
    mem = TauReservoirMemory(capacity=reservoir_capacity, rng=rng_mem, tau=tau,
                              K_pool=K_pool, M_pass=M_pass)

    state_counter = Counter(); trans_counter = Counter()
    mem_time_records = []; action_records = []; recent_states = []

    current = rng_walk.choice(list(g.nodes()))
    actual_steps = 0; stopped_reason = "completed"

    for step in range(n_steps):
        sid = f"v_{current:04d}"
        state_counter[sid] += 1
        recent_states.append(sid)
        if step + 1 >= n_steps:
            actual_steps = step + 1; break

        neighbors = neighbors_cache[current]; actions = action_cache[current]
        hints = mem.retrieve(current_state=sid, step=step)
        for ev in hints:
            mem_time_records.append((step, ev["memory_id"], ev["access_kind"], ev.get("retrieval_rank", -1)))

        picked_idx, is_llm = llm_pick_action(
            client, sid, payloads[current].entities, actions,
            neighbors, recent_states, hints, accountant)
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
        mid = f"tx_{tid}"; content = f"transition {sid}--{action}-->{nxt_sid}"
        for ev in mem.write(mid, content, prev=sid, action=action, nxt=nxt_sid, step=step):
            mem_time_records.append((step, ev["memory_id"], ev["access_kind"], -1))

        trans_counter[tid] += 1
        action_records.append({"step":step,"prev":sid,"action_idx":picked_idx,
                                "action":action,"next":nxt_sid,"is_llm_picked":is_llm})
        current = nxt_node
        actual_steps = step + 1

        if step % 100 == 99:
            used = accountant.estimated_usd()
            if step % checkpoint_every == checkpoint_every - 1:
                print(f"τ={tau} step {step+1}: used=${used:.3f} "
                      f"({accountant.api_calls} calls, {accountant.fallback_count} fb)", flush=True)
            if accountant.exceeded():
                stopped_reason = "budget_exceeded"
                print(f"τ={tau} BUDGET ${used:.3f}>=${budget_usd:.2f} stop step {step+1}", flush=True)
                break

    results_dir = os.path.join(out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    tag = f"tau_{tau:.2f}_llm"

    with open(os.path.join(results_dir, f"{tag}_mem_counts.json"), "w") as f:
        json.dump(dict(mem.access_counter), f)
    with open(os.path.join(results_dir, f"{tag}_state_counts.json"), "w") as f:
        json.dump(dict(state_counter), f)
    with open(os.path.join(results_dir, f"{tag}_trans_counts.json"), "w") as f:
        json.dump(dict(trans_counter), f)
    with open(os.path.join(results_dir, f"{tag}_mem_time.jsonl"), "w") as f:
        for st, mid, kind, rank in mem_time_records:
            f.write(json.dumps({"step":st,"mid":mid,"kind":kind,"rank":rank}) + "\n")
    with open(os.path.join(results_dir, f"{tag}_actions.jsonl"), "w") as f:
        for r in action_records:
            f.write(json.dumps(r) + "\n")

    idx_seq = [r["action_idx"] for r in action_records]
    if idx_seq:
        c = Counter(idx_seq)
        p = np.array(list(c.values()))/len(idx_seq)
        entropy = float(-np.sum(p*np.log2(p+1e-12)))
        same_prev = float(sum(1 for i in range(1,len(idx_seq)) if idx_seq[i]==idx_seq[i-1])/max(1,len(idx_seq)-1))
        llm_pick_rate = float(sum(1 for r in action_records if r["is_llm_picked"])/len(action_records))
    else:
        entropy = 0.0; same_prev = 0.0; llm_pick_rate = 0.0

    mem_freqs = list(mem.access_counter.values())
    result = {
        "tau": tau, "graph_type": graph_type, "n_nodes": n_nodes,
        "n_steps_target": n_steps, "actual_steps": actual_steps, "seed": seed,
        "reservoir_capacity": reservoir_capacity, "K_pool": K_pool, "M_pass": M_pass,
        "stopped_reason": stopped_reason, "elapsed_sec": time.time() - t0,
        "cost": {"budget_usd": budget_usd, "tokens_prompt": accountant.tokens_prompt,
                  "tokens_completion": accountant.tokens_completion,
                  "api_calls": accountant.api_calls,
                  "estimated_usd": accountant.estimated_usd()},
        "policy": {"type": "gpt-5.4-mini_llm_policy", "llm_pick_rate": llm_pick_rate,
                    "fallback_count": accountant.fallback_count,
                    "error_count": accountant.error_count,
                    "action_entropy_bits": entropy, "same_as_prev_step_rate": same_prev,
                    "last_error": accountant.last_error},
        "mem_stats": summary_stats(mem_freqs),
        "state_stats": summary_stats(list(state_counter.values())),
        "trans_stats": summary_stats(list(trans_counter.values())),
        "mem_fit": pl_fit(mem_freqs),
        "memory_bookkeeping": {"n_write": mem.write_events, "n_read": mem.read_events,
                                "n_evict": mem.evict_events, "n_conflicts": mem.conflicts,
                                "n_slots_used": len(mem.slots)},
        "artifacts": {"mem_counts": f"{tag}_mem_counts.json",
                       "state_counts": f"{tag}_state_counts.json",
                       "trans_counts": f"{tag}_trans_counts.json",
                       "mem_time_sidecar": f"{tag}_mem_time.jsonl",
                       "actions_sidecar": f"{tag}_actions.jsonl"},
    }
    with open(os.path.join(results_dir, f"{tag}_meta.json"), "w") as f:
        json.dump(result, f, indent=2)

    m = result["mem_stats"]; fit = result["mem_fit"]
    print(f"\n[τ={tau}] DONE elapsed={result['elapsed_sec']:.1f}s cost=${result['cost']['estimated_usd']:.3f}", flush=True)
    print(f"  mem gini={m['gini']:.3f} skew={m['skew']:.2f} max/med={m['max_over_median']:.1f} n_uniq={m['n_unique']}", flush=True)
    print(f"  α̂={fit.get('alpha_hat','n/a')} xmin={fit.get('xmin_hat','n/a')} ln_σ={fit.get('lognormal_sigma','n/a')}", flush=True)
    print(f"  policy: llm_pick={llm_pick_rate*100:.1f}% entropy={entropy:.2f} same_prev={same_prev*100:.1f}%", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--n_nodes", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reservoir_capacity", type=int, default=200)
    parser.add_argument("--K_pool", type=int, default=10)
    parser.add_argument("--M_pass", type=int, default=3)
    parser.add_argument("--budget_usd", type=float, default=1.0)
    parser.add_argument("--graph_type", type=str, default="scale_free")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_one_tau_llm(args.tau, args.n_nodes, args.n_steps, args.seed,
                     args.reservoir_capacity, args.K_pool, args.M_pass,
                     args.out_dir, args.budget_usd, args.graph_type)


if __name__ == "__main__":
    main()
