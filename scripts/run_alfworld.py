from __future__ import annotations
import argparse, hashlib, json, os, random, re, sys, time
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldmodelsoc.memory.backends_ctwm import B7_GraphMemory, B8_CTWM, MemoryRecord, TokenCodec, TransitionStore, pack_context
from worldmodelsoc.llm_config import LLM_MODEL, make_openai_client
from worldmodelsoc.pipeline.modules import ChatRecorder, paired_summary, parse_object, provenance, summary_function, write_json

PRICE_PROMPT_PER_1M = 0.15
PRICE_COMPL_PER_1M = 0.60

TASK_TYPE_IDS = {
    "pick_and_place_simple": 1,
    "look_at_obj_in_light": 2,
    "pick_clean_then_place_in_recep": 3,
    "pick_heat_then_place_in_recep": 4,
    "pick_cool_then_place_in_recep": 5,
    "pick_two_obj_and_place": 6,
}
TASK_TYPES_ORDER = [
    "pick_and_place_simple", "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep",
    "look_at_obj_in_light", "pick_two_obj_and_place",
]


def make_client():
    return make_openai_client()


class CostAccountant:
    def __init__(self, budget=2.0):
        self.tokens_prompt = 0; self.tokens_completion = 0; self.api_calls = 0
        self.budget = budget; self.fallback_count = 0; self.error_count = 0
    def add(self, p, c):
        self.tokens_prompt += p; self.tokens_completion += c; self.api_calls += 1
    def cost(self):
        return (self.tokens_prompt/1e6)*PRICE_PROMPT_PER_1M + (self.tokens_completion/1e6)*PRICE_COMPL_PER_1M


def clean_text(text: str) -> str:
    if not text: return text
    text = re.sub(r"(_bar_[_\w]+)+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def state_id(obs: str) -> str:
    return f"s_{hashlib.md5(obs[:200].strip().lower().encode()).hexdigest()[:8]}"


def canonical_action(a: str) -> str:
    return re.sub(r"\s+", "_", a.strip().lower())[:40]


def build_alfredworld_config(
    data_root: str,
    task_type: str,
    max_steps: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    data_root = os.path.abspath(os.path.expanduser(data_root))
    tt_id = TASK_TYPE_IDS[task_type]
    logic_dir = os.path.join(os.path.dirname(data_root), "..", "logic")
    return {
        "env": {
            "goal_desc_human_anns_prob": 0.0,
            "domain_randomization": False,
            "task_types": [tt_id],
            "expert_type": "handcoded",
            "expert_timeout_steps": 150,
            "num_train_games": -1,
            "num_eval_games": -1,
            "max_nb_steps_per_episode": max_steps,
            "regen_game_files": False,
            "hybrid": {"start_eps": 100000, "thor_prob": 0.5, "eval_mode": "eval_out_of_distribution"},
            "thor": {"screen_width": 300, "screen_height": 300, "smooth_nav": False,
                     "save_frames_to_disk": False, "save_frames_path": "./videos/"},
        },
        "dataset": {
            "data_path": data_root,
            "eval_id_data_path": data_root,
            "eval_ood_data_path": data_root,
            "num_train_games": -1, "num_eval_games": -1,
        },
        "logic": {
            "domain": os.path.join(logic_dir, "alfred.pddl"),
            "grammar": os.path.join(logic_dir, "alfred.twl2"),
        },
        "training": {"batch_size": 1},
        "general": {"training_method": "dagger", "random_seed": seed, "visdom": False},
        "dagger": {
            "training": {
                "max_nb_steps_per_episode": max_steps,
                "batch_size": 1,
            },
            "aggregated_expert_dataset_size": 100000,
        },
        "rl": {
            "training": {"max_nb_steps_per_episode": max_steps, "batch_size": 1},
        },
        "checkpoint": {"report_frequency": 100, "experiment_tag": "worldmodelsoc"},
    }


def load_alfredworld_env_official(config: Dict[str, Any]):
    try:
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    except ImportError as exc:
        raise RuntimeError(
            'ALFWorld is optional; install it with pip install -e ".[alfworld]"'
        ) from exc
    env = AlfredTWEnv(config, train_eval="eval_in_distribution")
    return env


def llm_react_step(client, obs, admissible, task_desc, memory_context,
                   recent_states, recent_actions, accountant, fallback_rng,
                   retries=2):
    if not admissible:
        return 0, False, "", "no admissible"
    admissible_shown = admissible[:30]
    n_cmd = len(admissible_shown)
    system = ("You are an agent solving an embodied text task via ReAct (Reason + Act) pattern.\n"
              "In each step, first reason about the current state (Thought), then pick ONE admissible command index.\n"
              "Reply with EXACT format:\n"
              "Thought: <brief reasoning>\n"
              "Action: {\"cmd_idx\": <int>}\n"
              "No other text.")
    user = (f"task: {clean_text(task_desc)[:200]}\n"
            f"observation: {clean_text(obs)[:400]}\n"
            f"admissible[0..{n_cmd-1}]: {[clean_text(c) for c in admissible_shown]}\n"
            f"recent_states: {recent_states[-3:]}\n"
            f"recent_actions: {recent_actions[-3:]}\n"
            f"memory: {memory_context}")
    last_err = None
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
                max_completion_tokens=150)
            if resp.usage:
                accountant.add(resp.usage.prompt_tokens or 0, resp.usage.completion_tokens or 0)
            content = (resp.choices[0].message.content or "").strip()
            thought = ""
            m_t = re.search(r"Thought:\s*(.+?)(?:\n|Action:)", content, re.DOTALL)
            if m_t: thought = m_t.group(1).strip()[:300]
            m_a = re.search(r"Action:\s*\{[^{}]*\}", content)
            candidate = m_a.group(0)[len("Action:"):].strip() if m_a else content
            m_j = re.search(r"\{[^{}]*\}", candidate)
            if m_j:
                try:
                    obj = json.loads(m_j.group(0))
                    idx = int(obj.get("cmd_idx", -1))
                    if 0 <= idx < n_cmd:
                        return idx, True, thought, ""
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            for m2 in re.findall(r"\d+", candidate):
                i = int(m2)
                if 0 <= i < n_cmd:
                    return i, True, thought, ""
            last_err = ValueError(f"parse: {content[:120]!r}")
        except Exception as e:
            last_err = e; time.sleep(1.0)
    accountant.fallback_count += 1
    accountant.error_count += 1
    return fallback_rng.randrange(n_cmd), False, "", str(last_err)[:200]


def run_one_episode(
    env,
    method_name,
    max_steps,
    budget_ep,
    tau,
    run_id,
    task_type,
    game_file,
    seed,
):
    t0 = time.time()
    obs_batch, infos = env.reset()
    obs = obs_batch[0] if isinstance(obs_batch, (list, tuple)) else obs_batch
    task_desc = clean_text(obs)[:400]

    client = make_client()
    accountant = CostAccountant(budget=budget_ep)
    fallback_rng = random.Random(seed)

    if method_name == "B7_GraphMemory":
        mem = B7_GraphMemory(episode_length=200, top_k=3); uses_entities = True
    elif method_name == "B8_CTWM":
        mem = B8_CTWM(
            tau=tau,
            core_pct=0.30,
            core_slots=3,
            tail_slots=2,
            capacity=200,
            seed=seed,
        )
        uses_entities = False
    else:
        raise ValueError(f"unknown: {method_name}")

    per_step_tokens, recent_states, recent_actions, trajectory = [], [], [], []
    won = False; steps = 0
    prev_sid = state_id(obs); recent_states.append(prev_sid)

    for step in range(max_steps):
        steps = step + 1
        admissible_raw = infos.get("admissible_commands", None)
        if admissible_raw is None: break
        if isinstance(admissible_raw, list) and admissible_raw and isinstance(admissible_raw[0], list):
            admissible = admissible_raw[0]
        else:
            admissible = admissible_raw
        if not admissible: break

        mem.retrieve_hints(prev_sid, step)
        mem_ctx = mem.context_string(prev_sid)

        prompt_before = accountant.tokens_prompt
        cmd_idx, is_llm, thought, _error = llm_react_step(
            client, obs, admissible, task_desc, mem_ctx,
            recent_states, recent_actions, accountant, fallback_rng)
        prompt_delta = accountant.tokens_prompt - prompt_before
        per_step_tokens.append(prompt_delta)

        cmd = admissible[cmd_idx]
        action_c = canonical_action(cmd)
        recent_actions.append(action_c)

        try:
            obs_batch, _reward, done, infos = env.step([cmd])
            obs_next = obs_batch[0] if isinstance(obs_batch, (list, tuple)) else obs_batch
            done_val = done[0] if isinstance(done, (list, tuple)) else done
        except Exception as e:
            print(f"  env.step exception: {e}", flush=True); break

        nxt_sid = state_id(obs_next)

        if uses_entities:
            words_p = set(re.findall(r"[a-z]+", clean_text(obs).lower()))
            words_n = set(re.findall(r"[a-z]+", clean_text(obs_next).lower()))
            mem.write_transition_with_entities(prev_sid, action_c, nxt_sid, step,
                                                entities_prev=list(words_p)[:5],
                                                entities_next=list(words_n)[:5])
        else:
            mem.write_transition(prev_sid, action_c, nxt_sid, step)

        trajectory.append({
            "schema_version": "0.1.2", "run_id": run_id, "event_type": "react_step",
            "event_seq": step, "step": step, "prev_state": prev_sid,
            "action_idx": cmd_idx, "action_raw": cmd, "action_canonical": action_c,
            "next_state": nxt_sid, "is_llm_picked": is_llm,
            "thought": thought, "prompt_tokens_actual": prompt_delta,
        })

        obs = obs_next; prev_sid = nxt_sid; recent_states.append(nxt_sid)

        won_val = infos.get("won", False)
        if isinstance(won_val, list): won_val = won_val[0] if won_val else False
        if won_val: won = True; break
        if done_val: break
        if accountant.cost() >= budget_ep:
            print(f"  [BUDGET] hit ${accountant.cost():.3f}, stop step {steps}", flush=True); break

    mem_freqs = list(mem.access_counter.values())
    return {
        "run_id": run_id, "method": method_name, "game_file": game_file,
        "task_type": task_type, "won": bool(won), "steps": steps,
        "avg_tokens_actual": float(np.mean(per_step_tokens)) if per_step_tokens else 0.0,
        "cost_usd": accountant.cost(), "api_calls": accountant.api_calls,
        "fallback_count": accountant.fallback_count, "error_count": accountant.error_count,
        "n_unique_states": len(mem.unique_states_seen),
        "n_unique_trans": len(mem.unique_trans_seen),
        "n_mem_ids": len(mem_freqs),
        "mem_top_freq": max(mem_freqs) if mem_freqs else 0,
        "elapsed_sec": time.time() - t0,
        "trajectory": trajectory,
    }


def legacy_main(argv=None):
    p = argparse.ArgumentParser(epilog='Paired CTMC experiments and demonstration settings: --mode paired --help.')
    p.add_argument("--data_root", default="~/.cache/alfworld/json_2.1.1/valid_seen")
    p.add_argument("--n_ep_per_type", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--methods", nargs="+", default=["B7_GraphMemory", "B8_CTWM"])
    p.add_argument("--budget_ep", type=float, default=0.35)
    p.add_argument("--budget_total", type=float, default=6.0)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    results_dir = os.path.join(args.out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    random.seed(args.seed)
    all_results = []
    total_cost = 0.0

    for method in args.methods:
        print(f"\n===== METHOD {method} =====", flush=True)
        for tt in TASK_TYPES_ORDER:
            config = build_alfredworld_config(
                args.data_root,
                tt,
                args.max_steps,
                seed=args.seed,
            )
            try:
                env = load_alfredworld_env_official(config)
                game_files = sorted(env.game_files)[:args.n_ep_per_type]
                print(f"[TASKS] {tt}: {len(env.game_files)} available, using first {len(game_files)}", flush=True)
            except Exception as e:
                print(f"  ERROR loading env for {tt}: {e}", flush=True)
                import traceback; traceback.print_exc(); continue

            for i in range(len(game_files)):
                if total_cost >= args.budget_total:
                    print("  [BUDGET total] stop", flush=True); break
                run_id = f"{method}_{tt}_{i:02d}"
                print(f"[EP] {method} / {tt} / #{i+1}", flush=True)
                episode_env = None
                try:
                    env.game_files = [game_files[i]]
                    episode_env = env.init_env(batch_size=1)
                    r = run_one_episode(episode_env, method, args.max_steps, args.budget_ep,
                                         args.tau, run_id, tt, game_files[i],
                                         seed=args.seed + i)
                    all_results.append(r)
                    total_cost += r["cost_usd"]
                    print(f"    won={r['won']} steps={r['steps']} cost=${r['cost_usd']:.3f} "
                          f"tokens/step={r['avg_tokens_actual']:.0f} "
                          f"n_states={r['n_unique_states']} n_trans={r['n_unique_trans']}", flush=True)
                except Exception as e:
                    print(f"  ERROR ep: {e}", flush=True)
                    import traceback; traceback.print_exc()
                    all_results.append({"method": method, "task_type": tt, "error": str(e)})
                finally:
                    if episode_env is not None and hasattr(episode_env, "close"):
                        episode_env.close()
            if total_cost >= args.budget_total: break
        if total_cost >= args.budget_total: break

    for r in all_results:
        if "run_id" not in r: continue
        tag = r["run_id"]
        traj = r.pop("trajectory", None)
        if traj:
            with open(os.path.join(results_dir, f"{tag}_traj.jsonl"), "w") as f:
                for e in traj: f.write(json.dumps(e) + "\n")
        with open(os.path.join(results_dir, f"{tag}_meta.json"), "w") as f:
            json.dump(r, f, indent=2)

    def agg(m):
        rs = [r for r in all_results if r.get("method") == m and "won" in r]
        if not rs: return {"method": m, "n": 0}
        return {
            "method": m, "n_episodes": len(rs),
            "success_count": sum(1 for r in rs if r["won"]),
            "success_rate": sum(1 for r in rs if r["won"]) / len(rs),
            "mean_steps": float(np.mean([r["steps"] for r in rs])),
            "mean_tokens_actual": float(np.mean([r["avg_tokens_actual"] for r in rs])),
            "mean_cost_ep": float(np.mean([r["cost_usd"] for r in rs])),
            "total_cost": float(sum(r["cost_usd"] for r in rs)),
            "mean_unique_states": float(np.mean([r["n_unique_states"] for r in rs])),
            "mean_unique_trans": float(np.mean([r["n_unique_trans"] for r in rs])),
            "by_task_type": {
                t: {
                    "n": sum(1 for r in rs if r.get("task_type") == t),
                    "wins": sum(1 for r in rs if r.get("task_type") == t and r["won"]),
                    "mean_tokens": float(np.mean([r["avg_tokens_actual"] for r in rs if r.get("task_type") == t])) if [r for r in rs if r.get("task_type") == t] else 0.0,
                } for t in TASK_TYPES_ORDER},
        }

    summary = {
        "study": "alfworld_external_validity",
        "config": vars(args),
        "total_cost_usd": total_cost,
        "n_episodes_run": len([r for r in all_results if "won" in r]),
        "aggregates": {m: agg(m) for m in args.methods},
        "all_results_meta": [{k: v for k, v in r.items() if k != "trajectory"} for r in all_results],
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n===== SUMMARY =====")
    for m in args.methods:
        a = summary["aggregates"][m]
        if a.get("n_episodes", 0) > 0:
            print(f"  {m}: {a['success_count']}/{a['n_episodes']} ({a['success_rate']*100:.1f}%), "
                  f"tokens/step={a['mean_tokens_actual']:.0f}, cost=${a['total_cost']:.3f}", flush=True)
    print(f"\nTotal cost: ${total_cost:.3f}")


def first(value):
    return value[0] if isinstance(value, (list, tuple)) else value


def observation_id(observation):
    return hashlib.sha256(clean_text(observation).encode()).hexdigest()[:16]


def entities(observation):
    return sorted(set(re.findall(r"[a-z]+", observation.lower())) - {"the", "a", "an", "you", "to", "in", "on", "is", "and", "of", "with", "are", "it"})


def run_episode(env, chat, config, demonstrations, output):
    observations, infos = env.reset()
    observation = str(first(observations))
    task = observation
    store = TransitionStore(config["capacity"])
    codec = TokenCodec(config["tokenizer"])
    won = False
    rows = []
    start_usage = chat.usage.copy()
    start_calls = len(chat.calls)
    with output.open("w", encoding="utf-8") as handle:
        for step in range(config["max_steps"]):
            admissible = list(first(infos.get("admissible_commands", [[]])))
            if not admissible:
                break
            state = observation_id(observation)
            ranked = store.rank(state, entities(observation), config["method"])
            context, allocation = pack_context(ranked, config["method"], config["serialization"], config["memory_budget"], config["core_size"], config["tau"], codec, summary_function(chat, task, config["summary_input_tokens"]))
            messages = [{"role": "system", "content": 'Solve the embodied task using the supplied observation and memory. Think briefly, then choose an admissible action. Return JSON with keys "thought" and "action_idx"; action_idx must index the given list.'}]
            for example in demonstrations[:config["n_shots"]]:
                messages.extend(example["messages"])
            messages.append({"role": "user", "content": f"Task: {task}\nObservation: {observation}\nMemory:\n{context}\nAdmissible commands: {json.dumps(admissible)}"})
            response = parse_object(chat.ask(messages, phase="policy", max_tokens=200))
            index = response.get("action_idx")
            if type(index) is not int or not 0 <= index < len(admissible):
                raise ValueError(f"invalid admissible action at step {step}")
            action = admissible[index]
            observations, scores, dones, infos = env.step([action])
            next_observation = str(first(observations))
            next_state = observation_id(next_observation)
            row = {"agent_step": step, "observation": observation, "prev": state, "action": action, "next": next_state, "next_observation": next_observation, "allocation": allocation, "score": float(first(scores)), "done": bool(first(dones))}
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            rows.append(row)
            store.write(state, action, next_state, step, sorted(set(entities(observation) + entities(next_observation))))
            identifier = f"{state}::{action}::{next_state}"
            record = store.history[-1]
            text = f"Observation: {clean_text(observation)}\nAction: {action}\nNext observation: {clean_text(next_observation)}"
            if identifier in store.records:
                store.records[identifier] = MemoryRecord(identifier, text, record.score, record.metadata)
            historical = store.history[-1]
            store.history[-1] = MemoryRecord(historical.memory_id, text, metadata=historical.metadata)
            observation = next_observation
            won = bool(first(infos.get("won", [False])))
            if row["done"]:
                break
    usage = {key: chat.usage[key] - start_usage[key] for key in chat.usage}
    operational = usage.get("policy_prompt_tokens", 0) + usage.get("summary_prompt_tokens", 0)
    return {
        "method": config["method"], "pair_id": config["pair_id"], "task_type": config["task_type"],
        "n_shots": config["n_shots"], "steps": len(rows), "success": int(won),
        "policy_prompt_tokens": usage.get("policy_prompt_tokens", 0), "operational_prompt_tokens": operational,
        "policy_prompt_tokens_per_step": usage.get("policy_prompt_tokens", 0) / len(rows) if rows else None,
        "operational_prompt_tokens_per_step": operational / len(rows) if rows else None,
        "usage": usage, "api_calls": len(chat.calls) - start_calls,
    }


def paired_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=["graph_memory", "ctmc"], default=["graph_memory", "ctmc"])
    parser.add_argument("--task_types", nargs="+", choices=TASK_TYPES_ORDER, default=TASK_TYPES_ORDER)
    parser.add_argument("--episodes_per_type", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--n_shots", type=int, nargs="+", default=[0])
    parser.add_argument("--demonstrations", type=Path, help="JSON list of demonstration objects with user/assistant messages")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--core_size", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=200)
    parser.add_argument("--memory_budget", type=int, default=512)
    parser.add_argument("--summary_input_tokens", type=int, default=4000)
    parser.add_argument("--serialization", choices=["compact", "verbose"], default="compact")
    parser.add_argument("--tokenizer", default="cl100k_base")
    parser.add_argument("--model")
    parser.add_argument("--max_calls", type=int, default=100000)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.methods) != len(set(args.methods)) or len(args.n_shots) != len(set(args.n_shots)):
        parser.error("methods and n_shots must be unique")
    demos = json.loads(args.demonstrations.read_text(encoding="utf-8")) if args.demonstrations else []
    if any(shots < 0 or shots > len(demos) for shots in args.n_shots):
        parser.error("n_shots exceeds the supplied demonstrations")
    if args.episodes_per_type < 1 or args.max_steps < 1 or args.summary_input_tokens < 1:
        parser.error("episode, step, and summary-input limits must be positive")
    for example in demos:
        if not example.get("messages") or any(message.get("role") not in ("user", "assistant") or not isinstance(message.get("content"), str) for message in example["messages"]):
            parser.error("demonstrations must contain user/assistant text messages")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.out_dir / "config.json", provenance(config))
    chat = ChatRecorder(args.out_dir / "api_calls.jsonl", args.model, args.max_calls)
    results = []
    for task_type in args.task_types:
        settings = build_alfredworld_config(str(args.data_root.expanduser()), task_type, args.max_steps, args.seed)
        catalog = load_alfredworld_env_official(settings)
        games = sorted(catalog.game_files)[:args.episodes_per_type]
        if not games:
            raise ValueError(f"no games found for {task_type}")
        for number, game in enumerate(games):
            for shots in args.n_shots:
                pair_id = json.dumps([str(game), args.seed, args.max_steps, shots])
                for method in args.methods:
                    setting = build_alfredworld_config(str(args.data_root.expanduser()), task_type, args.max_steps, args.seed + number)
                    loader = load_alfredworld_env_official(setting)
                    loader.game_files = [game]
                    env = loader.init_env(batch_size=1)
                    episode_config = {**config, "method": method, "n_shots": shots, "task_type": task_type, "pair_id": pair_id}
                    output = args.out_dir / f"{task_type}_{number}_{shots}shot_{method}.jsonl"
                    try:
                        result = run_episode(env, chat, episode_config, demos, output)
                        results.append(result)
                        write_json(args.out_dir / "episodes.json", results)
                    finally:
                        if hasattr(env, "close"):
                            env.close()
    metrics = ["success", "policy_prompt_tokens", "operational_prompt_tokens", "policy_prompt_tokens_per_step", "operational_prompt_tokens_per_step"]
    comparisons = [dict(n_shots=shots, **paired_summary([row for row in results if row["n_shots"] == shots], "graph_memory", "ctmc", metrics, args.seed)) for shots in args.n_shots]
    write_json(args.out_dir / "summary.json", {"paired_comparisons": comparisons, "api_usage": dict(chat.usage)})


def main(argv=None):
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument('--mode', choices=['legacy', 'paired'], default='legacy')
    selected, remaining = selector.parse_known_args(argv)
    if selected.mode == "paired":
        return paired_main(remaining)
    return legacy_main(remaining)


if __name__ == "__main__":
    main()
