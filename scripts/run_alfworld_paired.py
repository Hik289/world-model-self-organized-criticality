from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_alfworld import TASK_TYPES_ORDER, build_alfredworld_config, clean_text, load_alfredworld_env_official
from run_longmemeval import summary_function
from worldmodelsoc.experiment_io import ChatRecorder, paired_summary, provenance, write_json
from worldmodelsoc.memory.ctmc import MemoryRecord, TokenCodec, pack_context
from worldmodelsoc.synthetic_experiments import TransitionStore, parse_object


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


def main():
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
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
