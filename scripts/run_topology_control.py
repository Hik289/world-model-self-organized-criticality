"""Topology control: run the LLM policy on non-scale-free graph families."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_tau_sweep import run_one_tau_llm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph_type", required=True,
                    choices=["uniform_degree", "modular", "exponential_degree", "scale_free"])
    p.add_argument("--n_nodes", type=int, default=100)
    p.add_argument("--n_steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--budget_usd", type=float, default=0.5)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_one_tau_llm(
        tau=args.tau, n_nodes=args.n_nodes, n_steps=args.n_steps, seed=args.seed,
        reservoir_capacity=200, K_pool=10, M_pass=3,
        out_dir=args.out_dir, budget_usd=args.budget_usd,
        graph_type=args.graph_type,
    )

if __name__ == "__main__":
    main()
