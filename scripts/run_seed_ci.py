"""
Seed confidence intervals for CTWM and graph-memory baselines.
"""

from __future__ import annotations
import argparse, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_ctwm_comparison import run_method  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["B7_GraphMemory", "B8_CTWM"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n_nodes", type=int, default=100)
    p.add_argument("--n_steps", type=int, default=2000)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--budget_usd", type=float, default=0.15)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    run_method(
        method_name=args.method, n_nodes=args.n_nodes, n_steps=args.n_steps,
        seed=args.seed, budget_usd=args.budget_usd, tau=args.tau,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
