# Artifact Guide

This guide maps the public `world-model-self-organized-criticality` repository to a reviewer-friendly artifact workflow for `World Models Are Heavy-Tailed`. It is meant to make the release easier to inspect in the style of ICML, ICLR, NeurIPS, and similar artifact-review processes.

## What To Inspect First

- `src/`: Core source code and reusable implementations.
- `scripts/`: Command-line entry points for experiments, analysis, or reproduction.
- `assets/`: README and paper-facing visual assets.

## Environment Files

- `requirements.txt`: Primary Python dependency list.
- `pyproject.toml`: Package metadata and optional extras when available.

## Minimal Verification

Run these checks in a fresh environment before launching expensive jobs:

```bash
python -m compileall -q .
python scripts/run_sanity.py
```

## Reproduction And Analysis Entry Points

These are the main tracked files to inspect for paper-scale or benchmark-scale reproduction. Some require arguments, credentials, downloaded benchmarks, or local data paths described in the README.

- `python scripts/run_alfworld.py`
- `python scripts/run_ctwm_comparison.py`
- `python scripts/run_frozen_replay.py`
- `python scripts/run_llm_policy.py`
- `python scripts/run_method_comparison.py`
- `python scripts/run_random_walk_scaling.py`
- `python scripts/run_sanity.py`
- `python scripts/run_seed_ci.py`
- `python scripts/run_tau_sweep.py`
- `python scripts/run_topology_control.py`

## Figure Assets

- `assets/figures/intuition_lognormal_to_tpl.png`
- `assets/figures/pipeline_ctwm.png`
- `assets/figures/pipeline_llm_policy.png`
- `assets/figures/pipeline_retriever_ladder.png`

## Data, Credentials, And Generated Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reviewer Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
