# Artifact Guide

Operational notes for reproducing `World Models Are Heavy-Tailed` from the public `world-model-self-organized-criticality` repository.

## Review Path

- `src/`: Core source code and reusable implementations.
- `scripts/`: Command-line entry points for experiments, analysis, or reproduction.
- `assets/`: README and paper-facing visual assets.

## Environment Files

- `requirements.txt`: Primary Python dependency list.
- `pyproject.toml`: Package metadata and optional extras when available.

## Smoke Checks

Run these checks before long jobs:

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

After configuring an LLM endpoint, run the API-backed pipeline check with:

```bash
python scripts/run_sanity.py
```

## Reproduction Entry Points

Main tracked entry points for paper-scale or benchmark-scale runs:

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

## Data And Outputs

- API-backed runs read credentials from environment variables or the untracked
  key files documented in the README; never commit real keys.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
