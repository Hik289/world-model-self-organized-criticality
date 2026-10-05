from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldmodelsoc.experiment_io import provenance, read_records, write_json
from worldmodelsoc.tail_audit import audit_counts, psd_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=Path, nargs="+", required=True)
    parser.add_argument("--count_kind", choices=["state_visits", "memory_reads", "memory_accesses", "transition_visits"], required=True)
    parser.add_argument("--field")
    parser.add_argument("--min_tail", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xmin", type=float)
    parser.add_argument("--timeseries", type=Path)
    parser.add_argument("--series_field", default="access_count")
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--min_frequency", type=float)
    parser.add_argument("--max_frequency", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    output = {"provenance": provenance({key: str(value) for key, value in vars(args).items()}), "count_kind": args.count_kind, "audits": []}
    for path in args.counts:
        values = json.loads(path.read_text(encoding="utf-8"))
        if args.field:
            for part in args.field.split("."):
                values = values[part]
        values = list(values.values()) if isinstance(values, dict) else values
        output["audits"].append({"source": str(path), **audit_counts(values, args.min_tail, args.bootstrap, args.seed, args.xmin)})
        write_json(args.output, output)
    if args.timeseries:
        series = read_records(args.timeseries)
        output["temporal_audit"] = {
            "source": str(args.timeseries), "observable": args.series_field,
            **psd_audit([record[args.series_field] for record in series], args.sample_rate, args.min_frequency, args.max_frequency),
        }
    write_json(args.output, output)


if __name__ == "__main__":
    main()
