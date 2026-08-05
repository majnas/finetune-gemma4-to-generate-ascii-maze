#!/usr/bin/env python3
"""Generate fixed4x4_solver train/val/test JSONL splits.

Example:
    python -m asciimaze.fixed4x4_solver.build_dataset --n 7000 --seed 1934
"""

import argparse
import json
from pathlib import Path

from .dataset import build_sample
from .paths import BASE_DIR

DEFAULT_SPLIT = {"train": 0.9, "val": 0.05, "test": 0.05}


def split_counts(n: int) -> dict[str, int]:
    if n < 0:
        raise ValueError("n must be non-negative")
    counts = {name: int(n * fraction) for name, fraction in DEFAULT_SPLIT.items()}
    counts["train"] += n - sum(counts.values())
    return counts


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or BASE_DIR / "data"
    counts = split_counts(args.n)
    index = 0
    for split_name, count in counts.items():
        records = [
            build_sample(args.seed + index + offset)
            for offset in range(count)
        ]
        index += count
        path = out_dir / f"{split_name}.jsonl"
        write_jsonl(path, records)
        print(f"{split_name}: {count} samples -> {path}")


if __name__ == "__main__":
    main()
