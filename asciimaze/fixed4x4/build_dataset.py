#!/usr/bin/env python3
"""Generate the train/val/test JSONL splits for this dataset.

Example:
    python -m asciimaze.fixed4x4.build_dataset --n 500 --seed 0
"""

import argparse
import json
from pathlib import Path

from .dataset import build_sample
from .paths import BASE_DIR

DEFAULT_SPLIT = {"train": 0.9, "val": 0.05, "test": 0.05}


def split_counts(n: int, split: dict[str, float]) -> dict[str, int]:
    counts = {name: int(n * fraction) for name, fraction in split.items()}
    counts["train"] += n - sum(counts.values())
    return counts


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Total number of samples across all splits.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Each sample gets seed + index.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: <this folder>/data",
    )

    args = parser.parse_args()

    n = args.n
    seed = args.seed

    out_dir = args.out or BASE_DIR / "data"
    counts = split_counts(n, DEFAULT_SPLIT)

    index = 0

    for split_name, count in counts.items():
        records = [
            build_sample(seed=seed + index + offset) for offset in range(count)
        ]
        index += count

        write_jsonl(out_dir / f"{split_name}.jsonl", records)
        print(f"{split_name}: {count} samples -> {out_dir / f'{split_name}.jsonl'}")


if __name__ == "__main__":
    main()
