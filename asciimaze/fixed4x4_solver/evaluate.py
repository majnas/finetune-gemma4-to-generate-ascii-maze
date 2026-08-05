#!/usr/bin/env python3
"""Score fixed4x4_solver predictions stored in JSONL.

Each prediction record must contain ``prediction`` and either ``expected`` or
the original dataset record's ``conversations`` and ``meta`` fields.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from .dataset import format_path


def parse_prediction(text: str) -> list[str]:
    """Parse the strict S,cell,...,E answer format."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    parts = [part.strip().upper() for part in text.split(",")]
    if len(parts) < 2 or parts[0] != "S" or parts[-1] != "E":
        raise ValueError("answer must start with S and end with E")
    if any(not part for part in parts):
        raise ValueError("empty path item")
    for label in parts[1:-1]:
        if len(label) != 2 or label[0] not in "ABCD" or label[1] not in "1234":
            raise ValueError(f"invalid cell label: {label}")
    return parts


def expected_from_record(record: dict) -> str:
    if "expected" in record:
        return record["expected"]
    if "meta" in record and "path_cells" in record["meta"]:
        cells = [tuple(cell) for cell in record["meta"]["path_cells"]]
        return format_path(cells)
    return record["conversations"][-1]["content"]


def score_record(record: dict) -> tuple[bool, str | None]:
    expected = expected_from_record(record)
    try:
        predicted_parts = parse_prediction(record["prediction"])
    except (KeyError, TypeError, ValueError) as error:
        return False, str(error)
    expected_parts = parse_prediction(expected)
    if predicted_parts != expected_parts:
        return False, "wrong path"
    return True, None


def evaluate_file(path: Path) -> dict:
    total = correct = 0
    failures: Counter[str] = Counter()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            ok, error = score_record(json.loads(line))
            if ok:
                correct += 1
            else:
                failures[error or "unknown error"] += 1
    return {"total": total, "correct": correct, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    result = evaluate_file(args.predictions)
    accuracy = result["correct"] / result["total"] if result["total"] else 0.0
    print(f"exact match: {result['correct']}/{result['total']} ({accuracy:.2%})")
    for reason, count in result["failures"].most_common():
        print(f"{count:4d}  {reason}")


if __name__ == "__main__":
    main()
