#!/usr/bin/env python3
"""
Generate per-phase maze sample JSON for the docs/ interactive carousel.

Reuses `extract_samples` from generate_maze_gallery.py so the
"===== Sample N/Total =====" parsing logic lives in exactly one place.

Usage:
    python generate_maze_carousel_data.py
    python generate_maze_carousel_data.py -o docs/data
    python generate_maze_carousel_data.py --phase fixed4x4=path/to/samples.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_maze_gallery import extract_samples

# Mirrors KNOWN_PHASE_ORDER in asciimaze/maze/validate.py. Each entry points
# at that phase's gguf (deployed, quantized model) sample file.
DEFAULT_PHASE_FILES = {
    "fixed4x4": "asciimaze/fixed4x4/outputs/gguf_samples.txt",
    "varNxN": "asciimaze/varNxN/outputs/gguf_samples_6x6.txt",
    "varNxM": "asciimaze/varNxM/outputs/gguf_samples_4x6.txt",
    "varNxM_rndSE": "asciimaze/varNxM_rndSE/outputs/gguf_samples_4x6.txt",
}


def parse_phase_override(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=PATH, received: {raw!r}"
        )

    name, path = raw.split("=", 1)
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate docs/data/{phase}.json files for the interactive "
            "maze carousel."
        )
    )

    parser.add_argument(
        "--phase",
        action="append",
        type=parse_phase_override,
        default=[],
        metavar="NAME=PATH",
        help=(
            "Override or add a phase's source samples file. May be "
            "repeated. Default phases: "
            + ", ".join(DEFAULT_PHASE_FILES)
        ),
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("docs/data"),
        help="Output directory for the phase JSON files. Default: docs/data",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    phase_files = dict(DEFAULT_PHASE_FILES)
    phase_files.update(args.phase)

    args.outdir.mkdir(parents=True, exist_ok=True)

    for phase, source in phase_files.items():
        source_path = Path(source)

        if not source_path.is_file():
            raise SystemExit(f"Input file does not exist: {source_path}")

        text = source_path.read_text(encoding="utf-8")
        samples = extract_samples(text)

        output_path = args.outdir / f"{phase}.json"
        output_path.write_text(
            json.dumps(
                {
                    "phase": phase,
                    "sourceFile": source_path.name,
                    "samples": samples,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(
            f"{phase}: {len(samples)} samples from {source_path} "
            f"-> {output_path}"
        )


if __name__ == "__main__":
    main()
