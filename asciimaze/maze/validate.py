"""Validate free-generated maze samples against the maze format rules.

Checks two independent things for each sample produced by a
`gemma4_generate_100_samples_with_*.py` run: (1) format - correct size,
single S/E, fully enclosed boundary; (2) solvability - reusing this
package's own `solve_maze` (not a reimplementation) to confirm a path
actually exists from S to E. Works across all experiment phases (fixed4x4,
varNxN, varNxM, varNxM_rndSE, ...) since they share one ASCII format.

Usage:
    python -m asciimaze.maze.validate <file.txt> [file2.txt ...]
        Detailed report for the given file(s) - failure-reason breakdown
        and the first failing sample.

    python -m asciimaze.maze.validate
        No path: auto-discovers every asciimaze/*/outputs/*.txt, groups by
        phase and by source (lora/merged/gguf, from the filename), and
        prints one valid/total summary table.

Each samples file's own header line ("# prompt : Generate a random, valid
R×C ASCII maze ...") supplies the expected size - no --rows/--columns flag
needed unless a file's header is missing or you want to force a check
against a different size.
"""

import argparse
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .generator import Cell
from .solver import solve_maze

SAMPLE_DELIM_RE = re.compile(r"===== Sample \d+/\d+ =====")
CODE_BLOCK_RE = re.compile(r"```\n(.*?)\n```", re.DOTALL)
HEADER_SIZE_RE = re.compile(r"valid\s+(\d+)[×x](\d+)\s+ASCII maze")
SOURCE_PREFIX_RE = re.compile(r"^([^_]+)_samples")

KNOWN_PHASE_ORDER = ["fixed4x4", "varNxN", "varNxM", "varNxM_rndSE"]
KNOWN_SOURCE_ORDER = ["lora", "merged", "gguf"]


def detect_expected_size(text: str) -> tuple[int, int] | None:
    """Parse the "R×C" size out of the file's own "# prompt : ..." header."""
    match = HEADER_SIZE_RE.search(text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def extract_samples(text: str) -> list[str | None]:
    blocks = SAMPLE_DELIM_RE.split(text)[1:]
    samples = []
    for block in blocks:
        match = CODE_BLOCK_RE.search(block)
        samples.append(match.group(1) if match else None)
    return samples


def _wall_open_positions(wall_line: str, columns: int) -> list[bool]:
    opens = []
    for c in range(columns):
        seg_start = 3 + 4 * c
        seg = wall_line[seg_start:seg_start + 3] if len(wall_line) >= seg_start + 3 else "---"
        opens.append(seg.strip() == "")
    return opens


def parse_maze(text: str, expected_rows: int, expected_columns: int) -> dict:
    """Parse one rendered maze block and check format + solvability.

    Returns {"ok": bool, "errors": [...], "rows": int, "columns": int}.
    """
    lines = [l for l in text.split("\n") if l.strip() != ""]
    if len(lines) < 3:
        return {"ok": False, "errors": ["too few lines"], "rows": 0, "columns": 0}

    top_wall = lines[1]
    rest = lines[2:]
    row_lines = rest[0::2]
    wall_lines = [top_wall] + rest[1::2]

    rows = len(row_lines)
    columns = top_wall.count("+") - 1
    errors = []

    if rows != expected_rows or columns != expected_columns:
        errors.append(f"size mismatch: got {rows}x{columns}, expected {expected_rows}x{expected_columns}")
        return {"ok": False, "errors": errors, "rows": rows, "columns": columns}

    if len(wall_lines) != rows + 1:
        errors.append(f"wall/row line count mismatch: {len(wall_lines)} wall lines for {rows} rows")
        return {"ok": False, "errors": errors, "rows": rows, "columns": columns}

    if any(_wall_open_positions(top_wall, columns)):
        errors.append("top boundary has an opening")
    if any(_wall_open_positions(wall_lines[-1], columns)):
        errors.append("bottom boundary has an opening")

    start: Cell | None = None
    end: Cell | None = None
    s_count = e_count = 0
    h_open = [[False] * max(columns - 1, 0) for _ in range(rows)]
    v_open = [[False] * columns for _ in range(max(rows - 1, 0))]

    for r, line in enumerate(row_lines):
        left_wall = line[2] if len(line) > 2 else None
        if left_wall != "|":
            errors.append(f"row {r}: left boundary open/malformed ({left_wall!r})")
        for c in range(columns):
            content_start = 3 + 4 * c
            content = line[content_start:content_start + 3].strip() if len(line) >= content_start + 3 else "?"
            if content == "S":
                s_count += 1
                start = (r, c)
            elif content == "E":
                e_count += 1
                end = (r, c)
            elif content not in ("", "?"):
                errors.append(f"row {r} col {c}: unexpected cell content {content!r}")
            if c < columns - 1:
                wall_pos = 2 + 4 * (c + 1)
                wall_char = line[wall_pos] if len(line) > wall_pos else None
                h_open[r][c] = (wall_char == " ")
        right_wall_pos = 2 + 4 * columns
        right_wall = line[right_wall_pos] if len(line) > right_wall_pos else None
        if right_wall != "|":
            errors.append(f"row {r}: right boundary open/malformed ({right_wall!r})")

    for r in range(rows - 1):
        wl = wall_lines[r + 1]
        for c in range(columns):
            seg_start = 3 + 4 * c
            seg = wl[seg_start:seg_start + 3] if len(wl) >= seg_start + 3 else "---"
            v_open[r][c] = (seg.strip() == "")

    if s_count != 1:
        errors.append(f"expected exactly one S, found {s_count}")
    if e_count != 1:
        errors.append(f"expected exactly one E, found {e_count}")

    if s_count == 1 and e_count == 1:
        # Rebuild the same maze[][] shape asciimaze.maze.generator produces
        # (a grid of direction-sets), then reuse the package's own
        # solve_maze - not a reimplemented BFS - to check connectivity.
        maze: list[list[set[str]]] = [[set() for _ in range(columns)] for _ in range(rows)]
        for r in range(rows):
            for c in range(columns - 1):
                if h_open[r][c]:
                    maze[r][c].add("E")
                    maze[r][c + 1].add("W")
        for r in range(rows - 1):
            for c in range(columns):
                if v_open[r][c]:
                    maze[r][c].add("S")
                    maze[r + 1][c].add("N")

        try:
            solve_maze(maze, start, end)
        except KeyError:
            errors.append("no path from S to E")

    return {"ok": len(errors) == 0, "errors": errors, "rows": rows, "columns": columns}


def validate_sample(text: str, expected_rows: int, expected_columns: int) -> dict:
    return parse_maze(text, expected_rows, expected_columns)


def validate_file(path: str, rows: int | None = None, columns: int | None = None) -> dict:
    """Validate every sample in one output file. Returns aggregate stats."""
    text = open(path).read()
    detected = detect_expected_size(text)
    expected_rows = rows if rows is not None else (detected[0] if detected else None)
    expected_columns = columns if columns is not None else (detected[1] if detected else None)
    if expected_rows is None or expected_columns is None:
        raise ValueError(
            f"{path}: couldn't detect expected size from the header - pass --rows/--columns"
        )

    samples = extract_samples(text)
    results = []
    for s in samples:
        if s is None:
            results.append({"ok": False, "errors": ["no code block found"], "rows": 0, "columns": 0})
            continue
        results.append(validate_sample(s, expected_rows, expected_columns))

    n = len(results)
    valid = sum(1 for r in results if r["ok"])
    disconnected_only = sum(1 for r in results if not r["ok"] and r["errors"] == ["no path from S to E"])
    format_broken = n - valid - disconnected_only
    distinct = len({s for s in samples if s})

    return {
        "path": path,
        "expected_rows": expected_rows,
        "expected_columns": expected_columns,
        "n": n,
        "valid": valid,
        "disconnected": disconnected_only,
        "format_broken": format_broken,
        "distinct": distinct,
        "results": results,
        "samples": samples,
    }


def discover_output_files(root: str = ".") -> list[str]:
    return sorted(glob.glob(os.path.join(root, "asciimaze", "*", "outputs", "*.txt")))


def phase_and_source(path: str) -> tuple[str, str]:
    parts = Path(path).parts
    phase = "?"
    if "asciimaze" in parts:
        idx = parts.index("asciimaze")
        if idx + 1 < len(parts):
            phase = parts[idx + 1]
    match = SOURCE_PREFIX_RE.match(Path(path).name)
    source = match.group(1) if match else "?"
    return phase, source


def print_detailed_report(path: str, rows: int | None, columns: int | None) -> None:
    stats = validate_file(path, rows, columns)
    print(f"\n=== {path} ===")
    print(
        f"samples: {stats['n']}  valid: {stats['valid']}  "
        f"well-formed-but-disconnected: {stats['disconnected']}  "
        f"format-broken: {stats['format_broken']}  distinct mazes: {stats['distinct']}"
    )

    error_counts: dict[str, int] = {}
    for r in stats["results"]:
        if not r["ok"]:
            for e in r["errors"]:
                key = re.sub(r"\d+", "N", e)
                error_counts[key] = error_counts.get(key, 0) + 1
    if error_counts:
        print("failure reasons:")
        for k, v in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {v:4d}  {k}")

    for i, r in enumerate(stats["results"]):
        if not r["ok"]:
            print(f"\nfirst failing sample (#{i + 1}):")
            print(f"  errors: {r['errors']}")
            print(stats["samples"][i])
            break


def print_summary_table(root: str = ".") -> None:
    files = discover_output_files(root)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    phases_seen: list[str] = []
    sources_seen: list[str] = []

    for path in files:
        phase, source = phase_and_source(path)
        try:
            stats = validate_file(path)
        except ValueError as e:
            print(f"skipping {path}: {e}")
            continue
        cells[(phase, source)].append(stats)
        if phase not in phases_seen:
            phases_seen.append(phase)
        if source not in sources_seen:
            sources_seen.append(source)

    phase_order = [p for p in KNOWN_PHASE_ORDER if p in phases_seen]
    phase_order += [p for p in phases_seen if p not in phase_order]
    source_order = [s for s in KNOWN_SOURCE_ORDER if s in sources_seen]
    source_order += [s for s in sources_seen if s not in source_order]

    if not phase_order:
        print("No output files found under asciimaze/*/outputs/*.txt")
        return

    table = Table(title="Maze sample validation (valid / total)")
    table.add_column("phase", style="bold")
    for source in source_order:
        table.add_column(source, justify="right")

    for phase in phase_order:
        row_cells = [phase]
        for source in source_order:
            group = cells.get((phase, source), [])
            if not group:
                row_cells.append("-")
                continue
            total_valid = sum(g["valid"] for g in group)
            total_n = sum(g["n"] for g in group)
            ratio = total_valid / total_n if total_n else 0.0
            style = "green" if ratio >= 0.9 else ("yellow" if ratio >= 0.5 else "red")
            row_cells.append(f"[{style}]{total_valid}/{total_n}[/{style}]")
        table.add_row(*row_cells)

    console = Console()
    console.print(table)
    console.print(
        "\n[bold]Score = valid / total[/bold]: correctly formatted "
        "[italic]and[/italic] solvable S→E. Low scores on varN* phases "
        "reflect held-out-size generalization, not a broken model."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Specific output file(s) to validate in detail.")
    parser.add_argument("--rows", type=int, default=None, help="Override the expected row count.")
    parser.add_argument("--columns", type=int, default=None, help="Override the expected column count.")
    args = parser.parse_args()

    if args.files:
        for path in args.files:
            print_detailed_report(path, args.rows, args.columns)
    else:
        print_summary_table()


if __name__ == "__main__":
    main()
