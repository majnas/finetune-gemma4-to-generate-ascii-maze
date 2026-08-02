import random

from .config import MAZE_CONFIG
from ..maze.endpoints import choose_endpoints
from ..maze.generator import generate_maze
from ..maze.render import render_maze
from ..maze.solver import solve_maze
from .prompts import build_prompt


def build_sample(seed: int, sizes: tuple[int, ...]) -> dict:
    """Generate one training record: a user prompt paired with the
    assistant maze completion, plus the ground-truth metadata needed
    later to validate a model's own output.

    `sizes` is passed in explicitly (rather than read from MAZE_CONFIG
    directly) since which sizes apply depends on which split - train/val
    and test draw from disjoint size sets. Rows and columns are drawn
    independently and re-rolled until unequal - rows == columns is
    asciimaze.varNxN's square-maze case, not this one. MAZE_CONFIG.random_
    endpoints=True here, so start/end land at two random distinct cells
    rather than fixed opposite corners."""
    rng = random.Random(seed)

    rows = rng.choice(sizes)
    columns = rng.choice(sizes)
    while columns == rows:
        columns = rng.choice(sizes)

    maze = generate_maze(rows=rows, columns=columns, rng=rng)
    start, end = choose_endpoints(rows, columns, rng, MAZE_CONFIG.random_endpoints)

    prompt = build_prompt(rows, columns, start, end)
    maze_text = render_maze(maze, start, end)
    completion = f"```\n{maze_text}\n```"

    meta = {
        "seed": seed,
        "rows": rows,
        "columns": columns,
        "start": list(start),
        "end": list(end),
    }

    if MAZE_CONFIG.include_solution:
        cell_path, direction_path = solve_maze(maze, start, end)
        meta["path_cells"] = [list(cell) for cell in cell_path]
        meta["path_directions"] = direction_path
        completion += "\n\nPath: " + ", ".join(direction_path)

    return {
        # gemma4_train.py's data prep (unsloth's standardize_data_formats)
        # requires this field to be named "conversations", not "messages".
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "meta": meta,
    }
