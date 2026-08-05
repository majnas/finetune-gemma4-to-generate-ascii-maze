import random

from .config import MAZE_CONFIG
from .prompts import build_prompt
from ..maze.endpoints import choose_endpoints
from ..maze.generator import Cell, generate_maze
from ..maze.render import column_name, render_maze
from ..maze.solver import solve_maze


def format_path(cell_path: list[Cell]) -> str:
    """Format a path with S/E markers and labelled intermediate cells."""
    labels = ["S"]
    labels.extend(
        f"{column_name(column)}{row + 1}"
        for row, column in cell_path[1:-1]
    )
    labels.append("E")
    return ",".join(labels)


def build_sample(seed: int) -> dict:
    rng = random.Random(seed)
    rows = MAZE_CONFIG.rows
    columns = MAZE_CONFIG.columns
    maze = generate_maze(rows, columns, rng)
    start, end = choose_endpoints(
        rows, columns, rng, MAZE_CONFIG.random_endpoints
    )
    cell_path, direction_path = solve_maze(maze, start, end)
    maze_text = render_maze(maze, start, end)

    return {
        "conversations": [
            {"role": "user", "content": build_prompt(maze_text)},
            {"role": "assistant", "content": format_path(cell_path)},
        ],
        "meta": {
            "seed": seed,
            "rows": rows,
            "columns": columns,
            "start": list(start),
            "end": list(end),
            "path_cells": [list(cell) for cell in cell_path],
            "path_directions": direction_path,
        },
    }
