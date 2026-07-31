from ..maze.generator import Cell
from ..maze.render import column_name

MAZE_PROMPT = (
    "Generate a random, valid 4×4 ASCII maze with barriers between "
    "cells. Label the columns `A`, `B`, `C`, and `D`, and label the rows "
    "`1`, `2`, `3`, and `4`. Place `S` in the starting cell and `E` in "
    "the ending cell. The entire outer boundary of the maze must be "
    "fully enclosed with barriers on all four sides: top, bottom, left, "
    "and right. There must always be at least one valid path from the "
    "starting cell to the ending cell. Return only the maze inside a "
    "monospaced code block."
)


def build_prompt(rows: int, columns: int, start: Cell, end: Cell) -> str:
    return MAZE_PROMPT


def cell_label(cell: Cell) -> str:
    row, column = cell
    return f"{column_name(column)}{row + 1}"
