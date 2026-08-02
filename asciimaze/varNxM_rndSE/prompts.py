from ..maze.generator import Cell
from ..maze.render import column_name


def _format_label_list(labels: list[str]) -> str:
    quoted = [f"`{label}`" for label in labels]
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        return f"{quoted[0]} and {quoted[1]}"
    return ", ".join(quoted[:-1]) + f", and {quoted[-1]}"


def build_prompt(rows: int, columns: int, start: Cell, end: Cell) -> str:
    column_labels = _format_label_list([column_name(c) for c in range(columns)])
    row_labels = _format_label_list([str(r + 1) for r in range(rows)])
    return (
        f"Generate a random, valid {rows}×{columns} ASCII maze with barriers "
        f"between cells. Label the columns {column_labels}, and label the rows "
        f"{row_labels}. Place `S` in the starting cell and `E` in "
        "the ending cell. The entire outer boundary of the maze must be "
        "fully enclosed with barriers on all four sides: top, bottom, left, "
        "and right. There must always be at least one valid path from the "
        "starting cell to the ending cell. Return only the maze inside a "
        "monospaced code block."
    )


def cell_label(cell: Cell) -> str:
    row, column = cell
    return f"{column_name(column)}{row + 1}"
