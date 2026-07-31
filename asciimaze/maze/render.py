from .generator import Cell


def column_name(index: int) -> str:
    """Convert a zero-based column index to A, B, ..., Z, AA, AB, ..."""
    name = ""

    while True:
        index, remainder = divmod(index, 26)
        name = chr(ord("A") + remainder) + name

        if index == 0:
            return name

        index -= 1


def render_maze(
    maze: list[list[set[str]]],
    start: Cell,
    end: Cell,
) -> str:
    """Render the maze with column and row labels."""
    rows = len(maze)
    columns = len(maze[0])

    cell_width = 3
    row_label_width = len(str(rows))

    column_labels = " " * (row_label_width + 2)

    column_labels += " ".join(
        column_name(column).center(cell_width)
        for column in range(columns)
    )

    lines = [column_labels]

    # Top outside wall
    lines.append(
        " " * (row_label_width + 1)
        + "+"
        + "+".join("---" for _ in range(columns))
        + "+"
    )

    for row in range(rows):
        cell_line = f"{row + 1:>{row_label_width}} "

        for column in range(columns):
            # Draw the left wall unless movement west is allowed.
            if column == 0 or "W" not in maze[row][column]:
                cell_line += "|"
            else:
                cell_line += " "

            if (row, column) == start:
                content = "S"
            elif (row, column) == end:
                content = "E"
            else:
                content = " "

            cell_line += content.center(cell_width)

        # Right outside wall
        cell_line += "|"
        lines.append(cell_line)

        # Draw the bottom walls
        bottom_line = " " * (row_label_width + 1) + "+"

        for column in range(columns):
            if "S" in maze[row][column]:
                bottom_line += "   +"
            else:
                bottom_line += "---+"

        lines.append(bottom_line)

    return "\n".join(lines)
