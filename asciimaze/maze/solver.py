from collections import deque

from .generator import DIRECTIONS, Cell


def solve_maze(
    maze: list[list[set[str]]],
    start: Cell,
    end: Cell,
) -> tuple[list[Cell], list[str]]:
    """
    Find the path from start to end.

    The generator produces a perfect maze (a spanning tree over the
    cells), so exactly one simple path connects any two cells -- a
    plain BFS parent-walk reconstructs it.

    Returns the cell path (including start and end) and the matching
    list of compass directions taken between consecutive cells.
    """
    parents: dict[Cell, tuple[Cell, str]] = {}
    queue = deque([start])
    visited = {start}

    while queue:
        current = queue.popleft()

        if current == end:
            break

        row, column = current

        for dr, dc, direction, _opposite in DIRECTIONS:
            if direction not in maze[row][column]:
                continue

            neighbour = (row + dr, column + dc)

            if neighbour in visited:
                continue

            visited.add(neighbour)
            parents[neighbour] = (current, direction)
            queue.append(neighbour)

    cell_path = [end]
    direction_path: list[str] = []
    cursor = end

    while cursor != start:
        cursor, direction = parents[cursor]
        cell_path.append(cursor)
        direction_path.append(direction)

    cell_path.reverse()
    direction_path.reverse()

    return cell_path, direction_path
