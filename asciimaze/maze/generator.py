import random

Cell = tuple[int, int]

# Movement: row change, column change, direction, opposite direction
DIRECTIONS = [
    (-1, 0, "N", "S"),
    (0, 1, "E", "W"),
    (1, 0, "S", "N"),
    (0, -1, "W", "E"),
]


def generate_maze(
    rows: int,
    columns: int,
    rng: random.Random,
) -> list[list[set[str]]]:
    """
    Generate a random perfect maze using recursive backtracking.

    Each cell stores the directions in which movement is allowed.
    Because this produces a perfect maze, every cell is reachable and
    the path between any two cells is unique.
    """
    passages: list[list[set[str]]] = [
        [set() for _ in range(columns)]
        for _ in range(rows)
    ]

    visited = [
        [False for _ in range(columns)]
        for _ in range(rows)
    ]

    initial_cell = (
        rng.randrange(rows),
        rng.randrange(columns),
    )

    stack = [initial_cell]
    visited[initial_cell[0]][initial_cell[1]] = True

    while stack:
        row, column = stack[-1]
        available_neighbours = []

        for dr, dc, direction, opposite in DIRECTIONS:
            next_row = row + dr
            next_column = column + dc

            if (
                0 <= next_row < rows
                and 0 <= next_column < columns
                and not visited[next_row][next_column]
            ):
                available_neighbours.append(
                    (
                        next_row,
                        next_column,
                        direction,
                        opposite,
                    )
                )

        if not available_neighbours:
            stack.pop()
            continue

        next_row, next_column, direction, opposite = rng.choice(
            available_neighbours
        )

        # Remove the wall between the current cell and the next cell.
        passages[row][column].add(direction)
        passages[next_row][next_column].add(opposite)

        visited[next_row][next_column] = True
        stack.append((next_row, next_column))

    return passages
