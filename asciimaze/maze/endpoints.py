import random

from .generator import Cell


def choose_endpoints(
    rows: int,
    columns: int,
    rng: random.Random,
    random_endpoints: bool,
) -> tuple[Cell, Cell]:
    """Choose distinct start and end cells."""
    if not random_endpoints:
        return (0, 0), (rows - 1, columns - 1)

    all_cells = [
        (row, column)
        for row in range(rows)
        for column in range(columns)
    ]

    start, end = rng.sample(all_cells, 2)
    return start, end
