from dataclasses import dataclass


@dataclass(frozen=True)
class MazeConfig:
    """Defines what varies between generated maze samples.

    Unlike varNxN's single size tuple (rows == columns), varNxM draws rows
    and columns independently and excludes rows == columns (that square
    case is varNxN's job) - train/val and test draw from disjoint size
    pools, same generalization-to-unseen-size idea as varNxN, now across
    two independent dimensions."""

    train_sizes: tuple[int, ...]
    val_sizes: tuple[int, ...]
    test_sizes: tuple[int, ...]
    random_endpoints: bool
    include_solution: bool


MAZE_CONFIG = MazeConfig(
    train_sizes=(3, 5, 7, 9),
    val_sizes=(3, 5, 7, 9),
    test_sizes=(4, 6, 8),
    random_endpoints=False,
    include_solution=False,
)
