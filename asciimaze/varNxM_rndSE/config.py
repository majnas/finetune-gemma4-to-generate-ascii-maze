from dataclasses import dataclass


@dataclass(frozen=True)
class MazeConfig:
    """Defines what varies between generated maze samples.

    Same rectangular-maze setup as asciimaze.varNxM (rows != columns, drawn
    independently), except random_endpoints=True - S and E are placed at
    two random distinct cells instead of fixed opposite corners, so the
    model has to learn to actually locate/respect wherever S/E land rather
    than assuming top-left/bottom-right."""

    train_sizes: tuple[int, ...]
    val_sizes: tuple[int, ...]
    test_sizes: tuple[int, ...]
    random_endpoints: bool
    include_solution: bool


MAZE_CONFIG = MazeConfig(
    train_sizes=(3, 5, 7, 9),
    val_sizes=(3, 5, 7, 9),
    test_sizes=(4, 6, 8),
    random_endpoints=True,
    include_solution=False,
)
