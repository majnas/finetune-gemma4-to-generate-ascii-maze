from dataclasses import dataclass


@dataclass(frozen=True)
class MazeConfig:
    """Defines what varies between generated maze samples.

    Unlike fixed4x4's single `sizes` tuple, train/val and test draw from
    disjoint size sets - test sizes are never seen during training, so
    evaluating on them measures generalization across maze size."""

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
