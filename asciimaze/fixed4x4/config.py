from dataclasses import dataclass


@dataclass(frozen=True)
class MazeConfig:
    """Defines what varies between generated maze samples."""

    sizes: tuple[int, ...]
    random_endpoints: bool
    include_solution: bool


MAZE_CONFIG = MazeConfig(
    sizes=(4,),
    random_endpoints=False,
    include_solution=False,
)
