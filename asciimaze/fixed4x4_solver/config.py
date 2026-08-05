from dataclasses import dataclass


@dataclass(frozen=True)
class MazeConfig:
    rows: int
    columns: int
    random_endpoints: bool


MAZE_CONFIG = MazeConfig(rows=4, columns=4, random_endpoints=False)
