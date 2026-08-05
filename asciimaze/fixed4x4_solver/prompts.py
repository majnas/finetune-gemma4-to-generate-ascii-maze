SOLVER_INSTRUCTION = (
    "Solve the following ASCII maze from `S` to `E`. Return only the path as "
    "a comma-separated list. Write `S` first and `E` last, and write the "
    "column-and-row label of every intermediate cell (for example, "
    "`S,A2,B2,B3,E`). Do not include an explanation or a code block."
)


def build_prompt(maze_text: str) -> str:
    return f"{SOLVER_INSTRUCTION}\n\n```\n{maze_text}\n```"
