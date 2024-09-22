from pathlib import Path
from __init__ import judge_it

from rich.table import Table
from rich.console import Console
import typer

def run_path(path: Path):
    table = Table(title="Transpilation Results")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta")

    lim = 40
    idx = 0
    for file in path.glob("*.py"):
        if idx >= lim:
            break
        idx += 1
        score = judge_it(file)
        table.add_row(file.name, f"{score * 100:.2f}%")

    console = Console()
    console.print(table)


if __name__ == "__main__":
    typer.run(run_path)

