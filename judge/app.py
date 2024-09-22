from pathlib import Path
from __init__ import judge_it

from rich.table import Table
from rich.console import Console
import typer
import shutil
def run_path(path: Path):
    table = Table(title="Transpilation Results")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta")

    dumpdir = Path("dump")
    dumpdir.mkdir(parents=True, exist_ok=True)

    lim = 100000
    idx = 0
    for file in path.glob("*.py"):
        if idx >= lim:
            break
        idx += 1
        score = judge_it(file, num_tests=2)
        if score == 1.0:
            shutil.copy(file, dumpdir / file.name)
            
            
        table.add_row(file.name, f"{score * 100:.2f}%")

    console = Console()
    console.print(table)


if __name__ == "__main__":
    typer.run(run_path)

