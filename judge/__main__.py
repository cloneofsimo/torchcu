from judge import judge_it, DATA_DIR

from rich.table import Table
from rich.console import Console

table = Table(title="Transpilation Results")
table.add_column("File", style="cyan", no_wrap=True)
table.add_column("Score", style="magenta")


for file in DATA_DIR.glob("*.py"):
        score = judge_it(file)
        table.add_row(file.name, f"{score * 100:.2f}%")

console = Console()
console.print(table)
