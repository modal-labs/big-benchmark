import typer

from .run_benchmark_suite import run_benchmark_suite_cli

app = typer.Typer()
app.command()(run_benchmark_suite_cli)

if __name__ == "__main__":
    app()
