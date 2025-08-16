from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from . import __version__
from .config import get_settings
from .research import run_research

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def version() -> None:
	"""Show version."""
	console.print(f"Research Scholar Agent v{__version__}")


@app.command()
def research(
	query: str = typer.Argument(..., help="Research question or topic"),
	top_k: int = typer.Option(None, "--top-k", min=1, help="Number of sources to synthesize"),
	provider: str = typer.Option(None, "--provider", help="LLM provider: openai|together|ollama"),
	out: Path = typer.Option(None, "--out", help="Path to write the report markdown"),
) -> None:
	"""Run a research synthesis and optionally write to a file."""
	with Progress() as progress:
		task = progress.add_task("Searching & synthesizing...", total=None)
		resp = asyncio.run(run_research(query=query, top_k=top_k, provider_choice=provider))
		progress.update(task, completed=1)
	console.rule("Report")
	console.print(resp.report_markdown)
	console.rule("Citations")
	for c in resp.citations:
		authors = ", ".join(c.authors) if c.authors else "Unknown"
		console.print(f"[{c.index}] {c.title} — {authors} ({c.year or 'n.d.'}) | {c.doi or 'n/a'} | {c.url}")
	if out:
		out.write_text(resp.report_markdown, encoding="utf-8")
		console.print(f"\nSaved report to {out}")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
	"""Start the API server."""
	import uvicorn

	uvicorn.run("research_scholar_agent.api:app", host=host, port=port, reload=reload, factory=False)


@app.command()
def ui(port: int = 8501) -> None:
	"""Launch the Streamlit UI."""
	import subprocess

	cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("streamlit_app.py")), "--server.port", str(port)]
	env = os.environ.copy()
	subprocess.run(cmd, env=env, check=False)


if __name__ == "__main__":
	app()