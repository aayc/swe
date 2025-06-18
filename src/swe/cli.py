"""Main CLI interface for the SWE agent."""

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import typer
from rich.console import Console

from swe.agent import SWEAgent
from swe.llm import ChatModel

app = typer.Typer(help="SWE - Software Engineering Agent", no_args_is_help=True, add_completion=False)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: str | None = typer.Argument(None, help="What you want the SWE agent to do"),
    model: str = typer.Option("gemini-2.5-flash", "--model", "-m", help="Model to use"),
    directory: str = typer.Option(".", "--directory", "-d", help="Working directory"),
    system_prompt: str | None = typer.Option(None, "--system", "-s", help="System prompt"),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Maximum tokens"),
    temperature: float | None = typer.Option(None, "--temperature", help="Temperature"),
):
    """SWE - Software Engineering Agent

    A helpful agent that can read, edit, search files and help with coding tasks.

    Examples:
        swe "read main.py and explain what it does"
        swe "find all TODO comments in the codebase"
        swe "fix the bug in calculator.py where multiply returns wrong result"
        swe "add error handling to the file_processor.py"
        swe "write tests for the Calculator class"
    """

    if ctx.invoked_subcommand is not None:
        return

    if prompt is None:
        console.print("[yellow]Please provide a prompt. Use --help for usage information.")
        raise typer.Exit(1)

    # Validate model
    try:
        chat_model = ChatModel(model)
    except ValueError:
        console.print(f"[red]Error: Invalid model '{model}'. Available models:")
        for m in ChatModel:
            console.print(f"  - {m.value}")
        sys.exit(1)

    # Validate directory
    work_dir = Path(directory).resolve()
    if not work_dir.exists():
        console.print(f"[red]Error: Directory '{directory}' does not exist.")
        sys.exit(1)

    if not work_dir.is_dir():
        console.print(f"[red]Error: '{directory}' is not a directory.")
        sys.exit(1)

    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        console.print("[red]Error: GEMINI_API_KEY environment variable is required.")
        console.print("Please set your Google AI API key:")
        console.print("  export GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)

    # Create agent
    agent = SWEAgent(
        model=chat_model,
        working_directory=work_dir,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Process the prompt
    asyncio.run(_process_prompt(agent, prompt))


async def _process_prompt(agent: SWEAgent, prompt: str):
    """Process a single prompt with the agent with streaming output."""
    try:

        def stream_callback(text: str):
            """Callback to stream agent progress in real-time."""
            # Print rich markup text immediately without buffering
            console.print(text, end="", markup=True)

        console.print(f"[bold green]\nSWE Agent is working on: {prompt}[/bold green]\n")
        console.print("[dim]" + "=" * 80 + "[/dim]\n", markup=True)

        # Process with streaming
        await agent.process_message(prompt, [], stream_callback)

        console.print("\n[dim]" + "=" * 80 + "[/dim]", markup=True)
        console.print("[bold green]\n✅ Task completed![/bold green]", markup=True)

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback

        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
