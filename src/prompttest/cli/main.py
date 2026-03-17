"""CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from prompttest.core.loader import PROMPTTEST_DIR

app = typer.Typer(
    name="prompttest",
    help="A testing framework for LLM prompts.",
    no_args_is_help=True,
)
console = Console()

EXAMPLE_PROMPT = """\
name: summarize
version: "1"
provider: echo
model: gpt-4o-mini
system: You are a concise summarizer.
template: "Summarize the following text:\\n\\n{{input}}"
parameters:
  temperature: 0.3
"""

EXAMPLE_PROMPT_V2 = """\
name: summarize
version: "2"
provider: echo
model: gpt-4o-mini
system: You are a concise summarizer. Be brief and precise.
template: "Provide a one-sentence summary of:\\n\\n{{input}}"
parameters:
  temperature: 0.2
"""

EXAMPLE_DATASET = """\
name: summarize-basics
prompt: summarize
cases:
  - input: "The quick brown fox jumps over the lazy dog. The dog did not react."
    expected: "fox"
    tags: [smoke]
  - input: "Python is a popular programming language created by Guido van Rossum."
    expected: "Python"
    tags: [smoke]
"""


@app.command()
def init(
    directory: Path = typer.Argument(
        Path("."),
        help="Directory to initialize the project in.",
    ),
) -> None:
    """Initialize a new prompttest project with example files."""
    root = directory.resolve() / PROMPTTEST_DIR
    prompts_dir = root / "prompts"
    datasets_dir = root / "datasets"
    results_dir = root / "results"

    for d in [prompts_dir, datasets_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    prompt_file = prompts_dir / "summarize.yaml"
    if not prompt_file.exists():
        prompt_file.write_text(EXAMPLE_PROMPT)

    prompt_v2_file = prompts_dir / "summarize_v2.yaml"
    if not prompt_v2_file.exists():
        prompt_v2_file.write_text(EXAMPLE_PROMPT_V2)

    dataset_file = datasets_dir / "summarize-basics.yaml"
    if not dataset_file.exists():
        dataset_file.write_text(EXAMPLE_DATASET)

    console.print(f"[green]Initialized prompttest project in {root}[/green]")
    console.print(f"  prompts/   → {prompts_dir}")
    console.print(f"  datasets/  → {datasets_dir}")
    console.print(f"  results/   → {results_dir}")
    console.print("\nRun [bold]prompttest run[/bold] to execute tests.")


@app.command()
def run(
    directory: Path = typer.Option(
        Path("."),
        "--dir",
        "-d",
        help="Project root containing .prompttest/",
    ),
) -> None:
    """Run all datasets against their linked prompts."""
    from prompttest.core.runner import run_all

    root = directory.resolve() / PROMPTTEST_DIR
    if not root.exists():
        console.print("[red]No .prompttest/ directory found. Run 'prompttest init' first.[/red]")
        raise typer.Exit(1)

    console.print("[bold]Running prompt tests...[/bold]\n")
    try:
        run_results = run_all(root)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)

    any_failure = False
    for rr in run_results:
        table = Table(
            title=f"{rr.prompt_name} v{rr.prompt_version} — {rr.dataset_name}",
            show_lines=True,
        )
        table.add_column("Input", style="cyan", max_width=40)
        table.add_column("Expected", style="yellow", max_width=30)
        table.add_column("Output", style="white", max_width=40)
        table.add_column("Verdict", justify="center")
        table.add_column("Reason", style="dim", max_width=30)

        for cr in rr.results:
            verdict_style = {
                "pass": "[green]PASS[/green]",
                "fail": "[red]FAIL[/red]",
                "error": "[red bold]ERROR[/red bold]",
            }[cr.verdict.value]

            table.add_row(
                cr.case.input[:80],
                cr.case.expected,
                cr.output[:80],
                verdict_style,
                cr.reason,
            )

        console.print(table)
        console.print(
            f"  Results: [green]{rr.passed} passed[/green], "
            f"[red]{rr.failed} failed[/red], "
            f"{rr.total} total — "
            f"pass rate: {rr.pass_rate:.0%}\n"
        )
        if rr.failed > 0:
            any_failure = True

    if any_failure:
        raise typer.Exit(1)


@app.command("list-prompts")
def list_prompts(
    directory: Path = typer.Option(
        Path("."),
        "--dir",
        "-d",
        help="Project root containing .prompttest/",
    ),
) -> None:
    """List all registered prompts and their versions."""
    from prompttest.core.registry import PromptRegistry

    root = directory.resolve() / PROMPTTEST_DIR
    prompts_dir = root / "prompts"
    if not prompts_dir.exists():
        console.print("[red]No prompts/ directory found. Run 'prompttest init' first.[/red]")
        raise typer.Exit(1)

    registry = PromptRegistry.from_directory(prompts_dir)
    if not registry.names:
        console.print("[yellow]No prompts found.[/yellow]")
        raise typer.Exit(0)

    table = Table(title="Registered Prompts")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Provider", style="yellow")
    table.add_column("Model", style="white")
    table.add_column("File", style="dim")

    for entry in registry.all_entries():
        table.add_row(
            entry.config.name,
            entry.config.version,
            entry.config.provider,
            entry.config.model,
            entry.path.name,
        )

    console.print(table)


@app.command("show-prompt")
def show_prompt(
    identifier: str = typer.Argument(
        help="Prompt identifier: 'name' (latest version) or 'name_vN'.",
    ),
    directory: Path = typer.Option(
        Path("."),
        "--dir",
        "-d",
        help="Project root containing .prompttest/",
    ),
) -> None:
    """Show details of a specific prompt version."""
    from prompttest.core.registry import PromptRegistry

    root = directory.resolve() / PROMPTTEST_DIR
    prompts_dir = root / "prompts"
    if not prompts_dir.exists():
        console.print("[red]No prompts/ directory found. Run 'prompttest init' first.[/red]")
        raise typer.Exit(1)

    registry = PromptRegistry.from_directory(prompts_dir)

    # Parse identifier: support_v1 → name=support, version=v1
    name, version = _parse_prompt_identifier(identifier)

    entry = registry.get(name, version)
    if entry is None:
        console.print(f"[red]Prompt '{identifier}' not found.[/red]")
        available = registry.names
        if available:
            console.print(f"Available prompts: {', '.join(available)}")
        raise typer.Exit(1)

    cfg = entry.config
    console.print(f"[bold cyan]{cfg.name}[/bold cyan] [green]v{cfg.version}[/green]")
    console.print(f"  Provider: {cfg.provider}")
    console.print(f"  Model:    {cfg.model}")
    console.print(f"  File:     {entry.path}")
    if cfg.parameters:
        console.print(f"  Params:   {cfg.parameters}")
    console.print()
    console.print("[bold]System prompt:[/bold]")
    console.print(f"  {cfg.system}" if cfg.system else "  [dim](empty)[/dim]")
    console.print()
    console.print("[bold]User template:[/bold]")
    console.print(f"  {cfg.template}")

    # Show available versions for this prompt
    versions = registry.versions(cfg.name)
    if len(versions) > 1:
        console.print()
        console.print(f"[dim]Other versions: {', '.join(v for v in versions if v != cfg.version)}[/dim]")


@app.command("diff-prompts")
def diff_prompts(
    name: str = typer.Argument(help="Prompt name."),
    version_a: str = typer.Argument(help="First version (e.g. v1)."),
    version_b: str = typer.Argument(help="Second version (e.g. v2)."),
    directory: Path = typer.Option(
        Path("."),
        "--dir",
        "-d",
        help="Project root containing .prompttest/",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Save diff to a file (JSON).",
    ),
) -> None:
    """Show a diff between two versions of a prompt."""
    from prompttest.core.registry import PromptRegistry

    root = directory.resolve() / PROMPTTEST_DIR
    prompts_dir = root / "prompts"
    if not prompts_dir.exists():
        console.print("[red]No prompts/ directory found. Run 'prompttest init' first.[/red]")
        raise typer.Exit(1)

    registry = PromptRegistry.from_directory(prompts_dir)
    try:
        result = registry.diff(name, version_a, version_b)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if not result:
        console.print("[green]No differences found.[/green]")
    else:
        console.print(result)

    if output is not None:
        from prompttest.core.exporter import export_diff_json

        content = export_diff_json(result, name, version_a, version_b)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)
        console.print(f"\n[green]Diff saved to {output.resolve()}[/green]")


@app.command("eval")
def eval_dataset(
    dataset_file: Path = typer.Argument(help="Path to an evaluation dataset YAML file."),
    directory: Path = typer.Option(
        Path("."),
        "--dir",
        "-d",
        help="Project root containing .prompttest/",
    ),
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Override model (e.g. gpt-4o, claude-3-haiku-20240307). Auto-detects provider.",
    ),
    provider: str = typer.Option(
        "",
        "--provider",
        "-p",
        help="Override provider (openai, anthropic, local, echo).",
    ),
    scorer: str = typer.Option(
        "",
        "--scorer",
        "-s",
        help="Override the scoring function (e.g. exact, contains, fuzzy, semantic, llm_judge).",
    ),
    pass_threshold: float = typer.Option(
        0.7,
        "--pass-threshold",
        "-t",
        help="Minimum score for a test case to pass (0.0-1.0).",
        min=0.0,
        max=1.0,
    ),
    use_async: bool = typer.Option(
        False,
        "--async",
        help="Run evaluation cases concurrently.",
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Strict validation: fail on missing placeholders (default: strict).",
    ),
    max_concurrency: int = typer.Option(
        10,
        "--max-concurrency",
        "-c",
        help="Maximum number of concurrent requests (async mode only).",
        min=1,
    ),
    rate_limit: float = typer.Option(
        0.0,
        "--rate-limit",
        "-r",
        help="Maximum requests per second (0 = unlimited).",
        min=0.0,
    ),
    max_retries: int = typer.Option(
        3,
        "--max-retries",
        help="Maximum retry attempts on transient/rate-limit errors.",
        min=0,
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Save results to a file (e.g. results.json or results.csv).",
    ),
    output_format: str = typer.Option(
        "",
        "--format",
        "-f",
        help="Output format: json or csv. Inferred from --output extension if omitted.",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        help="Directory for auto-named timestamped result files.",
    ),
    tags: str = typer.Option(
        "",
        "--tags",
        help="Comma-separated tags to filter test cases (e.g. billing,critical).",
    ),
    match: str = typer.Option(
        "any",
        "--match",
        help="Tag matching mode: 'any' (match at least one tag) or 'all' (match every tag).",
    ),
) -> None:
    """Run an evaluation dataset against its linked prompt."""
    from prompttest.core.eval_runner import load_eval_dataset, run_eval, run_eval_async
    from prompttest.core.registry import PromptRegistry
    from prompttest.core.scoring import list_scorers
    from prompttest.providers.registry import get_provider as get_prov, resolve_model

    dataset_path = dataset_file.resolve()
    if not dataset_path.exists():
        console.print(f"[red]Dataset file not found: {dataset_path}[/red]")
        raise typer.Exit(1)

    root = directory.resolve() / PROMPTTEST_DIR
    prompts_dir = root / "prompts"
    if not prompts_dir.exists():
        console.print("[red]No prompts/ directory found. Run 'prompttest init' first.[/red]")
        raise typer.Exit(1)

    # Load dataset to resolve the prompt reference
    ds = load_eval_dataset(dataset_path)

    # Override scorer if provided via CLI
    if scorer:
        ds.scoring = scorer

    # --- Filter by tags ---
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    if tag_list:
        if match not in ("any", "all"):
            console.print(f"[red]--match must be 'any' or 'all', got '{match}'[/red]")
            raise typer.Exit(1)
        from prompttest.core.eval_runner import filter_by_tags
        original, filtered = filter_by_tags(ds, tag_list, match)
        console.print(
            f"Running [bold]{filtered}/{original}[/bold] tests "
            f"(filtered by tags: [cyan]{', '.join(tag_list)}[/cyan]"
            f" | match: [white]{match}[/white])\n"
        )
        if filtered == 0:
            console.print("[yellow]No test cases match the given tags.[/yellow]")
            raise typer.Exit(0)

    # Resolve prompt from registry
    registry = PromptRegistry.from_directory(prompts_dir)
    name, version = _parse_prompt_identifier(ds.prompt)
    entry = registry.get(name, version)
    if entry is None:
        console.print(f"[red]Prompt '{ds.prompt}' not found in registry.[/red]")
        available = registry.names
        if available:
            console.print(f"Available: {', '.join(available)}")
        raise typer.Exit(1)

    # Apply model/provider overrides to a copy of the prompt config
    from dataclasses import replace
    cfg = entry.config

    provider_override = None
    if model:
        # Auto-detect provider from model name unless --provider is explicit
        if not provider:
            try:
                detected_provider, model_id = resolve_model(model)
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(1)
            cfg = replace(cfg, model=model_id, provider=detected_provider)
        else:
            cfg = replace(cfg, model=model, provider=provider)
    elif provider:
        cfg = replace(cfg, provider=provider)

    if provider or model:
        provider_override = get_prov(cfg.provider)

    # --- Validate prompt template against dataset before running ---
    from prompttest.validation.prompt_validator import ValidationError, validate_dataset

    validation = validate_dataset(cfg, ds)

    if validation.warnings:
        for w in validation.warnings:
            console.print(f"[yellow]Warning: {w.message}[/yellow]")
        console.print()

    if validation.errors:
        console.print("[bold red]Validation Error[/bold red]\n")
        for err in validation.errors:
            console.print(f'[red]Missing placeholder: {", ".join(f"{f!r}" for f in err.missing)}[/red]')
            console.print(f"[red]In test case #{err.case_index}[/red]\n")
        if strict:
            raise typer.Exit(1)
        else:
            console.print("[yellow]Continuing with --no-strict mode...[/yellow]\n")

    eval_header = (
        f"[bold]Evaluating[/bold] prompt [cyan]{cfg.name}[/cyan] "
        f"[green]v{cfg.version}[/green] "
        f"| model [white]{cfg.model}[/white] "
        f"| provider [yellow]{cfg.provider}[/yellow] "
        f"| scorer [yellow]{ds.scoring}[/yellow]"
        f" | threshold [white]{pass_threshold}[/white]"
    )
    if use_async:
        eval_header += (
            f"\n  concurrency [white]{max_concurrency}[/white]"
            f" | rate limit [white]{'unlimited' if rate_limit <= 0 else f'{rate_limit}/s'}[/white]"
            f" | retries [white]{max_retries}[/white]"
        )
    console.print(eval_header + "\n")

    try:
        if use_async:
            import asyncio
            from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
            from prompttest.core.concurrency import ConcurrencyConfig

            cc = ConcurrencyConfig(
                max_concurrency=max_concurrency,
                rate_limit=rate_limit,
                max_retries=max_retries,
            )

            total_cases = len(ds.tests)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                console=console,
            ) as progress:
                task_id = progress.add_task("Evaluating", total=total_cases)

                def _advance() -> None:
                    progress.advance(task_id)

                result = asyncio.run(
                    run_eval_async(
                        dataset_path,
                        cfg,
                        provider_override,
                        strict=False,
                        pass_threshold=pass_threshold,
                        concurrency_config=cc,
                        on_case_complete=_advance,
                    )
                )
        else:
            result = run_eval(
                dataset_path, cfg, provider_override,
                strict=False, pass_threshold=pass_threshold,
            )
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(f"Available scorers: {', '.join(list_scorers())}")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1)

    _print_eval_result(result)

    # --- Save results if requested ---
    if output is not None or output_dir is not None:
        from prompttest.core.exporter import auto_filename, save_result

        fmt = output_format
        if not fmt and output is not None:
            fmt = output.suffix.lstrip(".")
        if not fmt:
            fmt = "json"
        if fmt not in ("json", "csv"):
            console.print(f"[red]Unknown format '{fmt}'. Use json or csv.[/red]")
            raise typer.Exit(1)

        if output is not None:
            dest = output
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            dest = output_dir / auto_filename(cfg, fmt)

        saved_path = save_result(result, cfg, dest, fmt)
        console.print(f"\n[green]Results saved to {saved_path}[/green]")

    if result.failed > 0 or result.errors > 0:
        raise typer.Exit(1)


def _print_eval_result(result: object) -> None:
    """Pretty-print evaluation results."""
    table = Table(
        title=f"Eval: {result.prompt_name} v{result.prompt_version}",
        show_lines=True,
    )
    table.add_column("#", style="dim", justify="right")
    table.add_column("Input", style="cyan", max_width=40)
    table.add_column("Expected", style="yellow", max_width=25)
    table.add_column("Output", style="white", max_width=40)
    table.add_column("Score", justify="right")
    table.add_column("Verdict", justify="center")
    table.add_column("Reason", style="dim", max_width=30)

    for i, cr in enumerate(result.case_results, 1):
        verdict_style = {
            "pass": "[green]PASS[/green]",
            "fail": "[red]FAIL[/red]",
            "error": "[red bold]ERROR[/red bold]",
        }[cr.verdict.value]

        score_str = f"{cr.score:.2f}" if cr.verdict.value != "error" else "-"

        table.add_row(
            str(i),
            cr.case.input_summary[:80],
            cr.case.expected,
            cr.output[:80],
            score_str,
            verdict_style,
            cr.reason,
        )

    console.print(table)
    console.print()
    console.print("[bold]Test Results[/bold]")
    console.print(f"  Total:    {result.total}")
    console.print(f"  Passed:   [green]{result.passed}[/green]")
    console.print(f"  Failed:   [red]{result.failed}[/red]")
    if result.errors:
        console.print(f"  Errors:   [red bold]{result.errors}[/red bold]")
    console.print(f"  Accuracy: {result.accuracy:.0%}")
    console.print()
    avg = result.average_score
    avg_color = "green" if avg >= result.pass_threshold else "red"
    console.print(f"  Average Score:  [{avg_color}]{avg:.2f}[/{avg_color}]")
    console.print(f"  Pass Threshold: {result.pass_threshold:.2f}")


def _parse_prompt_identifier(identifier: str) -> tuple[str, str | None]:
    """Parse 'name_vN' into (name, 'vN') or plain 'name' into (name, None).

    Examples:
        'support_v1'   → ('support', 'v1')
        'support_v2.1' → ('support', 'v2.1')
        'support'      → ('support', None)
        'my_bot_v3'    → ('my_bot', 'v3')
    """
    # Try to split on the last '_v' occurrence
    idx = identifier.rfind("_v")
    if idx > 0:
        name = identifier[:idx]
        version = identifier[idx + 1:]  # keeps the 'v' prefix
        return name, version
    return identifier, None


if __name__ == "__main__":
    app()
