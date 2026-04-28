"""Typer CLI entrypoint for swap-cli."""

from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import config, license
from .runtime import DEFAULT_PROMPT, RunOptions, run_session
from .version import __version__

app = typer.Typer(
    name="swap",
    help="Realtime deepfake on your desktop. Bring your own Decart API key.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True)

# Voice cloning subcommand group (Sprint 13). 13a ships the install hook +
# stubs for list/add/remove; 13b fills in the rest.
voices_app = typer.Typer(
    name="voices",
    help="Manage voice cloning library and dependencies (optional feature).",
    no_args_is_help=True,
)
app.add_typer(voices_app, name="voices")


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"swap-cli [bold]{__version__}[/bold]")


@app.command()
def gui() -> None:
    """Launch the desktop GUI (recommended for non-developers)."""
    try:
        from .gui import launch
    except ImportError as err:
        err_console.print(
            f"[red]GUI dependencies not installed: {err}[/red]\n"
            "Install with [bold]pip install 'swap-cli[gui]'[/bold] or "
            "[bold]pip install customtkinter[/bold]."
        )
        raise typer.Exit(1) from err
    launch()


@app.command()
def setup(
    license_key: Annotated[
        str | None,
        typer.Option(
            "--license",
            "-l",
            help="Your swap-cli license key (SWAP-CLI-…). Prompted if omitted.",
        ),
    ] = None,
    decart_api_key: Annotated[
        str | None,
        typer.Option(
            "--decart-key",
            "-d",
            help="Your Decart API key (dct_…). Prompted if omitted.",
        ),
    ] = None,
) -> None:
    """Save your license key and Decart API key to the user config dir."""
    if not license_key:
        license_key = typer.prompt("License key (SWAP-CLI-…)")
    if not decart_api_key:
        decart_api_key = typer.prompt("Decart API key (dct_…)", hide_input=True)

    cfg = config.update(
        license_key=license_key.strip(),
        decart_api_key=decart_api_key.strip(),
        # Reset the cached validation so the next launch will re-validate.
        license_cached_at=None,
        license_cached_valid_until=None,
    )
    path = config.config_path()
    console.print(
        Panel.fit(
            f"Saved to [bold]{path}[/bold]\n"
            f"License: {_redact(cfg.license_key)}\n"
            f"Decart key: {_redact(cfg.decart_api_key)}\n\n"
            "Run [bold cyan]swap doctor[/bold cyan] to verify everything works.",
            title="✓ Setup complete",
            border_style="green",
        )
    )


@app.command(name="config")
def show_config() -> None:
    """Show current config (keys redacted)."""
    cfg = config.load()
    table = Table(title="swap-cli config", show_header=False, box=None)
    table.add_column("key", style="dim")
    table.add_column("value")
    table.add_row("config path", str(config.config_path()))
    table.add_row("license key", _redact(cfg.license_key) or "[red]not set[/red]")
    table.add_row("decart api key", _redact(cfg.decart_api_key) or "[red]not set[/red]")
    table.add_row("machine id", config.machine_id())
    if cfg.license_cached_valid_until is not None:
        table.add_row(
            "license valid until",
            _format_unix(cfg.license_cached_valid_until),
        )
    console.print(table)


@app.command()
def doctor() -> None:
    """Verify camera, network, license, and Decart auth."""
    asyncio.run(_doctor())


@app.command()
def run(
    reference: Annotated[
        str | None,
        typer.Option(
            "--reference",
            "-r",
            help="Path or URL of the reference identity image.",
        ),
    ] = None,
    prompt: Annotated[
        str,
        typer.Option(
            "--prompt",
            "-p",
            help="Transformation prompt. Defaults to a deepfake template.",
        ),
    ] = DEFAULT_PROMPT,
    model_name: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Decart realtime model identifier.",
        ),
    ] = "lucy-2",
    device: Annotated[
        int,
        typer.Option(
            "--device",
            help="Camera device index (0 = default webcam).",
        ),
    ] = 0,
    record: Annotated[
        Path | None,
        typer.Option(
            "--record",
            help="Save the output stream to MP4 at the given path.",
        ),
    ] = None,
    skip_license: Annotated[
        bool,
        typer.Option(
            "--skip-license",
            help="[dev] skip license validation. Will be removed in 1.0.",
            hidden=True,
        ),
    ] = False,
) -> None:
    """Open a realtime Decart session and stream until you press Q."""
    cfg = config.load()
    if not cfg.is_complete:
        err_console.print(
            "[red]Run [bold]swap setup[/bold] first to save your license + Decart key.[/red]"
        )
        raise typer.Exit(2)
    assert cfg.decart_api_key  # narrowed by is_complete

    if not skip_license:
        try:
            status = asyncio.run(license.validate())
        except license.LicenseError as err:
            err_console.print(f"[red]license: {err}[/red]")
            raise typer.Exit(3) from err
        if not status.valid:
            err_console.print(
                f"[red]License invalid ({status.reason}). "
                "Buy or renew at https://swap.ikieguy.online[/red]"
            )
            raise typer.Exit(3)
        if status.cached:
            console.print(f"[dim]license: cached ({status.reason})[/dim]")

    opts = RunOptions(
        decart_api_key=cfg.decart_api_key,
        reference=reference,
        prompt=prompt,
        model_name=model_name,
        camera_device=device,
        record=record,
    )

    console.print(
        Panel.fit(
            f"model: [bold]{opts.model_name}[/bold]\n"
            f"reference: {opts.reference or '[dim]none[/dim]'}\n"
            f"camera device: {opts.camera_device}\n"
            f"record: {opts.record or '[dim]off[/dim]'}\n\n"
            "[dim]Press [bold]Q[/bold] in the preview window to quit.[/dim]",
            title="▶ swap · live",
            border_style="cyan",
        )
    )

    try:
        asyncio.run(run_session(opts))
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")
    except Exception as err:  # noqa: BLE001
        err_console.print(f"[red]session failed: {err}[/red]")
        raise typer.Exit(1) from err


# ── voices ─────────────────────────────────────────────────────────────────
# Sprint 13a: install hook is real; list/add/remove are stubs that print
# the planned behavior. 13b implements the rest.


@voices_app.command("install")
def voices_install() -> None:
    """Install the optional voice extra and download OpenVoice weights.

    Runs `pip install 'swap-cli[voice]'` against the current interpreter to
    pull torch + sounddevice + librosa (~3 GB). Weight download lands in 13b.
    """
    from . import voice_prereq

    pre = voice_prereq.check_all()
    if not pre.gpu.ok:
        err_console.print(
            f"[red]✗ {pre.gpu.label}.[/red] {pre.gpu.hint or ''}\n"
            "[red]Voice features require a supported GPU.[/red]"
        )
        raise typer.Exit(2)

    if pre.deps_installed.ok:
        console.print(f"[green]✓ voice deps already installed.[/green]")
    else:
        console.print(
            "Installing voice deps via "
            f"[bold]{sys.executable} -m pip install 'swap-cli[voice]'[/bold] …"
        )
        cmd = [sys.executable, "-m", "pip", "install", "swap-cli[voice]"]
        # Editable installs (`pip install -e .`) need the local path syntax.
        if (Path.cwd() / "pyproject.toml").exists():
            cmd = [sys.executable, "-m", "pip", "install", "-e", ".[voice]"]
        try:
            import subprocess

            subprocess.check_call(cmd)
            console.print("[green]✓ voice deps installed[/green]")
        except subprocess.CalledProcessError as err:
            err_console.print(f"[red]pip install failed: {err}[/red]")
            raise typer.Exit(1) from err

    console.print(
        "\n[yellow]Note:[/yellow] OpenVoice weight download (~5 GB) lands in "
        "Sprint 13b. Voice cloning is not yet usable."
    )


@voices_app.command("uninstall")
def voices_uninstall() -> None:
    """Remove OpenVoice weights from disk (frees ~5 GB)."""
    from . import voice_prereq

    weights = voice_prereq.openvoice_weights_dir()
    if not weights.exists():
        console.print("[dim]No weights present — nothing to remove.[/dim]")
        return
    console.print(
        f"[yellow]Sprint 13b will remove[/yellow] {weights} "
        f"(~5 GB once weights actually ship)."
    )


@voices_app.command("list")
def voices_list() -> None:
    """List the bundled library + any user-added voices."""
    console.print(
        "[yellow]Coming in Sprint 13b.[/yellow] Will print the bundled "
        "library (Aria, Ben, Sage, …) and any voices you added via "
        "[bold]swap voices add[/bold]."
    )


@voices_app.command("add")
def voices_add(
    path: Annotated[Path, typer.Argument(help="Path to a WAV/MP3 reference.")],
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Display name. Defaults to file stem."),
    ] = None,
) -> None:
    """Add a custom reference voice from a WAV/MP3."""
    if not path.exists():
        err_console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)
    voice_name = name or path.stem
    console.print(
        f"[yellow]Coming in Sprint 13b.[/yellow] Will extract an OpenVoice "
        f"embedding from [bold]{path}[/bold] and save as "
        f"[bold]{voice_name}[/bold]."
    )


@voices_app.command("remove")
def voices_remove(
    name: Annotated[str, typer.Argument(help="Voice name to remove.")],
) -> None:
    """Remove a custom voice from your library."""
    console.print(
        f"[yellow]Coming in Sprint 13b.[/yellow] Will delete custom voice "
        f"[bold]{name}[/bold] from ~/.swap/voices/."
    )


# ── Helpers ────────────────────────────────────────────────────────────────


def _redact(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:4]}…{value[-4:]}"


def _format_unix(ts: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


async def _doctor() -> None:
    cfg = config.load()
    table = Table(title="swap-cli doctor", show_header=False, box=None)
    table.add_column("check", style="dim")
    table.add_column("status")

    # Config
    if cfg.license_key:
        table.add_row("license key set", "[green]✓[/green]")
    else:
        table.add_row("license key set", "[red]✗ run `swap setup`[/red]")
    if cfg.decart_api_key:
        table.add_row("decart api key set", "[green]✓[/green]")
    else:
        table.add_row("decart api key set", "[red]✗ run `swap setup`[/red]")

    # Network — just DNS for the two endpoints we'll hit
    for host in ("swap.ikieguy.online", "api.decart.ai"):
        try:
            await asyncio.to_thread(socket.gethostbyname, host)
            table.add_row(f"dns {host}", "[green]✓[/green]")
        except OSError as err:
            table.add_row(f"dns {host}", f"[red]✗ {err}[/red]")

    # License validation
    if cfg.license_key:
        try:
            status = await license.validate(force_online=True)
            if status.valid:
                table.add_row("license validate", f"[green]✓ {status.reason}[/green]")
            else:
                table.add_row("license validate", f"[red]✗ {status.reason}[/red]")
        except license.LicenseError as err:
            table.add_row("license validate", f"[yellow]offline ({err})[/yellow]")

    # Camera (cheap probe)
    table.add_row("camera probe", _camera_probe_label())

    # Native deps
    table.add_row("aiortc import", _import_ok("aiortc"))
    table.add_row("decart import", _import_ok("decart"))
    table.add_row("opencv import", _import_ok("cv2"))
    table.add_row("av import", _import_ok("av"))

    console.print(table)

    failures = sum(
        1
        for row in table.rows
        if "✗" in str(row)  # naive — we just inspect the rendering
    )
    if failures:
        sys.exit(1)


def _camera_probe_label() -> str:
    try:
        import cv2

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            return "[red]✗ no camera at index 0[/red]"
        ok, _ = cap.read()
        cap.release()
        return "[green]✓[/green]" if ok else "[red]✗ read failed[/red]"
    except Exception as err:  # noqa: BLE001
        return f"[red]✗ {err}[/red]"


def _import_ok(name: str) -> str:
    try:
        __import__(name)
        return "[green]✓[/green]"
    except ImportError as err:
        return f"[red]✗ {err}[/red]"
