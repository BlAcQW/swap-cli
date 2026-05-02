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

# Voice cloning subcommand group.
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
def gui(
    voice_only: Annotated[
        bool,
        typer.Option(
            "--voice",
            help="Open a stripped voice-only window (no face/camera/Decart).",
        ),
    ] = False,
) -> None:
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
    launch(voice_only=voice_only)


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


@app.command()
def voice(
    voice: Annotated[
        str,
        typer.Option(
            "--voice",
            "-v",
            help="Voice id or name (e.g. 'aria' or your custom voice).",
        ),
    ],
    mic: Annotated[
        int | None,
        typer.Option(
            "--mic",
            help="Microphone device index. Default: system default mic.",
        ),
    ] = None,
    output: Annotated[
        int | None,
        typer.Option(
            "--output",
            "-o",
            help="Output device index. Default: auto-detected virtual cable.",
        ),
    ] = None,
) -> None:
    """Run voice cloning standalone for live calls. Press Ctrl+C to stop.

    Voice runs entirely on your local GPU — no Decart connection, no
    tokens spent. Open Zoom/Meet/Discord with the virtual cable as your
    microphone (e.g. 'CABLE Output' / 'BlackHole 2ch') and speak — the
    other side hears the cloned voice.
    """
    _run_voice_session(voice_name=voice, mic=mic, output=output, seconds=0)


def _run_voice_session(
    *,
    voice_name: str,
    mic: int | None,
    output: int | None,
    seconds: int,
) -> None:
    """Shared body for `swap voice` (forever) and `swap voices test` (timed)."""
    import asyncio

    from . import voice_ops, voice_router
    from .voice_track import VoiceTrack, VoiceTrackOptions

    target = voice_ops.find_voice_by_name_or_id(voice_name)
    if target is None:
        err_console.print(
            f"[red]Voice '{voice_name}' not found.[/red] "
            "Run [bold]swap voices list[/bold] to see available voices."
        )
        raise typer.Exit(1)

    mic_idx, out_idx = voice_ops.resolve_voice_devices(mic, output)
    cable_hint = voice_router.virtual_cable_hint()

    if out_idx is None:
        err_console.print(
            f"[yellow]No virtual audio cable detected.[/yellow] Install "
            f"[bold]{cable_hint.name}[/bold] so apps like Zoom/Meet can hear "
            f"the cloned voice. Continuing — converted audio will be silent."
        )

    duration = "until Ctrl+C" if seconds <= 0 else f"{seconds}s"
    console.print(
        Panel.fit(
            f"voice: [bold]{target.name}[/bold] ({target.source})\n"
            f"mic device: {mic_idx}\n"
            f"output device: {out_idx if out_idx is not None else '[dim]none — silent[/dim]'}\n"
            f"duration: {duration}\n\n"
            "[dim]Local GPU only. No Decart. Zero tokens spent.[/dim]",
            title="▶ swap voice",
            border_style="cyan",
        )
    )

    async def _run() -> None:
        track = VoiceTrack(
            VoiceTrackOptions(
                voice=target,
                microphone_device=mic_idx,
                output_device=out_idx,
            )
        )
        track.start(on_status=lambda s: console.print(f"[dim]{s}[/dim]"))
        try:
            if seconds > 0:
                await asyncio.sleep(seconds)
            else:
                while True:
                    await asyncio.sleep(60)
        finally:
            await track.stop()

    try:
        asyncio.run(_run())
        console.print("[green]✓ done.[/green]")
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted.[/dim]")
    except Exception as err:  # noqa: BLE001
        err_console.print(f"[red]voice session failed: {err}[/red]")
        raise typer.Exit(1) from err


# ── voices ─────────────────────────────────────────────────────────────────


@voices_app.command("install")
def voices_install() -> None:
    """Install voice deps + download OpenVoice tone-color weights (~300 MB).

    Pulls torch + torchaudio + sounddevice + librosa + soundfile +
    huggingface-hub + OpenVoice via pip, then downloads the converter
    checkpoint to the user data dir.
    """
    from . import voice_ops, voice_prereq

    pre = voice_prereq.check_all()
    if not pre.gpu.ok:
        err_console.print(
            f"[red]✗ {pre.gpu.label}.[/red] {pre.gpu.hint or ''}\n"
            "[red]Voice features require a supported GPU.[/red]"
        )
        raise typer.Exit(2)

    if pre.deps_installed.ok:
        console.print("[green]✓ voice deps already installed.[/green]")
    else:
        console.print(
            "Installing voice deps via "
            f"[bold]{sys.executable} -m pip install '.[voice]'[/bold] …"
        )
        if not voice_ops.install_voice_deps():
            err_console.print("[red]pip install failed.[/red]")
            raise typer.Exit(1)
        console.print("[green]✓ voice deps installed.[/green]")

    if pre.weights.ok:
        console.print(
            f"[green]✓ OpenVoice weights already present at "
            f"{voice_prereq.openvoice_weights_dir()}.[/green]"
        )
    else:
        console.print("Downloading OpenVoice weights …")
        target = voice_ops.download_openvoice_weights()
        console.print(f"[green]✓ weights ready at {target}.[/green]")

    console.print(
        "\n[dim]Voice cloning ready. Open `swap gui`, click Enable, "
        "pick a voice from the library and a virtual audio cable as the "
        "output. Click ③ Live and you're cloning live.[/dim]"
    )


@voices_app.command("uninstall")
def voices_uninstall() -> None:
    """Remove OpenVoice weights from disk."""
    from . import voice_ops

    if voice_ops.uninstall_openvoice_weights():
        console.print("[green]✓ OpenVoice weights removed.[/green]")
    else:
        console.print("[dim]No weights present — nothing to remove.[/dim]")


@voices_app.command("list")
def voices_list() -> None:
    """List the bundled library + any user-added voices."""
    from . import voice_ops

    library, user = voice_ops.list_all()

    table = Table(title="Library voices (bundled)", show_header=True, box=None)
    table.add_column("id", style="dim")
    table.add_column("name")
    table.add_column("description")
    if not library:
        table.add_row("(empty)", "—", "Run a swap-cli release that bundles voices.")
    for v in library:
        table.add_row(v.id, v.name, v.description)
    console.print(table)

    table = Table(title="Your voices (custom)", show_header=True, box=None)
    table.add_column("id", style="dim")
    table.add_column("name")
    table.add_column("description")
    if not user:
        table.add_row(
            "(empty)", "—",
            "Add one with `swap voices add ./me.wav --name \"Me\"`."
        )
    for v in user:
        table.add_row(v.id, v.name, v.description)
    console.print(table)


@voices_app.command("add")
def voices_add(
    path: Annotated[Path, typer.Argument(help="Path to a WAV/MP3 reference.")],
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Display name. Defaults to file stem."),
    ] = None,
) -> None:
    """Add a custom reference voice from a WAV/MP3."""
    from . import voice_ops

    try:
        voice = voice_ops.add_user_voice(path, name)
    except FileNotFoundError as err:
        err_console.print(f"[red]{err}[/red]")
        raise typer.Exit(1) from err
    except RuntimeError as err:
        err_console.print(f"[red]{err}[/red]")
        raise typer.Exit(2) from err

    console.print(
        f"[green]✓ Added[/green] [bold]{voice.name}[/bold] "
        f"(id: {voice.id}) — {voice.description}\n"
        f"[dim]Saved to {voice_ops.user_voices_dir() / (voice.id + '.json')}[/dim]"
    )


@voices_app.command("remove")
def voices_remove(
    name: Annotated[str, typer.Argument(help="Voice name or id to remove.")],
) -> None:
    """Remove a custom voice from your library."""
    from . import voice_ops

    if voice_ops.remove_user_voice(name):
        console.print(f"[green]✓ Removed[/green] {name}.")
    else:
        err_console.print(
            f"[red]No user voice named/id matching '{name}'.[/red]"
        )
        raise typer.Exit(1)


@voices_app.command("test")
def voices_test(
    voice: Annotated[
        str,
        typer.Option(
            "--voice",
            "-v",
            help="Voice id or name (e.g. 'aria', 'Aria', or your custom voice).",
        ),
    ],
    seconds: Annotated[
        int,
        typer.Option(
            "--seconds",
            "-s",
            help="Stop automatically after N seconds. 0 = run until Ctrl+C.",
        ),
    ] = 30,
    mic: Annotated[
        int | None,
        typer.Option("--mic", help="Microphone device index. Default: system default."),
    ] = None,
    output: Annotated[
        int | None,
        typer.Option("--output", help="Output device index. Default: auto-detected virtual cable."),
    ] = None,
) -> None:
    """Test voice cloning briefly (default 30s). Lighter sibling of `swap voice`."""
    _run_voice_session(voice_name=voice, mic=mic, output=output, seconds=seconds)


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
