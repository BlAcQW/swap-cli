"""Typer CLI entrypoint for swap-cli."""

from __future__ import annotations

import asyncio
import shutil
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

    # Sprint 14d: gate on engine readiness before spinning up audio
    # devices. Otherwise the user gets an opaque RuntimeError mid-init.
    from . import voice_engines

    cfg_engine = config.load().voice_engine
    engine = voice_engines.get_engine(cfg_engine)
    if not engine.is_available():
        err_console.print(
            f"[red]Voice engine '{cfg_engine}' isn't installed.[/red] "
            "Run [bold]swap voices install[/bold]."
        )
        raise typer.Exit(1)
    if not engine.is_ready():
        err_console.print(
            "[red]No RVC .pth voices registered.[/red] "
            "Add one with [bold]swap voices add-rvc /path/to/model.pth --name X[/bold].\n"
            "Get models from [link]https://huggingface.co/lj1995/VoiceConversionWebUI[/link] "
            "or [link]https://weights.gg[/link], or train your own with Applio."
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

    cfg = config.load()

    async def _run() -> None:
        track = VoiceTrack(
            VoiceTrackOptions(
                voice=target,
                microphone_device=mic_idx,
                output_device=out_idx,
                engine_name=cfg.voice_engine,
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
def voices_install(
    starter: Annotated[
        str | None,
        typer.Option(
            "--starter",
            help="Auto-download a specific catalog voice after install (e.g. soft-asmr).",
        ),
    ] = None,
    no_starter: Annotated[
        bool,
        typer.Option(
            "--no-starter",
            help="Skip the post-install starter-voice prompt entirely.",
        ),
    ] = False,
) -> None:
    """Install voice deps for the RVC streaming engine.

    Sprint 14e: Pulls CUDA-matched PyTorch first (Win/Linux NVIDIA),
    then RVC's runtime deps + rvc-python + fairseq. ~3–5 GB total.

    Sprint 14g: After deps install, optionally download a starter voice
    so users can run `swap gui --voice` immediately. Interactive by
    default; use --starter <slug> for CI or --no-starter to skip.
    """
    from . import rvc_catalog, voice_engines, voice_ops, voice_prereq

    pre = voice_prereq.check_all()
    if not pre.gpu.ok:
        err_console.print(
            f"[red]✗ {pre.gpu.label}.[/red] {pre.gpu.hint or ''}\n"
            "[red]Voice features require a supported GPU.[/red]"
        )
        raise typer.Exit(2)

    # Surface ffmpeg + Build Tools issues BEFORE the long pip install.
    if not pre.ffmpeg.ok:
        err_console.print(
            f"[red]✗ {pre.ffmpeg.label}.[/red] {pre.ffmpeg.hint}"
        )
        raise typer.Exit(2)
    if not pre.build_tools.ok:
        err_console.print(
            f"[red]✗ {pre.build_tools.label}.[/red] {pre.build_tools.hint}"
        )
        raise typer.Exit(2)

    if pre.deps_installed.ok:
        console.print("[green]✓ voice deps already installed.[/green]")
    else:
        console.print(
            "Installing RVC voice stack — CUDA torch, runtime deps, rvc-python, fairseq …"
        )
        if not voice_ops.install_voice_deps():
            err_console.print("[red]pip install failed.[/red]")
            raise typer.Exit(1)
        console.print("[green]✓ voice deps installed.[/green]")

    # Sprint 14g: voice deps are in place; offer a starter voice so the
    # user has something to immediately try.
    rvc_engine = voice_engines.get_engine("rvc")
    if rvc_engine.is_ready():
        # User already has at least one rvc-* voice — nothing to do.
        return

    if no_starter:
        console.print(
            "\n[dim]No voice registered yet. Browse with `swap voices catalog` "
            "or run `swap voices download <slug>`.[/dim]"
        )
        return

    if starter is not None:
        entry = rvc_catalog.find(starter)
        if entry is None:
            err_console.print(
                f"[red]Unknown catalog slug '{starter}'.[/red] "
                "Run [bold]swap voices catalog[/bold] to see options."
            )
            raise typer.Exit(1)
        _download_catalog_entry(entry)
        return

    # Interactive prompt — only when stdin is a TTY. Piped input skips.
    if not sys.stdin.isatty():
        console.print(
            "\n[dim]No voice registered yet. Run `swap voices catalog` to "
            "browse, or `swap voices install --starter <slug>` for a non-"
            "interactive setup.[/dim]"
        )
        return

    starter_entry = rvc_catalog.starter()
    prompt_msg = (
        f"\n[bold]Download a starter voice?[/bold] "
        f"({starter_entry.name}, ~{starter_entry.total_size_mb} MB) [Y/n] "
    )
    answer = typer.prompt(prompt_msg, default="Y", show_default=False).strip().lower()
    if answer in ("", "y", "yes"):
        _download_catalog_entry(starter_entry)
    else:
        console.print(
            "[dim]Skipped. Browse with `swap voices catalog` whenever you're ready.[/dim]"
        )


def _download_catalog_entry(entry) -> None:  # type: ignore[no-untyped-def]
    """Shared helper for the install starter + standalone download command."""
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    from . import voice_ops

    progress = Progress(
        TextColumn("[bold blue]{task.fields[fname]}[/bold blue]"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task_id: dict[str, int] = {}

    def on_progress(fname: str, done: int, total: int) -> None:
        if fname not in task_id:
            task_id[fname] = progress.add_task("download", total=total or None, fname=fname)
        progress.update(task_id[fname], completed=done)

    with progress:
        try:
            voice = voice_ops.download_catalog_voice(entry, on_progress=on_progress)
        except RuntimeError as err:
            err_console.print(f"[red]{err}[/red]")
            raise typer.Exit(1) from err
        except Exception as err:  # noqa: BLE001 — httpx/network failures
            err_console.print(f"[red]download failed: {err}[/red]")
            raise typer.Exit(1) from err

    console.print(
        f"[green]✓ Voice ready:[/green] [bold]{voice.name}[/bold] (id: {voice.id})\n"
        "[dim]Try it with `swap gui --voice`.[/dim]"
    )


@voices_app.command("list")
def voices_list() -> None:
    """List the user-added RVC voices."""
    from . import voice_ops

    _, user = voice_ops.list_all()

    table = Table(title="Your voices (RVC)", show_header=True, box=None)
    table.add_column("id", style="dim")
    table.add_column("name")
    table.add_column("description")
    if not user:
        table.add_row(
            "(empty)", "—",
            "Download an RVC .pth (weights.gg or HF lj1995/VoiceConversionWebUI), "
            "then `swap voices add-rvc /path/to/model.pth --name \"Name\"`.",
        )
    for v in user:
        table.add_row(v.id, v.name, v.description)
    console.print(table)


@voices_app.command("add")
def voices_add(
    path: Annotated[Path, typer.Argument(help="Path to an RVC .pth model.")],
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Display name. Defaults to file stem."),
    ] = None,
) -> None:
    """Deprecated alias of `swap voices add-rvc`.

    Sprint 14e removed OpenVoice; the WAV-to-embedding flow no longer
    exists. To clone your own voice, train an RVC model with Applio
    (https://github.com/IAHispano/Applio) and register the resulting
    .pth + .index here.
    """
    err_console.print(
        "[yellow]`swap voices add` (OpenVoice WAV→embedding) was removed in 14e.[/yellow]\n"
        "Use [bold]swap voices add-rvc /path/to/model.pth --name X[/bold] for an "
        "RVC .pth, or train your own with Applio.\n"
        "Forwarding to add-rvc with the supplied path …"
    )
    from . import voice_ops

    try:
        voice = voice_ops.add_rvc_voice(path, name)
    except FileNotFoundError as err:
        err_console.print(f"[red]{err}[/red]")
        raise typer.Exit(1) from err
    except (ValueError, RuntimeError) as err:
        err_console.print(f"[red]{err}[/red]")
        raise typer.Exit(2) from err

    console.print(
        f"[green]✓ Added[/green] [bold]{voice.name}[/bold] (id: {voice.id})"
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


@voices_app.command("add-rvc")
def voices_add_rvc(
    pth: Annotated[Path, typer.Argument(help="Path to the RVC .pth model file.")],
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Display name. Defaults to file stem."),
    ] = None,
    index: Annotated[
        Path | None,
        typer.Option(
            "--index",
            "-i",
            help="Optional .index file (Faiss retrieval index — improves quality).",
        ),
    ] = None,
) -> None:
    """Register an RVC voice model. The .pth file is copied into swap-cli's
    model directory; an optional .index file can be provided alongside.

    After registering, switch to the RVC engine with `swap voices engine rvc`
    and the voice appears in the GUI dropdown / `swap voice -v` list.
    """
    from . import voice_ops

    try:
        voice = voice_ops.add_rvc_voice(pth, name=name, index_path=index)
    except (FileNotFoundError, ValueError) as err:
        err_console.print(f"[red]{err}[/red]")
        raise typer.Exit(1) from err

    console.print(
        f"[green]✓ Registered RVC voice[/green] [bold]{voice.name}[/bold] "
        f"(id: {voice.id})"
    )
    console.print(
        "[dim]Switch to RVC engine: [bold]swap voices engine rvc[/bold]\n"
        "Then pick the voice in the GUI dropdown or `swap voice -v "
        f"{voice.id}`.[/dim]"
    )


@voices_app.command("remove-rvc")
def voices_remove_rvc(
    name: Annotated[str, typer.Argument(help="RVC voice name or id.")],
) -> None:
    """Remove an RVC voice and its model files."""
    from . import voice_ops

    if voice_ops.remove_rvc_voice(name):
        console.print(f"[green]✓ Removed RVC voice[/green] {name}.")
    else:
        err_console.print(f"[red]No RVC voice matching '{name}'.[/red]")
        raise typer.Exit(1)


@voices_app.command("catalog")
def voices_catalog() -> None:
    """List curated RVC voices available via `swap voices download`.

    These are mirrored to our GitHub Releases — stable URLs, license-
    vetted personas (no real people, no copyrighted IP).
    """
    from . import rvc_catalog

    table = Table(title="Curated voice catalog", show_header=True, box=None)
    table.add_column("slug", style="dim")
    table.add_column("name")
    table.add_column("size", justify="right")
    table.add_column("description")
    for entry in rvc_catalog.CATALOG:
        starter_marker = " [yellow]★[/yellow]" if entry.slug == rvc_catalog.STARTER_SLUG else ""
        table.add_row(
            entry.slug + starter_marker,
            entry.name,
            f"{entry.total_size_mb} MB",
            entry.description,
        )
    console.print(table)
    console.print(
        f"\n[dim]★ = default starter (smallest). "
        f"Download with [bold]swap voices download <slug>[/bold].[/dim]"
    )


@voices_app.command("download")
def voices_download(
    slug: Annotated[
        str,
        typer.Argument(help="Catalog slug — see `swap voices catalog` for options."),
    ],
) -> None:
    """Download a curated RVC voice from our mirror and register it."""
    from . import rvc_catalog, voice_engines

    rvc_engine = voice_engines.get_engine("rvc")
    if not rvc_engine.is_available():
        err_console.print(
            "[red]RVC isn't installed.[/red] Run [bold]swap voices install[/bold] first."
        )
        raise typer.Exit(1)

    entry = rvc_catalog.find(slug)
    if entry is None:
        err_console.print(
            f"[red]Unknown catalog slug '{slug}'.[/red] "
            "Run [bold]swap voices catalog[/bold] to see options."
        )
        raise typer.Exit(1)

    _download_catalog_entry(entry)


@voices_app.command("engine")
def voices_engine(
    name: Annotated[
        str | None,
        typer.Argument(
            help="Engine to set as default. Omit to print current setting + list.",
        ),
    ] = None,
) -> None:
    """Pick which engine handles live voice streaming.

    Sprint 14e: only 'rvc' is registered. Field exists for forward
    compatibility with future engines (Applio, GPT-SoVITS).
    """
    from . import config as _config
    from . import voice_engines

    cfg = _config.load()
    if name is None:
        # Show current + available engines. "available?" = deps installed,
        # "ready?" = also has at least one usable voice (RVC needs a .pth).
        table = Table(title="Voice engines", show_header=True, box=None)
        table.add_column("name", style="dim")
        table.add_column("display name")
        table.add_column("available?")
        table.add_column("ready?")
        table.add_column("active")
        for engine_name in voice_engines.available_engines():
            engine = voice_engines.get_engine(engine_name)
            available = "[green]✓[/green]" if engine.is_available() else "[red]✗[/red]"
            ready = "[green]✓[/green]" if engine.is_ready() else "[red]✗[/red]"
            active = "[bold]●[/bold]" if engine_name == cfg.voice_engine else ""
            table.add_row(engine.name, engine.display_name, available, ready, active)
        console.print(table)
        return

    if name not in voice_engines.available_engines():
        err_console.print(
            f"[red]Unknown engine '{name}'.[/red] "
            f"Known: {voice_engines.available_engines()}"
        )
        raise typer.Exit(1)

    engine = voice_engines.get_engine(name)
    if not engine.is_available():
        # Refuse — Sprint 14d. Setting an unavailable engine as active
        # left users in a corrupt state (engine ✗ ●) where Live failed
        # opaquely. Force them to install first.
        err_console.print(
            f"[red]✗ Engine '{name}' isn't installed.[/red] "
            "Run [bold]swap voices install[/bold] first, then retry."
        )
        raise typer.Exit(1)

    _config.update(voice_engine=name)
    console.print(
        f"[green]✓ Default voice engine set to[/green] [bold]{name}[/bold]"
        f" ({engine.display_name})."
    )
    # Soft-warn if available but not ready (e.g. RVC installed, no .pth yet).
    if not engine.is_ready():
        console.print(
            f"[yellow]Note: '{name}' is available but not ready — "
            "no usable voice registered yet.[/yellow] "
            "For RVC, run [bold]swap voices add-rvc /path/to/model.pth --name X[/bold]."
        )


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

    # ffmpeg on PATH — RVC needs it for any non-WAV codec; missing
    # ffmpeg accounts for ~half the 'file failed to load' errors per
    # the upstream RVC community.
    if shutil.which("ffmpeg") is not None:
        table.add_row("ffmpeg on PATH", "[green]✓[/green]")
    else:
        if sys.platform == "win32":
            hint = "winget install Gyan.FFmpeg"
        elif sys.platform == "darwin":
            hint = "brew install ffmpeg"
        else:
            hint = "sudo apt install ffmpeg"
        table.add_row("ffmpeg on PATH", f"[red]✗ {hint}[/red]")

    # macOS-only: customtkinter needs Tcl/Tk >= 8.6.9. The system Python
    # ships 8.5.9 which fails silently or renders broken windows. Surface
    # this here so users know to switch to python.org Python or
    # `brew install python-tk@3.11`.
    if sys.platform == "darwin":
        table.add_row("tcl/tk version", _tcl_tk_label())

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


def _tcl_tk_label() -> str:
    """Return a doctor-row label for the Tcl/Tk version on macOS.

    customtkinter requires >= 8.6.9; macOS system Python is stuck on
    Apple's 8.5.9 (which has known Tk bugs). Direct users to
    python.org Python or `brew install python-tk@3.11` when below the
    floor.
    """
    try:
        import tkinter

        ver = tkinter.Tcl().call("info", "patchlevel")
        parts = [int(p) for p in ver.split(".")[:3]]
        while len(parts) < 3:
            parts.append(0)
        if tuple(parts) >= (8, 6, 9):
            return f"[green]✓ {ver}[/green]"
        return (
            f"[red]✗ {ver} — need ≥ 8.6.9; "
            "install python.org Python or `brew install python-tk@3.11`[/red]"
        )
    except Exception as err:  # noqa: BLE001
        return f"[red]✗ {err}[/red]"
