"""Orchestrates a single Decart Lucy 2 realtime session end-to-end.

Camera → aiortc → Decart → remote track → cv2 window.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .camera import CameraTrack
from .display import Display, default_recording_path

# Decart's connect timeout now lives with the Decart provider. Re-exported here
# because it's part of this module's public surface (and guarded by a test).
from .providers.decart import CONNECT_TIMEOUT_S  # noqa: F401


def _is_harmless_ice_noise(context: dict) -> bool:
    """True for the benign aioice STUN teardown noise.

    When a WebRTC connection drops, aioice's retry timer can fire *after* the
    STUN transaction's future is already done and call ``set_exception`` on it,
    raising ``asyncio.InvalidStateError`` from inside the ``Transaction.__retry``
    callback. asyncio logs it as "Exception in callback". It does not affect our
    own teardown (which we handle explicitly), so we swallow just this to keep
    the console clean during network blips. Everything else is left untouched.
    """
    if isinstance(context.get("exception"), asyncio.InvalidStateError):
        return True
    blob = f"{context.get('message', '')} {context.get('handle', '')}"
    return "Transaction.__retry" in blob


def _noop_status(_msg: str) -> None:
    return None

DEFAULT_PROMPT = (
    "Substitute the person in the video with the person in the reference image. "
    "Match their face, body, and identity while preserving the original pose, "
    "motion, and lighting."
)

DEFAULT_MODEL_NAME = "lucy-2"


@dataclass(frozen=True)
class RunOptions:
    decart_api_key: str
    reference: str | None  # path or URL of the reference identity image
    prompt: str
    model_name: str = DEFAULT_MODEL_NAME
    camera_device: int = 0
    record: Path | None = None
    # Optional callback for surfacing state to a UI status bar. Always called
    # from the asyncio worker thread; the GUI is responsible for marshalling
    # back to the tk main thread (e.g. via root.after).
    on_status_change: Callable[[str], None] = field(default=_noop_status)
    # Optional callback that hands the caller a thread-safe stop function once
    # the session is set up. Calling that function from any thread cleanly
    # winds down the live session.
    on_runtime_ready: Callable[[Callable[[], None]], None] | None = None
    # Show the cv2 preview window from the Display's own (worker-thread) loop.
    # False on macOS GUI: the GUI pumps the SAME cv2 window from the main thread
    # (macOS forbids cv2 windows off the main thread). Windows/CLI keep True.
    show_preview_window: bool = True
    # Optional callback handed the live Display once created, so the macOS GUI
    # can read frames (latest_frame()) to drive the window + watermark capture.
    on_display_ready: Callable[[Any], None] | None = None
    # Voice cloning (optional, sprint 13). All None = video-only, current behavior.
    # When all three are set, run_session will spin a parallel voice task in 13b.
    reference_voice: str | None = None  # voice id (library) or path to WAV/MP3
    microphone_device: int | None = None  # sounddevice mic index
    voice_output_device: int | None = None  # virtual cable index for routing
    # Sprint 14k: also push Decart frames to the OBS Virtual Camera driver
    # so video-call apps see the deepfake as a camera device. Default off
    # so legacy users / CI without the driver aren't surprised.
    virtual_camera: bool = False
    # Sprint 15: per-frame watermark removal. Default off — same back-compat
    # rationale as virtual_camera, plus it costs detection+inpaint latency.
    remove_watermark: bool = False
    watermark_template: str | None = None
    watermark_method: str = "template"
    watermark_removal: str = "reconstruct"  # "reconstruct" | "blur"
    watermark_threshold: float = 0.50
    watermark_inpaint_radius: int = 3
    watermark_template_width: int | None = None  # capture-frame width (scale center)
    # Sprint 17: independent color/shape safety net (on by default).
    watermark_signature_fallback: bool = True
    # Sprint 18: which realtime backend runs the swap. "decart" (default, WebRTC)
    # or "fal" (the same Lucy model resold by fal.ai, over a websocket). fal is a
    # separate Decart account, so its watermark policy may differ.
    provider: str = "decart"
    fal_api_key: str = ""


async def _require_valid_license(notify: Callable[[str], None]) -> None:
    """Refuse to start a session without a valid license.

    Lives here, not in the GUI's Live handler, because run_session is the ONE
    choke point every caller funnels through (CLI, GUI, anything future) — the
    GUI previously did only a "best-effort, non-blocking" check that set a
    status label, so any non-empty key string could stream indefinitely.

    force_online=True is deliberate: the plain cached path would honour a stale
    "valid" for up to OFFLINE_GRACE_SECONDS, so a REVOKED user could keep
    streaming for a day. Going to the server on every session start makes a
    revocation effective at the next launch. The offline grace still applies —
    validate() falls back to the cache when the server is genuinely unreachable
    (a network error), so a paying user on a flaky connection keeps working.
    Only "never validated AND unreachable" is refused, matching `swap run`.
    """
    from . import license

    notify("Verifying license…")
    try:
        status = await license.validate(force_online=True)
    except license.LicenseError as err:
        raise RuntimeError(
            f"Couldn't verify your license: {err}\n"
            "Check your internet connection and try again."
        ) from err
    if not status.valid:
        raise RuntimeError(
            f"License check failed: {license.explain(status.reason)}.\n"
            "Buy or renew at https://swap.storelygh.com"
        )
    print(f"[runtime] license ok ({status.reason})", flush=True)


async def run_session(opts: RunOptions) -> None:
    """Open a realtime Decart session and stream until the user quits."""
    print("[runtime] entering run_session", flush=True)

    # Gate FIRST — before the camera is opened or any provider is contacted, so
    # an unlicensed run never grabs the webcam or bills a backend.
    await _require_valid_license(opts.on_status_change)

    # Make TLS verification trust certifi's public-CA bundle IN ADDITION to the
    # OS store (must happen before aiohttp builds its SSL context below). A fresh
    # Windows machine's cert store can lack the public root for api.decart.ai
    # (Windows fetches roots on-demand for browsers; Python/OpenSSL doesn't) →
    # CERTIFICATE_VERIFY_FAILED. The base context still loads the OS store, so
    # corporate/AV roots keep working too. Best-effort: never block a session.
    try:
        import ssl

        import certifi

        _orig_create_default_context = ssl.create_default_context

        def _ctx_with_certifi(*args: Any, **kwargs: Any) -> ssl.SSLContext:
            ctx = _orig_create_default_context(*args, **kwargs)
            with suppress(Exception):
                ctx.load_verify_locations(certifi.where())
            return ctx

        ssl.create_default_context = _ctx_with_certifi  # type: ignore[assignment]
        print("[runtime] TLS: trusting OS store + certifi", flush=True)
    except Exception as err:  # fall back to default certs; never block a session
        print(f"[runtime] TLS: cert setup skipped ({err}); using defaults", flush=True)

    # Pick the backend. Everything below is provider-agnostic — the provider owns
    # connecting, applying the reference identity, and its own teardown. The
    # decart>=0.1 (LiveKit) guard now lives in providers/decart.py.
    from .providers import get_provider

    provider = get_provider(getattr(opts, "provider", "decart"))
    print(f"[runtime] provider={provider.name}", flush=True)

    spec = provider.frame_spec(opts)
    camera = CameraTrack(
        device=opts.camera_device,
        width=spec.width,
        height=spec.height,
        fps=spec.fps,
    )
    print(f"[runtime] camera opened on device {opts.camera_device}", flush=True)

    quit_event = asyncio.Event()
    notify = opts.on_status_change

    # Keep connection-drop noise off the console (the user demos to clients).
    # aioice fires a harmless InvalidStateError from a STUN retry timer during
    # teardown; swallow only that, delegate everything else to the prior handler.
    running_loop = asyncio.get_running_loop()
    _prev_exc_handler = running_loop.get_exception_handler()

    def _quiet_loop_exceptions(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        if _is_harmless_ice_noise(context):
            return
        if _prev_exc_handler is not None:
            _prev_exc_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    running_loop.set_exception_handler(_quiet_loop_exceptions)
    voice_track: Any = None  # populated by _maybe_start_voice if opts say so

    # Hand the caller a thread-safe stop function. The GUI's Stop button
    # calls this from the tk main thread; we schedule quit_event.set() on
    # this asyncio loop via call_soon_threadsafe.
    if opts.on_runtime_ready is not None:

        def _request_stop() -> None:
            running_loop.call_soon_threadsafe(quit_event.set)

        opts.on_runtime_ready(_request_stop)

    session: Any = None
    display: Display | None = None
    try:
        session = await provider.connect(
            camera=camera,
            opts=opts,
            on_remote_stream=lambda remote_track: _on_remote_stream(
                remote_track,
                record=opts.record,
                quit_event=quit_event,
                display_box=display_box,
                virtual_camera=opts.virtual_camera,
                watermark=_build_watermark_remover(opts),
                show_window=opts.show_preview_window,
                on_display_ready=opts.on_display_ready,
            ),
            notify=notify,
            quit_event=quit_event,
        )

        # Apply the reference identity (URL or local file path; bytes also OK).
        # If the connection drops mid-send, the SDK surfaces a "Image send timed
        # out" error — a consequence of the drop, not a bug. Degrade to a clean
        # one-liner + graceful teardown instead of letting a traceback escape.
        if opts.reference:
            notify("Applying reference identity…")
            try:
                await session.set_identity(opts.prompt, opts.reference)
            except asyncio.CancelledError:
                raise  # genuine cancellation → GUI handles it gracefully
            except Exception as exc:  # a drop mid-send must not crash the session
                reason = "connection dropped" if quit_event.is_set() else str(exc)
                print(
                    f"[runtime] reference identity not applied — {reason}; "
                    "ending session",
                    flush=True,
                )
                notify("Session ended — connection interrupted.")
                return  # finally-block teardown still runs (disconnect, camera…)

        notify("Connected · waiting for first frame…")

        # Sprint 13b: voice cloning runs as a parallel asyncio task in the
        # same loop. If it fails, video continues — voice failures don't
        # take the session down.
        voice_track = await _maybe_start_voice(opts, notify)

        await quit_event.wait()
        print()  # newline after the tick line
    finally:
        # aiortc's pc.close() can raise asyncio.CancelledError during teardown
        # (a BaseException, NOT caught by suppress(Exception)). Suppress it too
        # so a connection drop / abrupt stop tears down cleanly instead of
        # escaping as a traceback.
        if voice_track is not None:
            with suppress(Exception, asyncio.CancelledError):
                await voice_track.stop()
        if session is not None:
            with suppress(Exception, asyncio.CancelledError):
                await session.disconnect()
        # Display is started inside `_on_remote_stream` and stored in display_box[0].
        if display_box and display_box[0] is not None:
            with suppress(Exception, asyncio.CancelledError):
                await display_box[0].stop()
        camera.stop()


async def _maybe_start_voice(opts: RunOptions, notify: Callable[[str], None]) -> Any:
    """Spin a parallel voice cloning track if the user opted in.

    Returns the started VoiceTrack (so run_session can stop it on cleanup),
    or None if voice was off / unavailable. All failures are caught and
    surfaced via notify — voice issues never kill the video session.
    """
    if not (opts.reference_voice and opts.microphone_device is not None):
        return None
    try:
        from . import config as _config
        from . import voice_library, voice_track as voice_track_mod

        target = voice_library.find_voice(opts.reference_voice)
        if target is None:
            notify(f"Voice: '{opts.reference_voice}' not found in library.")
            return None
        cfg = _config.load()
        track = voice_track_mod.VoiceTrack(
            voice_track_mod.VoiceTrackOptions(
                voice=target,
                microphone_device=opts.microphone_device,
                output_device=opts.voice_output_device,
                engine_name=cfg.voice_engine,
            )
        )
        track.start(on_status=notify)
        notify(f"Voice: started ({target.name})")
        return track
    except Exception as err:  # noqa: BLE001
        notify(f"Voice: disabled ({err})")
        print(f"[runtime] voice failed to start: {err}", flush=True)
        return None


# Mutable container so the on_remote_stream callback can hand the display
# back to the run_session frame for cleanup. Cleaner than nonlocal in a lambda.
display_box: list[Display | None] = [None]


def _build_watermark_remover(opts: RunOptions) -> Any:
    """Construct a WatermarkRemover from RunOptions, or None if off.

    Lazy-imported so `swap version` / `config` don't pull cv2 paths via
    this module (consistent with the lazy decart/aiortc imports above).
    Never raises — a misconfigured remover degrades to no removal.
    """
    if not opts.remove_watermark:
        return None
    try:
        from .watermark import (
            BUNDLED_TEMPLATE_REF_WIDTH,
            WatermarkParams,
            WatermarkRemover,
            bundled_template_path,
        )

        # User's captured template wins; otherwise fall back to the bundled
        # default so removal works out of the box (no capture step).
        template = Path(opts.watermark_template) if opts.watermark_template else None
        if template is None:
            template = bundled_template_path()
        method = "threshold" if opts.watermark_method == "threshold" else "template"
        removal = "blur" if opts.watermark_removal == "blur" else "reconstruct"
        if method == "template" and template is None:
            print(
                "[runtime] watermark removal on but no template available — "
                "run `swap capture-watermark` (or press W in the preview). "
                "Skipping removal.",
                flush=True,
            )
            return None
        # A captured template carries the width it was grabbed at so the
        # scale search centers exactly; the bundled default uses its own known
        # crop width.
        is_custom = bool(opts.watermark_template)
        ref_width = (
            (opts.watermark_template_width or 1280)
            if is_custom
            else BUNDLED_TEMPLATE_REF_WIDTH
        )
        params = WatermarkParams(
            method=method,
            removal=removal,
            template_path=template,
            threshold=opts.watermark_threshold,
            inpaint_radius=opts.watermark_inpaint_radius,
            template_ref_width=ref_width,
            signature_fallback=opts.watermark_signature_fallback,
        )
        # Log on SUCCESS — previously silent, which is why a non-matching
        # session showed no `[watermark]` lines at all.
        print(
            f"[runtime] watermark removal ON · template="
            f"{'custom' if is_custom else 'bundled'} ({template.name}) · "
            f"method={method} · removal={removal} · gate={opts.watermark_threshold}",
            flush=True,
        )
        return WatermarkRemover(params)
    except Exception as err:  # noqa: BLE001 — removal is optional
        print(f"[runtime] watermark remover init failed: {err}", flush=True)
        return None


def _on_remote_stream(
    remote_track: Any,
    *,
    record: Path | None,
    quit_event: asyncio.Event,
    display_box: list[Display | None],
    virtual_camera: bool = False,
    watermark: Any = None,
    show_window: bool = True,
    on_display_ready: Callable[[Any], None] | None = None,
) -> None:
    record_path = record if record is not None else None
    if record is not None and not record.is_absolute():
        record_path = record
    if record == Path("auto"):
        record_path = default_recording_path()

    disp = Display(
        track=remote_track,
        record_path=record_path,
        on_quit=quit_event.set,
        virtual_camera=virtual_camera,
        watermark=watermark,
        show_window=show_window,
    )
    disp.start()
    display_box[0] = disp
    if on_display_ready is not None:
        with suppress(Exception):
            on_display_ready(disp)


