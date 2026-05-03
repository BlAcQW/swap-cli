"""customtkinter GUI for swap-cli.

Layout mirrors Deep-Live-Cam: face thumbnail, camera dropdown, options,
Start / Destroy / Preview / Live buttons. The Live button opens the
realtime stream in a separate window via the existing display.py.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING, Callable

import customtkinter as ctk
from PIL import Image

from . import config, license
from .devices import CameraDevice, enumerate_cameras
from .runtime import DEFAULT_PROMPT, RunOptions, run_session
from .version import __version__

if TYPE_CHECKING:
    pass

if sys.platform == "win32":
    # Make Tk render correctly on high-DPI Alienware/Surface/4K displays.
    # Without this, customtkinter's internal scaling fights the OS and the
    # window can render off-screen or at the wrong size.
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
    except Exception:  # noqa: BLE001 — best effort on older Windows
        pass

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

THUMB_SIZE = (140, 140)


class SwapGUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"swap-cli {__version__} · live deepfake")
        W, H = 520, 720
        self.minsize(480, 660)
        # Center on the primary monitor so the window can't open off-screen
        # on multi-monitor setups (Alienware ships with this misconfigured).
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = max(0, (sw - W) // 2)
        y = max(0, (sh - H) // 2)
        self.geometry(f"{W}x{H}+{x}+{y}")
        # Flash topmost so the window pops to the foreground even if another
        # app stole focus during start-up. Released after 200ms so users can
        # alt-tab freely afterward.
        self.after(0, self._raise_to_front)

        self._reference_path: Path | None = None
        self._thumb_image: ctk.CTkImage | None = None
        self._cameras: list[CameraDevice] = []
        self._session_thread: threading.Thread | None = None
        self._session_loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        # Set by runtime.run_session via the on_runtime_ready callback. Calling
        # this from the tk main thread cleanly winds the session down.
        self._stop_session: Callable[[], None] | None = None
        # Voice-only test state (no Decart). Independent of _session_thread.
        self._voice_test_thread: threading.Thread | None = None
        self._voice_test_loop: asyncio.AbstractEventLoop | None = None
        self._voice_test_stop: Callable[[], None] | None = None
        self._status_var = tk.StringVar(value="Idle.")

        self._build_ui()
        self._refresh_cameras()
        self._refresh_status()
        self._refresh_voice_section()

    # ── UI build ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=20, pady=18)

        # Top: face thumbnail + reference picker
        top = ctk.CTkFrame(outer, fg_color="transparent")
        top.pack(fill="x")
        self._face_label = ctk.CTkLabel(
            top,
            text="No face\nselected",
            width=THUMB_SIZE[0],
            height=THUMB_SIZE[1],
            corner_radius=12,
            fg_color="#1f2937",
            text_color="#9ca3af",
            font=ctk.CTkFont(size=11),
        )
        self._face_label.pack(pady=(0, 10))

        self._select_face_btn = ctk.CTkButton(
            top,
            text="① Select a face",
            command=self._on_select_face,
            height=42,
            corner_radius=8,
        )
        self._select_face_btn.pack(fill="x")

        # Options row 1
        opts = ctk.CTkFrame(outer, fg_color="transparent")
        opts.pack(fill="x", pady=(18, 0))
        opts.columnconfigure((0, 1), weight=1, uniform="opt")

        self._mirror_var = tk.BooleanVar(value=True)
        ctk.CTkSwitch(
            opts, text="Mirror camera", variable=self._mirror_var
        ).grid(row=0, column=0, sticky="w", pady=4)

        self._record_var = tk.BooleanVar(value=False)
        ctk.CTkSwitch(
            opts, text="Record to MP4", variable=self._record_var
        ).grid(row=0, column=1, sticky="w", pady=4)

        # Model selector. Decart fixes width/height/fps per model — we display
        # the native dimensions next to each option so the user knows what
        # they're getting. There is no orientation knob: the SDK only accepts
        # a model's native dimensions; portrait / 1:1 / 4:5 / 3:4 framing
        # would have to be done post-hoc on the saved file (not exposed yet).
        tier_row = ctk.CTkFrame(outer, fg_color="transparent")
        tier_row.pack(fill="x", pady=(14, 0))
        ctk.CTkLabel(
            tier_row, text="Model", anchor="w", font=ctk.CTkFont(size=11)
        ).pack(anchor="w")

        from decart import models as _decart_models

        self._model_specs: dict[str, tuple[int, int, int]] = {}
        labels: list[str] = []
        for name in ("lucy-2", "lucy-2.1"):
            try:
                m = _decart_models.realtime(name)
                self._model_specs[name] = (int(m.width), int(m.height), int(m.fps))
                labels.append(f"{name}  ({m.width}×{m.height}, {m.fps} fps)")
            except Exception:
                continue
        if not labels:
            labels = ["lucy-2 (default)"]
            self._model_specs["lucy-2"] = (1280, 720, 20)

        self._tier_var = tk.StringVar(value=labels[0])
        self._tier_dropdown = ctk.CTkComboBox(
            tier_row,
            values=labels,
            variable=self._tier_var,
            state="readonly",
            height=34,
        )
        self._tier_dropdown.pack(fill="x")
        ctk.CTkLabel(
            tier_row,
            text="Decart fixes the resolution per model. Pricing tier "
            "(Fast / Pro) is set on your Decart account.",
            anchor="w",
            text_color="#6b7280",
            font=ctk.CTkFont(size=10),
            wraplength=460,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))

        # Prompt — hidden behind Advanced. Default deepfake template covers
        # the common case ("become this face"). Power users expand to tweak.
        adv_row = ctk.CTkFrame(outer, fg_color="transparent")
        adv_row.pack(fill="x", pady=(14, 0))
        self._advanced_open = tk.BooleanVar(value=False)
        self._advanced_toggle = ctk.CTkButton(
            adv_row,
            text="⚙ Advanced",
            command=self._toggle_advanced,
            height=28,
            corner_radius=6,
            fg_color="transparent",
            hover_color="#1f2937",
            anchor="w",
            text_color="#9ca3af",
        )
        self._advanced_toggle.pack(fill="x")

        self._advanced_panel = ctk.CTkFrame(outer, fg_color="transparent")
        ctk.CTkLabel(
            self._advanced_panel,
            text="Prompt (optional — guides the swap)",
            anchor="w",
            font=ctk.CTkFont(size=11),
        ).pack(anchor="w", pady=(8, 0))
        self._prompt_box = ctk.CTkTextbox(self._advanced_panel, height=72)
        self._prompt_box.pack(fill="x")
        self._prompt_box.insert("1.0", DEFAULT_PROMPT)
        # Note: panel is NOT packed by default (collapsed).

        # Camera dropdown + refresh
        cam_row = ctk.CTkFrame(outer, fg_color="transparent")
        cam_row.pack(fill="x", pady=(14, 0))
        cam_row.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            cam_row, text="② Select camera", anchor="w", font=ctk.CTkFont(size=11)
        ).grid(row=0, column=0, sticky="w", columnspan=2)

        self._camera_var = tk.StringVar(value="No cameras detected")
        self._camera_dropdown = ctk.CTkComboBox(
            cam_row,
            values=[],
            variable=self._camera_var,
            state="readonly",
            height=34,
        )
        self._camera_dropdown.grid(row=1, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            cam_row,
            text="↻",
            width=42,
            height=34,
            command=self._refresh_cameras,
        ).grid(row=1, column=1)

        # Voice section (Sprint 13b). Two states:
        # - collapsed: single line with Off label + Enable… button → opens modal
        # - expanded: library dropdown + mic dropdown when toggle ON
        # Restored from config.voice_enabled at startup.
        voice_row = ctk.CTkFrame(outer, fg_color="transparent")
        voice_row.pack(fill="x", pady=(14, 0))
        ctk.CTkLabel(
            voice_row, text="Voice", anchor="w", font=ctk.CTkFont(size=11)
        ).pack(anchor="w")

        # Collapsed row (default state)
        self._voice_collapsed_row = ctk.CTkFrame(voice_row, fg_color="transparent")
        self._voice_collapsed_row.pack(fill="x")
        ctk.CTkLabel(
            self._voice_collapsed_row,
            text="☐ Off · clone your voice (requires GPU)",
            anchor="w",
            text_color="#6b7280",
        ).pack(side="left", fill="x", expand=True)
        self._enable_voice_btn = ctk.CTkButton(
            self._voice_collapsed_row,
            text="Enable…",
            width=86,
            height=28,
            corner_radius=6,
            command=self._on_enable_voice,
        )
        self._enable_voice_btn.pack(side="right")

        # Expanded row (hidden until voice is enabled)
        self._voice_expanded_row = ctk.CTkFrame(voice_row, fg_color="transparent")
        # not packed yet — _refresh_voice_section() shows it when voice is on
        self._voice_var = tk.StringVar(value="(no voices found)")
        self._voice_dropdown = ctk.CTkComboBox(
            self._voice_expanded_row,
            values=[],
            variable=self._voice_var,
            state="readonly",
            height=30,
        )
        self._voice_dropdown.pack(fill="x", pady=(2, 4))
        voice_actions = ctk.CTkFrame(self._voice_expanded_row, fg_color="transparent")
        voice_actions.pack(fill="x")
        self._voice_status_label = ctk.CTkLabel(
            voice_actions,
            text="On · cloning enabled",
            anchor="w",
            text_color="#10b981",
        )
        self._voice_status_label.pack(side="left", fill="x", expand=True)
        self._test_voice_btn = ctk.CTkButton(
            voice_actions,
            text="Test voice",
            width=92,
            height=24,
            corner_radius=6,
            fg_color="#0ea5e9",
            hover_color="#0284c7",
            command=self._on_test_voice,
        )
        self._test_voice_btn.pack(side="right", padx=(0, 6))
        self._disable_voice_btn = ctk.CTkButton(
            voice_actions,
            text="Disable",
            width=72,
            height=24,
            corner_radius=6,
            fg_color="#374151",
            hover_color="#4b5563",
            command=self._on_disable_voice,
        )
        self._disable_voice_btn.pack(side="right")

        # Action buttons row
        actions = ctk.CTkFrame(outer, fg_color="transparent")
        actions.pack(fill="x", pady=(20, 0))
        actions.columnconfigure((0, 1, 2), weight=1, uniform="act")

        self._live_btn = ctk.CTkButton(
            actions,
            text="③ Live",
            command=self._on_live,
            height=44,
            corner_radius=8,
            fg_color="#ec4899",
            hover_color="#db2777",
        )
        self._live_btn.grid(row=0, column=0, columnspan=2, sticky="ew", padx=(0, 6))

        self._stop_btn = ctk.CTkButton(
            actions,
            text="Stop",
            command=self._on_stop,
            height=44,
            corner_radius=8,
            fg_color="#374151",
            hover_color="#4b5563",
            state="disabled",
        )
        self._stop_btn.grid(row=0, column=2, sticky="ew")

        # Status bar
        status = ctk.CTkFrame(outer, fg_color="transparent")
        status.pack(fill="x", pady=(18, 0))
        ctk.CTkLabel(
            status,
            textvariable=self._status_var,
            anchor="w",
            text_color="#9ca3af",
            font=ctk.CTkFont(size=11),
        ).pack(fill="x")

        # Footer
        ctk.CTkLabel(
            outer,
            text="swap-cli — Press Q in the preview window to stop",
            font=ctk.CTkFont(size=10),
            text_color="#6b7280",
        ).pack(side="bottom", pady=(8, 0))

    # ── Actions ───────────────────────────────────────────────────────

    def _toggle_advanced(self) -> None:
        if self._advanced_open.get():
            self._advanced_panel.pack_forget()
            self._advanced_open.set(False)
            self._advanced_toggle.configure(text="⚙ Advanced")
        else:
            # Pack right after the toggle button, before the camera row.
            self._advanced_panel.pack(
                fill="x", pady=(0, 0), after=self._advanced_toggle.master
            )
            self._advanced_open.set(True)
            self._advanced_toggle.configure(text="⚙ Advanced (hide)")

    def _on_select_face(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a reference face",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._reference_path = Path(path)
        try:
            img = Image.open(path)
            img.thumbnail(THUMB_SIZE)
            self._thumb_image = ctk.CTkImage(light_image=img, dark_image=img, size=THUMB_SIZE)
            self._face_label.configure(image=self._thumb_image, text="")
        except Exception as err:
            self._status_var.set(f"Error loading face: {err}")
            return
        self._status_var.set(f"Face: {self._reference_path.name}")

    def _refresh_cameras(self) -> None:
        self._status_var.set("Probing cameras…")
        self.update_idletasks()
        try:
            self._cameras = enumerate_cameras()
        except Exception as err:
            self._cameras = []
            self._status_var.set(f"Camera probe failed: {err}")

        if not self._cameras:
            self._camera_dropdown.configure(values=["No cameras detected"])
            self._camera_var.set("No cameras detected")
            self._status_var.set("No cameras detected. Plug one in and click ↻.")
            return

        labels = [c.label for c in self._cameras]
        self._camera_dropdown.configure(values=labels)
        self._camera_var.set(labels[0])
        self._status_var.set(f"{len(self._cameras)} camera(s) detected.")

    def _refresh_status(self) -> None:
        cfg = config.load()
        if not cfg.is_complete:
            self._status_var.set("⚠ Run `swap setup` first to save your license + Decart key.")

    def _selected_camera(self) -> CameraDevice | None:
        label = self._camera_var.get()
        for cam in self._cameras:
            if cam.label == label:
                return cam
        return None

    def _selected_model(self) -> str:
        # Labels look like "lucy-2  (1280×720, 20 fps)" — take the first token.
        v = self._tier_var.get()
        return v.split()[0] if v else "lucy-2"

    def _on_live(self) -> None:
        print("[gui] live clicked", flush=True)
        if self._session_thread is not None and self._session_thread.is_alive():
            print("[gui] bail: session already live", flush=True)
            self._status_var.set("Already live.")
            return

        cfg = config.load()
        if not cfg.is_complete:
            print("[gui] bail: config incomplete", flush=True)
            self._status_var.set("⚠ Run `swap setup` in a terminal first.")
            return
        if not self._reference_path:
            print("[gui] bail: no reference face loaded", flush=True)
            self._status_var.set("Pick a reference face first.")
            return
        camera = self._selected_camera()
        if camera is None:
            print("[gui] bail: no camera selected", flush=True)
            self._status_var.set("No camera selected.")
            return
        print(f"[gui] starting session: face={self._reference_path}, camera={camera.index}", flush=True)

        record_path: Path | None = None
        if self._record_var.get():
            from .display import default_recording_path

            record_path = default_recording_path()

        def _emit_status(msg: str) -> None:
            # Worker thread → tk main thread. Bind msg via default arg.
            self.after(0, lambda m=msg: self._status_var.set(m))

        def _capture_stop(stop_fn: Callable[[], None]) -> None:
            self._stop_session = stop_fn

        # Voice opts: only set when toggle is on AND a library/user voice is
        # picked. We resolve the mic + virtual cable here using
        # voice_router's auto-detect so the user doesn't have to pick from
        # a sounddevice-numbered list — config remembers their last choice
        # if they had one. None on any of these = video-only path.
        voice_id = self._selected_voice_id() if cfg.voice_enabled else None
        mic_device: int | None = None
        out_device: int | None = None
        if voice_id:
            from . import voice_router

            mic = voice_router.pick_input_device(cfg.last_microphone)
            mic_device = int(mic["index"]) if mic else 0  # fall back to default
            out = voice_router.pick_output_device(cfg.last_voice_output)
            out_device = int(out["index"]) if out else None
            if out_device is None:
                _emit_status(
                    "Voice on, but no virtual audio cable detected — "
                    "converted audio will be silent. Install BlackHole / VB-Cable."
                )
            # Persist for next launch.
            from . import config as _config

            _config.update(last_microphone=mic_device, last_voice_output=out_device)

        opts = RunOptions(
            decart_api_key=cfg.decart_api_key or "",
            reference=str(self._reference_path),
            prompt=self._prompt_box.get("1.0", "end-1c").strip() or DEFAULT_PROMPT,
            model_name=self._selected_model(),
            camera_device=camera.index,
            record=record_path,
            on_status_change=_emit_status,
            on_runtime_ready=_capture_stop,
            reference_voice=voice_id,
            microphone_device=mic_device,
            voice_output_device=out_device,
        )

        # Persist the voice id so next launch defaults to it.
        if voice_id and voice_id != cfg.last_voice_id:
            config.update(last_voice_id=voice_id)

        self._stop_event = asyncio.Event()
        self._set_running(True)
        self._status_var.set("Connecting…")

        def worker() -> None:
            print("[gui] worker thread started", flush=True)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._session_loop = loop
            try:
                loop.run_until_complete(self._supervised_run(opts))
            except BaseException as e:
                import traceback
                traceback.print_exc()
                print(f"[gui] worker died: {e}", flush=True)
            finally:
                print("[gui] worker thread exiting", flush=True)
                loop.close()
                self._session_loop = None
                self._stop_session = None
                self.after(0, lambda: self._set_running(False))
                self.after(0, lambda: self._status_var.set("Session ended."))

        self._session_thread = threading.Thread(target=worker, daemon=True)
        self._session_thread.start()

        # Best-effort license check (non-blocking).
        threading.Thread(target=self._check_license_async, daemon=True).start()

    async def _supervised_run(self, opts: RunOptions) -> None:
        print("[gui] _supervised_run entered", flush=True)
        try:
            await run_session(opts)
        except Exception as err:
            import traceback
            traceback.print_exc()
            msg = str(err) or err.__class__.__name__
            print(f"[gui] caught exception: {msg}", flush=True)
            self.after(0, lambda m=msg: self._status_var.set(f"Error: {m}"))
        else:
            print("[gui] run_session returned cleanly", flush=True)

    def _check_license_async(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                status = loop.run_until_complete(license.validate())
            finally:
                loop.close()
            if not status.valid:
                self.after(
                    0,
                    lambda: self._status_var.set(
                        f"License invalid ({status.reason}) — buy at swap.ikieguy.online"
                    ),
                )
        except Exception:
            # Network down or backend not reachable; offline grace handles it
            pass

    def _on_stop(self) -> None:
        if self._stop_session is None:
            self._status_var.set("Nothing running.")
            return
        print("[gui] stop clicked", flush=True)
        self._status_var.set("Stopping…")
        try:
            self._stop_session()
        except Exception as err:  # noqa: BLE001
            self._status_var.set(f"Stop failed: {err}")
            return
        # Disable Stop immediately so the user can't double-click; the worker's
        # `finally` will reset _set_running(False) when the loop fully unwinds.
        self._stop_btn.configure(state="disabled")
        self._stop_session = None

    def _on_enable_voice(self) -> None:
        """Open the Enable Voice modal: prereq check + guided install."""
        modal = _EnableVoiceModal(self)
        modal.grab_set()  # modal: blocks input on the main window

    def _on_disable_voice(self) -> None:
        """Turn voice off (sticky in config). Doesn't uninstall deps."""
        from . import config as _config

        _config.update(voice_enabled=False)
        self._status_var.set("Voice: disabled.")
        self._refresh_voice_section()

    # ── Standalone voice test (no Decart, zero tokens) ─────────────────

    def _on_test_voice(self) -> None:
        """Toggle: start voice-only test if idle, stop if already running."""
        if self._voice_test_thread is not None and self._voice_test_thread.is_alive():
            self._stop_voice_test()
            return
        self._start_voice_test()

    def _start_voice_test(self) -> None:
        # Don't run while a full session is up — they'd both want the mic.
        if self._session_thread is not None and self._session_thread.is_alive():
            self._status_var.set("Stop the live session before testing voice.")
            return

        voice_id = self._selected_voice_id()
        if not voice_id:
            self._status_var.set("Pick a voice first.")
            return

        from . import voice_library, voice_router

        target = voice_library.find_voice(voice_id)
        if target is None:
            self._status_var.set(f"Voice '{voice_id}' not found.")
            return

        mic = voice_router.pick_input_device(None)
        mic_idx = int(mic["index"]) if mic else 0
        out = voice_router.pick_output_device(None)
        out_idx = int(out["index"]) if out else None

        if out_idx is None:
            cable = voice_router.virtual_cable_hint()
            self._status_var.set(
                f"No virtual audio cable detected. Install {cable.name} "
                "to route the cloned voice to Zoom/Meet/OBS."
            )
            # We still run — gives the user a way to verify the model
            # itself works even without routing.

        def _emit(msg: str) -> None:
            self.after(0, lambda m=msg: self._status_var.set(m))

        cfg_for_engine = config.load()

        async def _runner() -> None:
            from .voice_track import VoiceTrack, VoiceTrackOptions

            track = VoiceTrack(
                VoiceTrackOptions(
                    voice=target,
                    microphone_device=mic_idx,
                    output_device=out_idx,
                    engine_name=cfg_for_engine.voice_engine,
                )
            )

            # Expose a thread-safe stop function back to the tk thread.
            loop = asyncio.get_running_loop()

            def _request_stop() -> None:
                loop.call_soon_threadsafe(stop_evt.set)

            self._voice_test_stop = _request_stop
            stop_evt = asyncio.Event()

            track.start(on_status=_emit)
            try:
                await stop_evt.wait()
            finally:
                await track.stop()

        def _worker() -> None:
            print("[gui] voice test thread started", flush=True)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._voice_test_loop = loop
            try:
                loop.run_until_complete(_runner())
            except Exception as err:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                _emit(f"Voice test failed: {err}")
            finally:
                print("[gui] voice test thread exiting", flush=True)
                loop.close()
                self._voice_test_loop = None
                self._voice_test_stop = None
                self.after(0, self._voice_test_finished)

        self._set_voice_testing(True)
        _emit(f"Voice test: starting ({target.name})…")
        self._voice_test_thread = threading.Thread(target=_worker, daemon=True)
        self._voice_test_thread.start()

    def _stop_voice_test(self) -> None:
        if self._voice_test_stop is not None:
            try:
                self._voice_test_stop()
            except Exception as err:  # noqa: BLE001
                self._status_var.set(f"Voice test stop failed: {err}")
                return
        self._test_voice_btn.configure(state="disabled", text="Stopping…")

    def _voice_test_finished(self) -> None:
        self._set_voice_testing(False)
        self._voice_test_thread = None
        self._status_var.set("Voice test ended.")

    def _set_voice_testing(self, testing: bool) -> None:
        """Disable Live / Disable / dropdown while voice test is running."""
        self._test_voice_btn.configure(
            state="normal",
            text="Stop test" if testing else "Test voice",
            fg_color="#ef4444" if testing else "#0ea5e9",
            hover_color="#dc2626" if testing else "#0284c7",
        )
        self._live_btn.configure(state="disabled" if testing else "normal")
        self._disable_voice_btn.configure(state="disabled" if testing else "normal")
        try:
            self._voice_dropdown.configure(state="disabled" if testing else "readonly")
        except Exception:
            pass

    def _refresh_voice_section(self) -> None:
        """Show collapsed vs expanded voice UI based on config.voice_enabled."""
        from . import config as _config

        cfg = _config.load()
        if cfg.voice_enabled:
            # Hide collapsed row, show expanded with library dropdown.
            try:
                self._voice_collapsed_row.pack_forget()
            except Exception:
                pass
            if not self._voice_expanded_row.winfo_ismapped():
                self._voice_expanded_row.pack(fill="x", pady=(2, 0))
            self._populate_voice_library(cfg.last_voice_id)
        else:
            try:
                self._voice_expanded_row.pack_forget()
            except Exception:
                pass
            if not self._voice_collapsed_row.winfo_ismapped():
                self._voice_collapsed_row.pack(fill="x")

    def _populate_voice_library(self, preferred_id: str | None) -> None:
        from . import voice_library

        voices = voice_library.load_all_voices()
        if not voices:
            self._voice_dropdown.configure(values=["(no voices found)"])
            self._voice_var.set("(no voices found)")
            return

        labels = [self._format_voice_label(v) for v in voices]
        self._voice_dropdown.configure(values=labels)
        self._voice_label_to_id = {
            self._format_voice_label(v): v.id for v in voices
        }

        # Restore previously selected voice if still present, else default to first.
        chosen_label = labels[0]
        if preferred_id:
            for v, label in zip(voices, labels, strict=False):
                if v.id == preferred_id:
                    chosen_label = label
                    break
        self._voice_var.set(chosen_label)

    @staticmethod
    def _format_voice_label(voice) -> str:  # type: ignore[no-untyped-def]
        suffix = "library" if voice.is_library else "custom"
        return f"{voice.name} — {voice.description}  [{suffix}]"

    def _selected_voice_id(self) -> str | None:
        label_to_id = getattr(self, "_voice_label_to_id", {})
        return label_to_id.get(self._voice_var.get())

    def _set_running(self, running: bool) -> None:
        self._live_btn.configure(state="disabled" if running else "normal")
        self._stop_btn.configure(state="normal" if running else "disabled")
        self._select_face_btn.configure(state="disabled" if running else "normal")
        self._camera_dropdown.configure(state="disabled" if running else "readonly")
        self._tier_dropdown.configure(state="disabled" if running else "readonly")
        self._enable_voice_btn.configure(state="disabled" if running else "normal")
        try:
            self._voice_dropdown.configure(state="disabled" if running else "readonly")
            self._disable_voice_btn.configure(state="disabled" if running else "normal")
            # Disable Test voice during a full live session — both grab the mic.
            self._test_voice_btn.configure(state="disabled" if running else "normal")
        except Exception:
            pass

    def _raise_to_front(self) -> None:
        """Flash the window topmost for one beat so it's visible at startup."""
        try:
            self.lift()
            self.attributes("-topmost", True)
            self.after(200, lambda: self.attributes("-topmost", False))
            self.focus_force()
        except Exception:  # noqa: BLE001 — non-fatal cosmetic
            pass


class _EnableVoiceModal(ctk.CTkToplevel):
    """Voice setup wizard. Shows prereq checks inline, lets user run install.

    On success, sets config.voice_enabled = True and refreshes the parent
    GUI's voice section so the library dropdown appears.
    """

    def __init__(self, parent: SwapGUI) -> None:
        super().__init__(parent)
        self._parent = parent
        self.title("Enable voice cloning")
        self.geometry("520x440")
        self.resizable(False, False)
        # Center over parent.
        self.after(0, self._center_on_parent)

        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=20, pady=18)

        ctk.CTkLabel(
            outer,
            text="Voice cloning prerequisites",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 6))

        self._checks_frame = ctk.CTkFrame(outer, fg_color="#1f2937", corner_radius=8)
        self._checks_frame.pack(fill="x", pady=(0, 10))
        self._render_prereqs()

        ctk.CTkLabel(
            outer,
            text=(
                "Voice features add ~3 GB of dependencies plus an OpenVoice\n"
                "model checkpoint. Audio routing into Zoom/OBS needs a virtual\n"
                "audio cable (BlackHole on macOS, VB-Cable on Windows)."
            ),
            anchor="w",
            justify="left",
            text_color="#9ca3af",
        ).pack(fill="x", pady=(0, 12))

        self._status_var = tk.StringVar(value="")
        ctk.CTkLabel(
            outer,
            textvariable=self._status_var,
            anchor="w",
            text_color="#10b981",
        ).pack(fill="x")

        # Action buttons
        actions = ctk.CTkFrame(outer, fg_color="transparent")
        actions.pack(fill="x", side="bottom", pady=(12, 0))
        actions.columnconfigure((0, 1), weight=1, uniform="ev")

        self._skip_btn = ctk.CTkButton(
            actions,
            text="Skip — keep video only",
            command=self._on_skip,
            height=36,
            fg_color="#374151",
            hover_color="#4b5563",
        )
        self._skip_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self._continue_btn = ctk.CTkButton(
            actions,
            text="Continue",
            command=self._on_continue,
            height=36,
            fg_color="#ec4899",
            hover_color="#db2777",
        )
        self._continue_btn.grid(row=0, column=1, sticky="ew")

    def _render_prereqs(self) -> None:
        # Wipe and re-render — called after each step so the UI tracks state.
        for child in self._checks_frame.winfo_children():
            child.destroy()

        from . import voice_prereq

        result = voice_prereq.check_all()
        self._latest_result = result

        rows = [
            ("GPU", result.gpu),
            ("Voice deps (torch, sounddevice, librosa)", result.deps_installed),
            ("OpenVoice weights", result.weights),
            ("Virtual audio cable", result.audio_cable),
        ]
        for title, check in rows:
            row = ctk.CTkFrame(self._checks_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=4)
            ctk.CTkLabel(
                row,
                text="✓" if check.ok else "✗",
                width=20,
                text_color="#10b981" if check.ok else "#ef4444",
                font=ctk.CTkFont(size=14, weight="bold"),
            ).pack(side="left")
            label_text = f"{title}: {check.label}"
            if check.hint and not check.ok:
                label_text += f"  → {check.hint}"
            ctk.CTkLabel(
                row,
                text=label_text,
                anchor="w",
                justify="left",
                wraplength=420,
            ).pack(side="left", fill="x", expand=True)

        # If GPU is blocked, replace the Continue button with a Got-it close.
        if hasattr(self, "_continue_btn") and result.gpu_blocked:
            self._continue_btn.configure(
                text="Got it — keep video only",
                command=self._on_skip,
            )

    def _center_on_parent(self) -> None:
        try:
            px = self._parent.winfo_x()
            py = self._parent.winfo_y()
            pw = self._parent.winfo_width()
            ph = self._parent.winfo_height()
            ww = 520
            wh = 440
            x = px + (pw - ww) // 2
            y = py + (ph - wh) // 2
            self.geometry(f"{ww}x{wh}+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

    def _on_skip(self) -> None:
        self.destroy()

    def _on_continue(self) -> None:
        # Disable buttons during install.
        self._skip_btn.configure(state="disabled")
        self._continue_btn.configure(state="disabled", text="Installing…")
        self._status_var.set("Installing voice deps + downloading weights …")
        self.update_idletasks()
        # Run the install in a worker thread so the UI stays responsive.
        threading.Thread(target=self._install_worker, daemon=True).start()

    def _install_worker(self) -> None:
        from . import config as _config
        from . import voice_ops, voice_prereq

        try:
            pre = voice_prereq.check_all()
            if not pre.deps_installed.ok:
                ok = voice_ops.install_voice_deps()
                if not ok:
                    self._on_install_error("pip install failed")
                    return
            if not pre.weights.ok:
                voice_ops.download_openvoice_weights()
            _config.update(voice_enabled=True)
            self.after(0, self._on_install_done)
        except Exception as err:  # noqa: BLE001
            self.after(0, lambda e=err: self._on_install_error(str(e)))

    def _on_install_done(self) -> None:
        self._status_var.set("✓ Voice features ready.")
        self._render_prereqs()
        self._skip_btn.configure(state="normal", text="Close")
        self._continue_btn.configure(text="Done", command=self._finish_and_close)
        self._continue_btn.configure(state="normal")

    def _on_install_error(self, msg: str) -> None:
        self._status_var.set(f"✗ {msg}")
        self._skip_btn.configure(state="normal", text="Close")
        self._continue_btn.configure(state="normal", text="Retry", command=self._on_continue)

    def _finish_and_close(self) -> None:
        self._parent._refresh_voice_section()
        self._parent._status_var.set("Voice: enabled. Pick a voice to use.")
        self.destroy()


class VoiceOnlyGUI(ctk.CTk):
    """Stripped single-purpose window — voice cloning only, no face/camera/Decart.

    Launched via `swap gui --voice`. Reuses VoiceTrack, voice_library,
    voice_router. License + Decart-key checks are skipped: voice runs
    locally and we trust the swap-cli install itself.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title(f"swap voice {__version__}")
        W, H = 460, 460
        self.minsize(420, 420)
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{W}x{H}+{max(0, (sw - W) // 2)}+{max(0, (sh - H) // 2)}")
        self.after(0, self._raise_to_front)

        self._track = None  # voice_track.VoiceTrack | None
        self._track_thread: threading.Thread | None = None
        self._track_loop: asyncio.AbstractEventLoop | None = None
        self._track_stop: Callable[[], None] | None = None
        self._status_var = tk.StringVar(value="Idle.")

        self._build_ui()
        self._refresh_devices()

    # ── UI build ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=20, pady=18)

        ctk.CTkLabel(
            outer,
            text="Voice clone · live",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            outer,
            text=(
                "Local-only voice transformation for calls. No video, no "
                "Decart, zero token cost."
            ),
            anchor="w",
            justify="left",
            text_color="#6b7280",
            wraplength=400,
        ).pack(fill="x", pady=(0, 14))

        # ① Reference voice
        ctk.CTkLabel(
            outer, text="① Reference voice", anchor="w", font=ctk.CTkFont(size=11)
        ).pack(anchor="w")
        self._voice_var = tk.StringVar(value="(no voices found)")
        self._voice_dropdown = ctk.CTkComboBox(
            outer, values=[], variable=self._voice_var, state="readonly", height=32
        )
        self._voice_dropdown.pack(fill="x", pady=(2, 14))

        # ② Microphone
        ctk.CTkLabel(
            outer, text="② Microphone", anchor="w", font=ctk.CTkFont(size=11)
        ).pack(anchor="w")
        self._mic_var = tk.StringVar(value="(no mic found)")
        self._mic_dropdown = ctk.CTkComboBox(
            outer, values=[], variable=self._mic_var, state="readonly", height=32
        )
        self._mic_dropdown.pack(fill="x", pady=(2, 14))

        # ③ Output (virtual cable)
        out_row = ctk.CTkFrame(outer, fg_color="transparent")
        out_row.pack(fill="x")
        out_row.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            out_row, text="③ Output (virtual cable)", anchor="w", font=ctk.CTkFont(size=11)
        ).grid(row=0, column=0, sticky="w", columnspan=2)
        self._output_var = tk.StringVar(value="(no output found)")
        self._output_dropdown = ctk.CTkComboBox(
            out_row, values=[], variable=self._output_var, state="readonly", height=32
        )
        self._output_dropdown.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(
            out_row, text="↻", width=42, height=32, command=self._refresh_devices
        ).grid(row=1, column=1)

        # Action row
        actions = ctk.CTkFrame(outer, fg_color="transparent")
        actions.pack(fill="x", pady=(20, 0))
        actions.columnconfigure((0, 1), weight=1, uniform="act")

        self._start_btn = ctk.CTkButton(
            actions,
            text="▶ Start",
            command=self._on_start,
            height=44,
            corner_radius=8,
            fg_color="#10b981",
            hover_color="#059669",
        )
        self._start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self._stop_btn = ctk.CTkButton(
            actions,
            text="Stop",
            command=self._on_stop,
            height=44,
            corner_radius=8,
            fg_color="#374151",
            hover_color="#4b5563",
            state="disabled",
        )
        self._stop_btn.grid(row=0, column=1, sticky="ew")

        # Status bar
        status = ctk.CTkFrame(outer, fg_color="transparent")
        status.pack(fill="x", pady=(18, 0))
        ctk.CTkLabel(
            status,
            textvariable=self._status_var,
            anchor="w",
            text_color="#9ca3af",
        ).pack(side="left", fill="x", expand=True)

    # ── Device discovery ──────────────────────────────────────────────

    def _refresh_devices(self) -> None:
        from . import voice_library, voice_router

        # Voices
        voices = voice_library.load_all_voices()
        if voices:
            self._voice_label_to_id = {self._fmt_voice(v): v.id for v in voices}
            labels = list(self._voice_label_to_id.keys())
            self._voice_dropdown.configure(values=labels)
            self._voice_var.set(labels[0])
        else:
            self._voice_dropdown.configure(values=["(no voices found)"])
            self._voice_var.set("(no voices found)")

        # Mics + outputs (require sounddevice)
        inputs, outputs = voice_router.list_audio_devices()

        if inputs:
            self._mic_label_to_idx = {
                f"{d['name']} (#{d['index']})": int(d["index"]) for d in inputs
            }
            self._mic_dropdown.configure(values=list(self._mic_label_to_idx.keys()))
            self._mic_var.set(next(iter(self._mic_label_to_idx)))
        else:
            self._mic_dropdown.configure(values=["(no mic found)"])
            self._mic_var.set("(no mic found)")

        if outputs:
            self._out_label_to_idx = {
                f"{d['name']} (#{d['index']})": int(d["index"]) for d in outputs
            }
            self._output_dropdown.configure(values=list(self._out_label_to_idx.keys()))
            # Prefer auto-detected virtual cable.
            cable = voice_router.detect_virtual_cable_in_devices(outputs)
            if cable is not None:
                self._output_var.set(f"{cable['name']} (#{cable['index']})")
            else:
                self._output_var.set(next(iter(self._out_label_to_idx)))
        else:
            self._output_dropdown.configure(values=["(no output found)"])
            self._output_var.set("(no output found)")

    @staticmethod
    def _fmt_voice(voice) -> str:  # type: ignore[no-untyped-def]
        suffix = "library" if voice.is_library else "custom"
        return f"{voice.name} — {voice.description}  [{suffix}]"

    def _selected_voice_id(self) -> str | None:
        return getattr(self, "_voice_label_to_id", {}).get(self._voice_var.get())

    def _selected_mic(self) -> int | None:
        return getattr(self, "_mic_label_to_idx", {}).get(self._mic_var.get())

    def _selected_output(self) -> int | None:
        return getattr(self, "_out_label_to_idx", {}).get(self._output_var.get())

    # ── Start / stop ─────────────────────────────────────────────────

    def _on_start(self) -> None:
        if self._track_thread is not None and self._track_thread.is_alive():
            self._status_var.set("Already running.")
            return
        voice_id = self._selected_voice_id()
        mic = self._selected_mic()
        output = self._selected_output()

        if not voice_id:
            self._status_var.set("Pick a voice first.")
            return
        if mic is None:
            self._status_var.set("Pick a microphone first.")
            return

        from . import voice_library

        target = voice_library.find_voice(voice_id)
        if target is None:
            self._status_var.set(f"Voice '{voice_id}' not found.")
            return

        def _emit(msg: str) -> None:
            self.after(0, lambda m=msg: self._status_var.set(m))

        from . import config as _cfg_mod

        cfg_for_engine = _cfg_mod.load()

        async def _runner() -> None:
            from .voice_track import VoiceTrack, VoiceTrackOptions

            stop_evt = asyncio.Event()
            loop = asyncio.get_running_loop()

            def _request_stop() -> None:
                loop.call_soon_threadsafe(stop_evt.set)

            self._track_stop = _request_stop

            track = VoiceTrack(
                VoiceTrackOptions(
                    voice=target,
                    microphone_device=mic,
                    output_device=output,
                    engine_name=cfg_for_engine.voice_engine,
                )
            )
            track.start(on_status=_emit)
            try:
                await stop_evt.wait()
            finally:
                await track.stop()

        def _worker() -> None:
            print("[voice-gui] worker started", flush=True)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._track_loop = loop
            try:
                loop.run_until_complete(_runner())
            except Exception as err:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                _emit(f"Failed: {err}")
            finally:
                print("[voice-gui] worker exiting", flush=True)
                loop.close()
                self._track_loop = None
                self._track_stop = None
                self.after(0, self._on_stopped)

        self._set_running(True)
        _emit(f"Starting ({target.name})…")
        self._track_thread = threading.Thread(target=_worker, daemon=True)
        self._track_thread.start()

    def _on_stop(self) -> None:
        if self._track_stop is not None:
            try:
                self._track_stop()
            except Exception as err:  # noqa: BLE001
                self._status_var.set(f"Stop failed: {err}")
                return
        self._stop_btn.configure(state="disabled", text="Stopping…")

    def _on_stopped(self) -> None:
        self._set_running(False)
        self._track_thread = None
        self._status_var.set("Stopped.")

    def _set_running(self, running: bool) -> None:
        self._start_btn.configure(state="disabled" if running else "normal")
        self._stop_btn.configure(state="normal" if running else "disabled", text="Stop")
        self._voice_dropdown.configure(state="disabled" if running else "readonly")
        self._mic_dropdown.configure(state="disabled" if running else "readonly")
        self._output_dropdown.configure(state="disabled" if running else "readonly")

    def _raise_to_front(self) -> None:
        try:
            self.lift()
            self.attributes("-topmost", True)
            self.after(200, lambda: self.attributes("-topmost", False))
            self.focus_force()
        except Exception:
            pass


def launch(voice_only: bool = False) -> None:
    """Entrypoint used by `swap gui`. With voice_only=True, opens a
    stripped single-purpose window for live voice cloning only — no
    face, no camera, no Decart connection.
    """
    print(f"[gui] starting swap-cli GUI (voice_only={voice_only})", flush=True)
    try:
        app: ctk.CTk = VoiceOnlyGUI() if voice_only else SwapGUI()
    except Exception:
        import traceback

        print("[gui] failed to construct window:", flush=True)
        traceback.print_exc()
        # Pause so the user can read the trace before the cmd window closes
        # if they ran via a desktop shortcut instead of a terminal.
        try:
            input("\nPress Enter to exit…")
        except EOFError:
            pass
        raise
    print("[gui] entering mainloop", flush=True)
    try:
        app.mainloop()
    except Exception:
        import traceback

        print("[gui] mainloop crashed:", flush=True)
        traceback.print_exc()
        raise
    print("[gui] mainloop returned", flush=True)
