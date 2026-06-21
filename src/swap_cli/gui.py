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
from contextlib import suppress
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING, Any, Callable

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


# Sprint 14l: settings panel helpers.
# Pure functions so they're trivially unit-testable without spinning a tk root.


def _redact_key(value: str | None) -> str:
    """Show last 4 chars only: 'dct_a…AB12' or '—' if absent."""
    if not value:
        return "—"
    if len(value) <= 4:
        return value
    return f"{value[:4]}…{value[-4:]}"


class DecartKeyValidationError(ValueError):
    """Raised when a candidate Decart API key fails basic format checks."""


def apply_decart_key_update(new_key: str) -> None:
    """Validate + persist a new Decart API key.

    Loose validation: must start with 'dct_' and be at least 20 chars.
    Decart's exact format isn't documented and changes over time, so we
    only catch obvious typos (empty, "dct_short", pasted gibberish) here.
    Real validation happens at session-open time when the SDK rejects
    the key.

    Side effects on success:
      - config.update(decart_api_key=<new>, license_cached_at=None,
        license_cached_valid_until=None) — the cache reset forces the
        next session to re-validate the license against the server.
    """
    trimmed = (new_key or "").strip()
    if not trimmed.startswith("dct_"):
        raise DecartKeyValidationError("Decart key must start with 'dct_'.")
    if len(trimmed) < 20:
        raise DecartKeyValidationError(
            f"Decart key looks too short ({len(trimmed)} chars; expect ≥ 20)."
        )
    config.update(
        decart_api_key=trimmed,
        license_cached_at=None,
        license_cached_valid_until=None,
    )


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
        # macOS only: the GUI pumps the cv2 preview window from the MAIN thread
        # (macOS forbids cv2 windows off the main thread, and the session runs on
        # a worker thread). _display is handed over via on_display_ready.
        self._display: Any = None
        self._mac_preview_after_id: str | None = None
        self._mac_window_open = False
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
        # Pinned bottom bar (Live/Stop + status + footer) — packed first with
        # side="bottom" so it's always reachable even when the content above
        # overflows. The scrollable area then takes the remaining space.
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=20, pady=(0, 12))
        self._bottom_bar = bottom

        # Scrollable content area — everything above the action bar lives here
        # so smaller windows can scroll instead of clipping widgets.
        outer = ctk.CTkScrollableFrame(self, fg_color="transparent")
        outer.pack(side="top", fill="both", expand=True, padx=14, pady=(14, 0))

        # Settings (gear) button — top-right, opens the settings modal.
        title_bar = ctk.CTkFrame(outer, fg_color="transparent")
        title_bar.pack(fill="x", pady=(0, 4))
        ctk.CTkButton(
            title_bar,
            text="⚙",
            width=36,
            height=36,
            corner_radius=18,
            command=self._on_settings_clicked,
            fg_color="#374151",
            hover_color="#4b5563",
            font=ctk.CTkFont(size=16),
        ).pack(side="right")

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

        # Sprint 14k: virtual camera output. When on, Zoom/Meet/Discord
        # see the deepfake as a camera device ("OBS Virtual Camera") —
        # no OBS app to open, no manual switching. Default on; user can
        # disable for screen-recording / preview-only sessions.
        self._vcam_var = tk.BooleanVar(value=True)
        ctk.CTkSwitch(
            opts,
            text="Output to virtual camera",
            variable=self._vcam_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=4)

        # Sprint 15: per-frame Decart watermark removal. Defaults ON — the
        # app ships a bundled default template so removal works out of the
        # box; clients get a clean feed without any capture step. The toggle
        # is read when Live is clicked (start-time); flip it off here before
        # going Live if you ever want the raw feed.
        self._watermark_var = tk.BooleanVar(value=True)
        ctk.CTkSwitch(
            opts,
            text="Remove watermark",
            variable=self._watermark_var,
            command=self._sync_watermark_options,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)

        # Removal style (only used when "Remove watermark" is on, so the whole
        # row is hidden while removal is off). Reconstruct rebuilds the real
        # background behind the roaming badge (invisible, best quality); Blur
        # just smears the badge into an unreadable soft patch (always works, no
        # reconstruction artifacts, but a visible soft blob). Seeded from the
        # saved preference; read at start-time like the toggle.
        _wm_cfg = config.load()
        self._watermark_removal_row = ctk.CTkFrame(opts, fg_color="transparent")
        removal_row = self._watermark_removal_row
        removal_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 4))
        ctk.CTkLabel(
            removal_row, text="Removal style", anchor="w", font=ctk.CTkFont(size=11)
        ).pack(anchor="w")
        self._watermark_removal_var = tk.StringVar(
            value=(_wm_cfg.watermark_removal or "reconstruct").capitalize()
        )
        ctk.CTkSegmentedButton(
            removal_row,
            values=["Reconstruct", "Blur"],
            variable=self._watermark_removal_var,
        ).pack(anchor="w", fill="x", pady=(2, 0))
        ctk.CTkLabel(
            removal_row,
            text="Reconstruct: invisible, best quality. "
            "Blur: smears the badge — always works, leaves a soft patch.",
            anchor="w",
            font=ctk.CTkFont(size=10),
            text_color="#6b7280",
            wraplength=460,
            justify="left",
        ).pack(anchor="w", pady=(2, 0))
        self._sync_watermark_options()  # hide the picker if removal starts off

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

        # Action buttons row — lives in the pinned bottom bar so Live/Stop
        # stay visible regardless of scroll position.
        actions = ctk.CTkFrame(bottom, fg_color="transparent")
        actions.pack(fill="x", pady=(8, 0))
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

        # Status bar (pinned)
        status = ctk.CTkFrame(bottom, fg_color="transparent")
        status.pack(fill="x", pady=(10, 0))
        ctk.CTkLabel(
            status,
            textvariable=self._status_var,
            anchor="w",
            text_color="#9ca3af",
            font=ctk.CTkFont(size=11),
        ).pack(fill="x")

        # Footer (pinned)
        ctk.CTkLabel(
            bottom,
            text="swap-cli — Press Q in the preview window to stop",
            font=ctk.CTkFont(size=10),
            text_color="#6b7280",
        ).pack(pady=(8, 0))

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

    def _on_settings_clicked(self) -> None:
        """Spawn the settings modal. Reuses the existing one if open."""
        try:
            if getattr(self, "_settings_modal", None) is not None and self._settings_modal.winfo_exists():
                self._settings_modal.deiconify()
                self._settings_modal.focus_force()
                return
        except Exception:  # noqa: BLE001
            pass
        self._settings_modal = _SettingsModal(self)

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

        # Sprint 14o: prefix virtual cameras with ⚠ in the dropdown so
        # users see the feedback-loop warning at a glance. Then auto-
        # select the first REAL (non-virtual) camera; fall back to the
        # first device of any kind if only virtuals are present.
        def _label_for_display(cam: CameraDevice) -> str:
            return f"⚠ {cam.label}" if cam.virtual else cam.label

        labels = [_label_for_display(c) for c in self._cameras]
        self._camera_dropdown.configure(values=labels)

        real_idx = next(
            (i for i, c in enumerate(self._cameras) if not c.virtual),
            None,
        )
        if real_idx is not None:
            self._camera_var.set(labels[real_idx])
            virtual_count = sum(1 for c in self._cameras if c.virtual)
            note = (
                f" ({virtual_count} virtual hidden from default)"
                if virtual_count
                else ""
            )
            self._status_var.set(
                f"{len(self._cameras)} camera(s) detected{note}."
            )
        else:
            # Only virtuals available — picking one would create the
            # feedback loop if vcam output is on. Warn the user.
            self._camera_var.set(labels[0])
            self._status_var.set(
                "⚠ Only virtual cameras detected. Close any app holding "
                "your real webcam (Zoom/Teams/Discord) and click ↻."
            )

    def _refresh_status(self) -> None:
        cfg = config.load()
        if not cfg.is_complete:
            self._status_var.set("⚠ Run `swap setup` first to save your license + Decart key.")

    def _selected_camera(self) -> CameraDevice | None:
        label = self._camera_var.get()
        # Sprint 14o: dropdown labels for virtual cameras are prefixed
        # with "⚠ " — strip that before matching against the original
        # CameraDevice.label.
        if label.startswith("⚠ "):
            label = label[2:]
        for cam in self._cameras:
            if cam.label == label:
                return cam
        return None

    def _selected_model(self) -> str:
        # Labels look like "lucy-2  (1280×720, 20 fps)" — take the first token.
        v = self._tier_var.get()
        return v.split()[0] if v else "lucy-2"

    def _sync_watermark_options(self) -> None:
        """Show the removal-style picker only when watermark removal is on —
        there's nothing to configure when it's off."""
        row = getattr(self, "_watermark_removal_row", None)
        if row is None:
            return
        if bool(self._watermark_var.get()):
            row.grid()  # restores prior grid options
        else:
            row.grid_remove()

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

        # Sprint 14o: refuse the feedback-loop combo. If the user has
        # picked a virtual camera as input AND has vcam output enabled,
        # swap would be reading from the same device it writes to —
        # Lucy would consume its own previous frames, producing the
        # "very weird" recursive transform the user reported.
        vcam_on = bool(self._vcam_var.get()) if hasattr(self, "_vcam_var") else True
        if camera.virtual and vcam_on:
            print(
                f"[gui] bail: feedback loop — camera={camera.label} + vcam on",
                flush=True,
            )
            self._status_var.set(
                "⚠ Feedback loop: virtual camera as input + virtual camera "
                "output. Pick a real webcam, or toggle off 'Output to "
                "virtual camera' in options."
            )
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

        def _on_display_ready(disp: Any) -> None:
            # Worker thread → just store the ref; the main-thread mac preview
            # loop reads disp.latest_frame() to pump the cv2 window.
            self._display = disp

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
            # macOS: the GUI pumps the cv2 window from the main thread (cv2 can't
            # open a window off the main thread on macOS). Windows/Linux keep the
            # original behaviour — the Display owns the window on its own thread.
            show_preview_window=(sys.platform != "darwin"),
            on_display_ready=(_on_display_ready if sys.platform == "darwin" else None),
            reference_voice=voice_id,
            microphone_device=mic_device,
            voice_output_device=out_device,
            virtual_camera=bool(self._vcam_var.get()) if hasattr(self, "_vcam_var") else True,
            remove_watermark=(
                bool(self._watermark_var.get())
                if hasattr(self, "_watermark_var")
                else False
            ),
            watermark_template=cfg.watermark_template,
            watermark_method=cfg.watermark_method,
            watermark_removal=(
                self._watermark_removal_var.get().lower()
                if hasattr(self, "_watermark_removal_var")
                else cfg.watermark_removal
            ),
            watermark_threshold=cfg.watermark_threshold,
            watermark_inpaint_radius=cfg.watermark_inpaint_radius,
            watermark_template_width=cfg.watermark_template_width,
            watermark_signature_fallback=cfg.watermark_signature_fallback,
        )

        # Persist the voice id so next launch defaults to it.
        if voice_id and voice_id != cfg.last_voice_id:
            config.update(last_voice_id=voice_id)
        # Persist the watermark toggle so it sticks across launches.
        if hasattr(self, "_watermark_var"):
            wm_on = bool(self._watermark_var.get())
            if wm_on != cfg.remove_watermark:
                config.update(remove_watermark=wm_on)
        # Persist the removal style too.
        if hasattr(self, "_watermark_removal_var"):
            wm_removal = self._watermark_removal_var.get().lower()
            if wm_removal != cfg.watermark_removal:
                config.update(watermark_removal=wm_removal)

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

        # macOS: drive the cv2 preview window from the main thread.
        if sys.platform == "darwin":
            self._start_mac_preview()

        # Best-effort license check (non-blocking).
        threading.Thread(target=self._check_license_async, daemon=True).start()

    async def _supervised_run(self, opts: RunOptions) -> None:
        print("[gui] _supervised_run entered", flush=True)
        try:
            await run_session(opts)
        except asyncio.CancelledError:
            # Expected on an abrupt stop / connection drop during teardown —
            # not a real error, so no scary traceback.
            print("[gui] session cancelled during shutdown", flush=True)
            self.after(0, lambda: self._status_var.set("Session ended — connection interrupted."))
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
                        f"License invalid ({status.reason}) — buy at swap.storelygh.com"
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

    # ── macOS: pump the cv2 preview window from the main thread ─────────
    # (macOS forbids cv2 windows off the main thread; the session runs on a
    #  worker thread. The window is the SAME cv2 window as on Windows.)

    def _start_mac_preview(self) -> None:
        self._mac_window_open = False
        self._mac_preview_tick()

    def _mac_preview_tick(self) -> None:
        # Session ended → tear the window down and stop the loop.
        if self._session_thread is None or not self._session_thread.is_alive():
            self._close_mac_preview()
            return
        disp = self._display
        frame = disp.latest_frame() if disp is not None else None
        if frame is not None:
            try:
                import cv2

                from .display import WINDOW_TITLE

                if not self._mac_window_open:
                    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW_TITLE, 960, 540)
                    self._mac_window_open = True
                cv2.imshow(WINDOW_TITLE, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # Q / ESC → stop
                    self._on_stop()
                elif key in (ord("w"), ord("W")):  # W → capture watermark
                    try:
                        disp.capture_watermark(None)  # cv2.selectROI on main thread
                    except Exception as err:
                        self._status_var.set(f"Capture failed: {err}")
            except Exception:  # preview is best-effort; never crash the GUI
                pass
        self._mac_preview_after_id = self.after(33, self._mac_preview_tick)

    def _close_mac_preview(self) -> None:
        if self._mac_preview_after_id is not None:
            with suppress(Exception):
                self.after_cancel(self._mac_preview_after_id)
            self._mac_preview_after_id = None
        if self._mac_window_open:
            with suppress(Exception):
                import cv2

                cv2.destroyAllWindows()
            self._mac_window_open = False
        self._display = None

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
                    fast=cfg_for_engine.voice_fast,
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
        if not running:
            # Session ended (stop / drop / error) → close the macOS preview loop.
            self._close_mac_preview()
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
                "Voice features add ~3 GB of CUDA-matched PyTorch + RVC stack.\n"
                "You'll also need an RVC .pth model (download from weights.gg)\n"
                "and a virtual audio cable (BlackHole on macOS, VB-Cable on Windows)."
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
            ("Voice deps (torch, rvc-python, fairseq)", result.deps_installed),
            ("ffmpeg on PATH", result.ffmpeg),
            ("Visual C++ Build Tools", result.build_tools),
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
                    fast=cfg_for_engine.voice_fast,
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


class _SettingsModal(ctk.CTkToplevel):
    """Settings panel — view license key, rotate Decart API key, open
    the config folder. Sprint 14l.

    All state lives in config.toml on disk. We never cache it in memory
    longer than the modal's lifetime, so save → close → reopen always
    reflects current truth.
    """

    def __init__(self, parent: "SwapGUI") -> None:
        super().__init__(parent)
        self._parent = parent
        self.title("Settings")
        self.geometry("560x420")
        self.resizable(False, False)
        self.after(0, self._center_on_parent)

        cfg = config.load()

        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=20, pady=18)

        ctk.CTkLabel(
            outer,
            text="Settings",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 10))

        # ── License key (read-only) ────────────────────────────────────
        license_frame = ctk.CTkFrame(outer, fg_color="#1f2937", corner_radius=8)
        license_frame.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(
            license_frame,
            text="License key",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#9ca3af",
            anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 0))
        ctk.CTkLabel(
            license_frame,
            text=_redact_key(cfg.license_key),
            font=ctk.CTkFont(size=13),
            anchor="w",
        ).pack(fill="x", padx=14, pady=(2, 4))
        ctk.CTkLabel(
            license_frame,
            text="Bought from swap.storelygh.com — rotate by contacting support.",
            font=ctk.CTkFont(size=10),
            text_color="#6b7280",
            anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 10))

        # ── Decart key (editable) ──────────────────────────────────────
        decart_frame = ctk.CTkFrame(outer, fg_color="#1f2937", corner_radius=8)
        decart_frame.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(
            decart_frame,
            text="Decart API key",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#9ca3af",
            anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 0))

        self._decart_var = tk.StringVar(value=cfg.decart_api_key or "")
        self._editing = False
        self._decart_show = tk.BooleanVar(value=False)
        self._decart_label = ctk.CTkLabel(
            decart_frame,
            text=_redact_key(cfg.decart_api_key),
            font=ctk.CTkFont(size=13),
            anchor="w",
        )
        self._decart_label.pack(fill="x", padx=14, pady=(2, 4))

        self._decart_entry = ctk.CTkEntry(
            decart_frame,
            textvariable=self._decart_var,
            show="•",
            placeholder_text="dct_…",
        )
        # Hidden until Edit is clicked.

        btn_row = ctk.CTkFrame(decart_frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(0, 10))
        self._edit_btn = ctk.CTkButton(
            btn_row, text="✎ Edit", width=80, command=self._on_edit
        )
        self._edit_btn.pack(side="left")
        self._show_btn = ctk.CTkButton(
            btn_row,
            text="👁 Show",
            width=80,
            command=self._on_toggle_show,
            fg_color="#374151",
            hover_color="#4b5563",
        )
        self._show_btn.pack(side="left", padx=(8, 0))
        self._save_btn = ctk.CTkButton(
            btn_row, text="Save", width=80, command=self._on_save
        )
        self._cancel_btn = ctk.CTkButton(
            btn_row,
            text="Cancel",
            width=80,
            command=self._on_cancel,
            fg_color="#374151",
            hover_color="#4b5563",
        )
        # Save/Cancel hidden until Edit is clicked.

        self._error_label = ctk.CTkLabel(
            decart_frame, text="", text_color="#ef4444", anchor="w"
        )
        self._error_label.pack(fill="x", padx=14, pady=(0, 4))

        # ── Config file path ───────────────────────────────────────────
        path_frame = ctk.CTkFrame(outer, fg_color="#1f2937", corner_radius=8)
        path_frame.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(
            path_frame,
            text="Config file",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#9ca3af",
            anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 0))
        cfg_path = config.config_path()
        ctk.CTkLabel(
            path_frame,
            text=str(cfg_path),
            font=ctk.CTkFont(size=11),
            text_color="#d1d5db",
            anchor="w",
            wraplength=480,
            justify="left",
        ).pack(fill="x", padx=14, pady=(2, 4))
        ctk.CTkButton(
            path_frame,
            text="📁 Open folder",
            width=140,
            command=self._on_open_folder,
            fg_color="#374151",
            hover_color="#4b5563",
        ).pack(anchor="w", padx=14, pady=(0, 10))

        # ── Close ──────────────────────────────────────────────────────
        ctk.CTkButton(
            outer,
            text="Close",
            command=self.destroy,
            height=36,
            fg_color="#ec4899",
            hover_color="#db2777",
        ).pack(fill="x", side="bottom")

        self._status_var = tk.StringVar(value="")
        ctk.CTkLabel(
            outer,
            textvariable=self._status_var,
            text_color="#10b981",
            anchor="w",
        ).pack(fill="x", side="bottom", pady=(0, 8))

    def _on_edit(self) -> None:
        self._editing = True
        self._decart_label.pack_forget()
        self._decart_entry.pack(fill="x", padx=14, pady=(0, 4))
        self._edit_btn.pack_forget()
        self._show_btn.pack_forget()
        self._save_btn.pack(side="left")
        self._cancel_btn.pack(side="left", padx=(8, 0))
        self._error_label.configure(text="")

    def _on_cancel(self) -> None:
        # Restore current saved value into the var.
        cfg = config.load()
        self._decart_var.set(cfg.decart_api_key or "")
        self._editing = False
        self._decart_entry.pack_forget()
        self._save_btn.pack_forget()
        self._cancel_btn.pack_forget()
        self._decart_label.configure(text=_redact_key(cfg.decart_api_key))
        self._decart_label.pack(fill="x", padx=14, pady=(2, 4))
        self._edit_btn.pack(side="left")
        self._show_btn.pack(side="left", padx=(8, 0))
        self._error_label.configure(text="")

    def _on_save(self) -> None:
        try:
            apply_decart_key_update(self._decart_var.get())
        except DecartKeyValidationError as err:
            self._error_label.configure(text=str(err))
            return
        # Saved. Collapse the editor + show a toast.
        cfg = config.load()
        self._editing = False
        self._decart_entry.pack_forget()
        self._save_btn.pack_forget()
        self._cancel_btn.pack_forget()
        self._decart_label.configure(text=_redact_key(cfg.decart_api_key))
        self._decart_label.pack(fill="x", padx=14, pady=(2, 4))
        self._edit_btn.pack(side="left")
        self._show_btn.pack(side="left", padx=(8, 0))
        self._status_var.set("✓ Decart key updated. Takes effect next session.")
        self.after(3500, lambda: self._status_var.set(""))
        # Refresh the parent status row so it picks up the change.
        try:
            self._parent._refresh_status()
        except Exception:  # noqa: BLE001 — non-fatal cosmetic
            pass

    def _on_toggle_show(self) -> None:
        self._decart_show.set(not self._decart_show.get())
        cfg = config.load()
        if self._decart_show.get():
            self._decart_label.configure(text=cfg.decart_api_key or "—")
            self._show_btn.configure(text="🙈 Hide")
        else:
            self._decart_label.configure(text=_redact_key(cfg.decart_api_key))
            self._show_btn.configure(text="👁 Show")

    def _on_open_folder(self) -> None:
        import os
        import subprocess

        folder = config.config_path().parent
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as err:  # noqa: BLE001
            self._error_label.configure(text=f"Couldn't open folder: {err}")

    def _center_on_parent(self) -> None:
        try:
            px = self._parent.winfo_x()
            py = self._parent.winfo_y()
            pw = self._parent.winfo_width()
            ph = self._parent.winfo_height()
            ww = 560
            wh = 420
            x = px + (pw - ww) // 2
            y = py + (ph - wh) // 2
            self.geometry(f"{ww}x{wh}+{max(0, x)}+{max(0, y)}")
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
