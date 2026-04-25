"""customtkinter GUI for swap-cli.

Layout mirrors Deep-Live-Cam: face thumbnail, camera dropdown, options,
Start / Destroy / Preview / Live buttons. The Live button opens the
realtime stream in a separate window via the existing display.py.
"""

from __future__ import annotations

import asyncio
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING

import customtkinter as ctk
from PIL import Image

from . import config, license
from .devices import CameraDevice, enumerate_cameras
from .runtime import DEFAULT_PROMPT, RunOptions, run_session
from .version import __version__

if TYPE_CHECKING:
    pass

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

THUMB_SIZE = (140, 140)


class SwapGUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"swap-cli {__version__} · live deepfake")
        self.geometry("520x720")
        self.minsize(480, 660)

        self._reference_path: Path | None = None
        self._thumb_image: ctk.CTkImage | None = None
        self._cameras: list[CameraDevice] = []
        self._session_thread: threading.Thread | None = None
        self._session_loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._status_var = tk.StringVar(value="Idle.")

        self._build_ui()
        self._refresh_cameras()
        self._refresh_status()

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

        opts = RunOptions(
            decart_api_key=cfg.decart_api_key or "",
            reference=str(self._reference_path),
            prompt=self._prompt_box.get("1.0", "end-1c").strip() or DEFAULT_PROMPT,
            model_name=self._selected_model(),
            camera_device=camera.index,
            record=record_path,
            on_status_change=_emit_status,
        )

        self._stop_event = asyncio.Event()
        self._set_running(True)
        self._status_var.set("Connecting…")

        def worker() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._session_loop = loop
            try:
                loop.run_until_complete(self._supervised_run(opts))
            finally:
                loop.close()
                self._session_loop = None
                self.after(0, lambda: self._set_running(False))
                self.after(0, lambda: self._status_var.set("Session ended."))

        self._session_thread = threading.Thread(target=worker, daemon=True)
        self._session_thread.start()

        # Best-effort license check (non-blocking).
        threading.Thread(target=self._check_license_async, daemon=True).start()

    async def _supervised_run(self, opts: RunOptions) -> None:
        try:
            await run_session(opts)
        except Exception as err:
            msg = str(err) or err.__class__.__name__
            self.after(0, lambda m=msg: self._status_var.set(f"Error: {m}"))

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
        # The runtime listens on the user pressing Q in the preview window.
        # We can't programmatically signal it from here without a deeper hook,
        # so this is a placeholder until we wire a stop event into runtime.
        self._status_var.set("Press Q in the preview window to stop the session.")

    def _set_running(self, running: bool) -> None:
        self._live_btn.configure(state="disabled" if running else "normal")
        self._stop_btn.configure(state="normal" if running else "disabled")
        self._select_face_btn.configure(state="disabled" if running else "normal")
        self._camera_dropdown.configure(state="disabled" if running else "readonly")
        self._tier_dropdown.configure(state="disabled" if running else "readonly")


def launch() -> None:
    """Entrypoint used by `swap gui`."""
    app = SwapGUI()
    app.mainloop()
