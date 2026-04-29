"""Audio output routing for voice track.

Picks where the converted voice goes:
- A virtual audio cable device (BlackHole / VB-Cable / pulse-loopback)
  so Zoom/OBS/Discord can pick it up as a microphone source.
- Optionally also playback through default speakers for self-monitoring.

Streaming output itself happens in voice_track._loop. This module hosts
the helpers the GUI/runtime use to pick the right device for the user.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class VirtualCableHint:
    name: str
    install_cmd: str | None  # CLI-friendly install hint
    install_url: str | None  # link for users who'd rather click


def virtual_cable_hint() -> VirtualCableHint:
    """Tell the user how to install a virtual audio cable for their OS."""
    if sys.platform == "darwin":
        return VirtualCableHint(
            name="BlackHole 2ch",
            install_cmd="brew install blackhole-2ch",
            install_url="https://existential.audio/blackhole/",
        )
    if sys.platform == "win32":
        return VirtualCableHint(
            name="VB-Cable",
            install_cmd=None,
            install_url="https://vb-audio.com/Cable/",
        )
    return VirtualCableHint(
        name="PulseAudio loopback",
        install_cmd="pactl load-module module-null-sink sink_name=swap-voice",
        install_url=None,
    )


def detect_virtual_cable_in_devices(output_devices: list[dict]) -> dict | None:
    """Given the list of sounddevice output devices, return the virtual
    cable entry if one is present. Used by the GUI to default the
    'Output' dropdown to the right device.

    output_devices is what `enumerate_audio_outputs()` returns — keep
    this signature dependency-free so unit tests don't need sounddevice.
    """
    candidates_by_platform = {
        "darwin": ("blackhole",),
        "win32": ("vb-cable", "cable input", "vb cable", "cable output"),
        "linux": ("swap-voice", "null sink", "pulse loopback"),
    }
    needles = candidates_by_platform.get(sys.platform, ())
    if not needles:
        return None
    for dev in output_devices:
        name = str(dev.get("name", "")).lower()
        if any(n in name for n in needles):
            return dev
    return None


def list_audio_devices() -> tuple[list[dict], list[dict]]:
    """Return (input_devices, output_devices) by querying sounddevice.

    Each device dict carries at least: index, name, max_input_channels,
    max_output_channels, default_samplerate. Returns ([], []) if
    sounddevice isn't installed (i.e. user hasn't run `swap voices install`).
    """
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except ImportError:
        return [], []

    inputs: list[dict] = []
    outputs: list[dict] = []
    try:
        for idx, dev in enumerate(sd.query_devices()):
            entry = dict(dev)
            entry["index"] = idx
            if int(dev.get("max_input_channels", 0)) > 0:
                inputs.append(entry)
            if int(dev.get("max_output_channels", 0)) > 0:
                outputs.append(entry)
    except Exception as err:  # noqa: BLE001
        print(f"[voice_router] device query failed: {err}", flush=True)
    return inputs, outputs


def pick_output_device(preferred_index: int | None = None) -> dict | None:
    """Best-effort: return the most appropriate output device.

    Order of preference:
      1. preferred_index (if it's still present in the device list)
      2. virtual cable detected in the device list (BlackHole / VB-Cable / …)
      3. None (caller decides whether to fall back to "drop output")
    """
    _inputs, outputs = list_audio_devices()
    if not outputs:
        return None
    if preferred_index is not None:
        for dev in outputs:
            if dev.get("index") == preferred_index:
                return dev
    return detect_virtual_cable_in_devices(outputs)


def pick_input_device(preferred_index: int | None = None) -> dict | None:
    """Pick the user's default microphone (preferred index → first input)."""
    inputs, _outputs = list_audio_devices()
    if not inputs:
        return None
    if preferred_index is not None:
        for dev in inputs:
            if dev.get("index") == preferred_index:
                return dev
    return inputs[0]
