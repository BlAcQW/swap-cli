"""Audio output routing for voice track.

Picks where the converted voice goes:
- A virtual audio cable device (BlackHole / VB-Cable / pulse-loopback)
  so Zoom/OBS/Discord can pick it up as a microphone source.
- Optionally also playback through default speakers for self-monitoring.

13b.1: helpers + constants. Actual streaming output is wired into
voice_track._loop in 13b.2 via sounddevice OutputStream.
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
