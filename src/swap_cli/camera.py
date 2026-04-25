"""OpenCV webcam capture exposed as an aiortc VideoStreamTrack."""

from __future__ import annotations

import asyncio
import fractions
import time
from typing import Final

import av
import cv2
from aiortc import VideoStreamTrack

VIDEO_CLOCK_RATE: Final = 90_000  # standard for video tracks


class CameraTrack(VideoStreamTrack):
    """Wraps a `cv2.VideoCapture` device as a WebRTC video track.

    Designed to be passed to `RealtimeClient.connect(local_track=...)`.
    Frames are pulled lazily as Decart consumes them, paced to the model's fps.
    """

    kind = "video"

    def __init__(self, *, device: int = 0, width: int, height: int, fps: int) -> None:
        super().__init__()
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera device {device}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        self._width = width
        self._height = height
        self._fps = fps
        self._frame_interval = 1.0 / fps
        self._next_frame_at = time.monotonic()
        self._frame_count = 0
        self._stopped = False

    async def recv(self) -> av.VideoFrame:  # noqa: D401 — aiortc protocol
        # Pace at the model's fps so we don't spam frames faster than Decart
        # negotiates them on the wire.
        now = time.monotonic()
        wait = self._next_frame_at - now
        if wait > 0:
            await asyncio.sleep(wait)
        self._next_frame_at = max(now, self._next_frame_at) + self._frame_interval

        ok, frame = await asyncio.to_thread(self._cap.read)
        if not ok or frame is None:
            raise RuntimeError("Camera read failed — is another app using the webcam?")

        # OpenCV is BGR; aiortc/PyAV want a known pixel format.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        video_frame.pts = self._frame_count
        video_frame.time_base = fractions.Fraction(1, self._fps)
        self._frame_count += 1
        return video_frame

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self._cap.release()
        except Exception:
            pass
        super().stop()
