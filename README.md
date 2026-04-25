# swap-cli

**Real-time deepfake on your desktop. Bring your own Decart API key.**

Live face-swap built on Decart Lucy 2 — your camera, your machine,
your Decart account. We charge once for the wrapper; you pay Decart
directly for the seconds you stream.

```bash
pip install swap-cli
swap setup        # paste license + Decart key, one time
swap doctor       # verify everything works
swap gui          # OR  swap run -r face.jpg
```

---

## What you need before you start

| Item | Where to get it | Cost |
|---|---|---|
| A laptop with a webcam | Your own | — |
| Python 3.11+ | [python.org](https://python.org) / Homebrew | Free |
| **swap-cli license key** | Bought at [swap.ikieguy.online](https://swap.ikieguy.online) | One-time fee |
| **Decart API key** | [platform.decart.ai](https://platform.decart.ai) | Free credits → ~$0.01/sec after |

---

## 1 — Install

```bash
pip install swap-cli
```

Pulls native deps: `decart`, `aiortc`, `opencv-python`, `customtkinter`. ~150 MB,
~60 seconds on a normal connection.

If `swap` isn't on your `$PATH` after install, fix it:

```bash
python -m swap_cli --help
```

---

## 2 — First-time setup

```bash
$ swap setup
License key (SWAP-CLI-…): SWAP-CLI-7K3M-9PQR-XW4T
Decart API key (dct_…):  ************************

╭───────── ✓ Setup complete ─────────╮
│ Saved to ~/.config/swap-cli/config.toml
│ License: SWAP…XW4T
│ Decart key: dct_…AB12
╰────────────────────────────────────╯
```

Both keys live at `~/.config/swap-cli/config.toml` (`chmod 600`).
Linux/macOS: `~/.config/swap-cli/`. Windows: `%APPDATA%\swap-cli\`.

---

## 3 — Verify (recommended)

```bash
$ swap doctor
                 swap-cli doctor
 license key set         ✓
 decart api key set      ✓
 dns swap.ikieguy.online  ✓
 dns api.decart.ai        ✓
 license validate         ✓ ok
 camera probe             ✓
 aiortc import            ✓
 decart import            ✓
 opencv import            ✓
 av import                ✓
```

Any ✗ tells you exactly what to fix before opening a real session.

---

## 4 — Run it

### Option A — GUI (recommended for non-developers)

```bash
swap gui
```

A small dark window opens:

```
┌────────────────────────────────────┐
│ swap-cli · live deepfake           │
├────────────────────────────────────┤
│       ┌────────────────┐           │
│       │  [face thumb]  │           │
│       └────────────────┘           │
│       [ ① Select a face ]          │
│                                    │
│  ⚪ Mirror camera   ⚪ Record MP4  │
│                                    │
│  Model  [ lucy-2 (1280×720, 20fps)▼│
│  Decart fixes resolution per model.│
│                                    │
│  ⚙ Advanced                        │
│                                    │
│  ② Camera                          │
│  [ Camera 0 (default) ▼ ]    [↻]   │
│                                    │
│  [    ③  Live    ]   [   Stop   ]  │
└────────────────────────────────────┘
```

Three steps:

1. **Select a face** — file picker → JPG/PNG of who you want to become
2. **Camera** — auto-populated; ↻ refreshes
3. **Live** — opens the deepfake stream in a new window

The "⚙ Advanced" expander reveals an optional prompt textbox if you want
to add stylistic modifiers ("…with neon eye makeup", etc). Most users
leave it alone.

### Option B — CLI (power users / scripting)

```bash
swap run --reference identity.jpg
```

Long form:

```bash
swap run \
  --reference faces/elon.jpg \
  --prompt "Match the reference person's face and identity" \
  --model lucy-2 \
  --device 0 \
  --record output.mp4
```

Phase log on stdout:

```
[runtime] connection: connecting
[runtime] connection: connected
[runtime] connection: generating
streaming · 14s
```

Press **Q** in the preview window or **Ctrl-C** in the terminal to stop.

---

## How it works under the hood

```
                    YOUR LAPTOP
   ┌─────────────────────────────────────────────┐
   │                                             │
   │  webcam ──► cv2.VideoCapture                │
   │              ↓                              │
   │       aiortc VideoStreamTrack               │
   │              ↓                              │
   │           WebRTC ─────────► api.decart.ai   │
   │                              (Lucy 2 GPU)   │
   │                                ↓            │
   │           WebRTC ◄────────── transformed    │
   │              ↓                stream        │
   │       cv2.imshow window                     │
   │              ↓                              │
   │  optional MP4 recording                     │
   └─────────────────────────────────────────────┘
```

- **Latency**: ~150–300 ms end-to-end (camera → Decart → screen)
- **FPS**: 20 (capped by Lucy 2; higher tiers TBA)
- **Quality**: 1280×720 fixed (model-side; we don't choose)
- **Network**: ~2–3 Mbps up + down per stream
- **Privacy**: every frame stays between your laptop and Decart.
  swap-cli phones home **once per launch** for license validation —
  no camera frames, no audio, ever leave your machine via us.

---

## Output files

If you passed `--record`:

```
$ ls -la *.mp4
-rw-r--r-- 12.4M  output.mp4
```

Standard H.264 MP4 of the **transformed** stream. Drag into iMovie /
Premiere / DaVinci, ffmpeg-crop to 9:16 for TikTok or 1:1 for
Instagram, upload anywhere.

> ⚠ swap-cli streams at the model's native 16:9 — Decart doesn't
> accept other ratios on the wire. To export 9:16 / 1:1 / 4:5 for
> social, ffmpeg-crop the saved MP4 *after* the session.

---

## Costs you pay, ongoing

We bill once for the license. Compute is on your Decart account.

| Decart pricing tier (your account) | Cost / sec | 5 min session | 1 hour session |
|---|---|---|---|
| Lucy Fast | ~$0.01 | $3 | $36 |
| Lucy Pro  | ~$0.05 | $15 | $180 |

Your Decart dashboard shows the running tally. We're never in that loop.

---

## Commands

| Command | Purpose |
|---|---|
| `swap setup` | Save license + Decart API key |
| `swap config` | Show current config (keys redacted) |
| `swap doctor` | Verify camera, network, license, deps |
| `swap gui` | Launch the desktop GUI |
| `swap run` | Start a realtime session from the terminal |
| `swap version` | Print version |

---

## Troubleshooting

**`swap doctor` says `license validate ✗`**

| Reason | Fix |
|---|---|
| `license_revoked` | Email the seller — most likely chargeback or abuse flag |
| `license_expired` | Renew with the seller |
| `too_many_machines` | You hit your seat cap. Email the seller to either bump it or reset registered machines |
| `invalid_format` | You typed it wrong. Re-paste with `swap setup` |

**`camera probe ✗ no camera at index 0`**

- macOS: System Settings → Privacy → Camera → enable for Terminal/iTerm
- Windows: Settings → Privacy → Camera → allow desktop apps
- Linux: check `/dev/video*` exists and your user is in the `video` group

**Preview window is black / frozen**

- Decart's first frames take ~2 seconds to arrive. Wait.
- Stuck longer than 10s? Press Q, check `swap doctor`, retry
- Decart credits exhausted → check your dashboard at platform.decart.ai

**`tkinter` import error on Linux**

```bash
sudo apt install python3-tk    # Debian/Ubuntu
sudo dnf install python3-tkinter  # Fedora
```

---

## Update

```bash
pip install --upgrade swap-cli
```

License keys carry over (stored in user config, not in the package).
Your machine ID stays stable so you don't burn a slot on the seat cap.

---

## Privacy

- Your **Decart API key** never leaves your machine.
- License validation pings `swap.ikieguy.online` once per launch with
  a hashed machine ID. No camera frames, no IP geolocation, no analytics.
- Generated frames stay on your machine unless you opt in to recording.

---

## License

Commercial. See [LICENSE.md](LICENSE.md). Buy a license key at
[swap.ikieguy.online](https://swap.ikieguy.online).

---

## In one screen

```bash
pip install swap-cli                                     # install
swap setup                                               # paste keys once
swap doctor                                              # verify
swap gui          # OR:  swap run -r face.jpg            # use it
```

That's the whole system.
