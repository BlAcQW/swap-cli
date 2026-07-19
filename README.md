# swap-cli

**Real-time deepfake on your desktop. Bring your own API key.**

Live face-swap built on Decart's Lucy realtime model — your camera, your
machine, your account. We charge once for the wrapper; you pay the AI backend
(fal.ai or Decart) directly for the seconds you stream.

```bash
pip install swap-cli
swap setup        # paste your license key, one time
swap doctor       # verify everything works
swap gui          # OR  swap run -r face.jpg
```

---

## What you need before you start

### 1. Always required

| Item | Where to get it | Cost |
|---|---|---|
| A laptop with a webcam | Your own | — |
| Python 3.11+ | [python.org](https://python.org) / Homebrew | Free |
| **swap-cli license key** | [swap.storelygh.com](https://swap.storelygh.com) | One-time fee |
| **swap-cli itself** | `pip install swap-cli` | — |

### 2. An AI backend key — pick ONE

The swap runs on the same Lucy model either way; the difference is the
watermark and who bills you. Set it in **⚙ Settings** (`swap setup` only asks
for your license key).

| Backend | Where to get the key | Cost | Watermark |
|---|---|---|---|
| **fal.ai** *(recommended)* | [fal.ai/dashboard/keys](https://fal.ai/dashboard/keys) | **$0.04/sec**, pay as you go | **None** |
| Decart | [platform.decart.ai](https://platform.decart.ai) | Free credits → **$0.02/sec** | Adds an "✦ AI Generated" badge |

> Decart stamps its badge at the account level and won't lift it on request.
> swap-cli can remove it per frame (`Remove watermark` in the GUI), but fal's
> output has no badge to begin with — which is why it's the default recommendation.

### 3. A camera driver — only if you want swap-cli inside a video call

Without this, swap-cli still runs and shows its preview window; you just can't
select it as a camera in Zoom/WhatsApp/etc. **Which one you need depends on the
app you're calling from:**

| Driver | Download | Works in | Does NOT work in |
|---|---|---|---|
| **DroidCam drivers** | [dev47apps releases](https://github.com/dev47apps/droidcam-obs-virtual-output/releases) → `DroidCam.Drivers.exe` | **WhatsApp Desktop**, Zoom, Meet, Discord | — (Windows only) |
| **OBS Studio** | [obsproject.com/download](https://obsproject.com/download) | Zoom, Meet, Discord, Teams | **WhatsApp Desktop** |

> **Downloading DroidCam? Take `DroidCam.Drivers.exe` only** — *not*
> `DroidCam.OBSVirtualOut.Plugin.exe`. swap-cli replaces that plugin and writes
> to the drivers directly, so you need neither the plugin nor OBS. Installing
> both would make two programs fight over the same buffer.
>
> **Why WhatsApp is fussy:** OBS's camera is a DirectShow filter, which
> WhatsApp Desktop doesn't enumerate. DroidCam registers a *real* camera device
> ("Droidcam Video" in Device Manager), so WhatsApp accepts it.

Pick the driver in the GUI under **Camera output**. Neither is needed for
recording to a file or for the preview window.

### 4. Optional extras

| Item | Download | What it's for |
|---|---|---|
| ffmpeg | [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) / `brew install ffmpeg` | Voice cloning; non-WAV audio |
| Virtual audio cable | [VB-Cable](https://vb-audio.com/Cable/) / `brew install blackhole-2ch` | Routing cloned voice into calls |

---

## 1 — Install

```bash
pip install swap-cli
```

Pulls native deps: `decart`, `aiortc`, `opencv-python`, `customtkinter`,
`pyvirtualcam`, `fal-client`. ~150 MB, ~60 seconds on a normal connection.
Nothing else is needed to *run* swap-cli — the downloads in section 3 above
only matter for appearing as a camera inside a video call.

If `swap` isn't on your `$PATH` after install, fix it:

```bash
python -m swap_cli --help
```

> **macOS users — read this first.** Don't use the system Python at
> `/usr/bin/python3`. Its Tcl/Tk is 8.5.9 (broken) and the GUI will fail.
> Use Homebrew Python or python.org Python:
>
> ```bash
> brew install python@3.11 python-tk@3.11        # GUI needs Tcl/Tk ≥ 8.6.9
> xcode-select --install                          # git + C compiler (voice)
> brew install blackhole-2ch                      # voice routing (optional)
> # Then download OBS Studio for the virtual camera driver:
> #   https://obsproject.com/download
> pip install swap-cli
> ```
>
> See **macOS compatibility** below for the full breakdown.

---

## 2 — First-time setup

```bash
$ swap setup
License key (SWAP-CLI-…): SWAP-CLI-7K3M-9PQR-XW4T

╭───────── ✓ Setup complete ─────────╮
│ Saved to ~/.config/swap-cli/config.toml
│ License: SWAP…XW4T
│ No backend API key yet — add one in ⚙ Settings
╰────────────────────────────────────╯
```

Setup only asks for your **license**. Add your fal (or Decart) API key in
**⚙ Settings** inside `swap gui`, where you also choose which backend runs the
swap. For scripted installs you can pass keys directly instead:

```bash
swap setup --license "SWAP-CLI-…" --fal-key "fal_sk_…"
```

Everything lives at `~/.config/swap-cli/config.toml` (`chmod 600`).
Linux/macOS: `~/.config/swap-cli/`. Windows: `%APPDATA%\swap-cli\`.

---

## 3 — Verify (recommended)

```bash
$ swap doctor
                 swap-cli doctor
 license key set         ✓
 decart api key set      ⚠ not set
 fal api key set         ✓
 dns swap.storelygh.com  ✓
 license validate        ✓ valid
 camera probe            ✓
 aiortc import           ✓
 opencv import           ✓
 av import               ✓
 virtual camera          ✓ DroidCam camera ready (works in WhatsApp)
```

**✗ = broken**, and `swap doctor` exits non-zero so scripts can gate on it.
**⚠ = fine to ignore** — above, only one backend key is set, which is all you
need. Every ✗ names the fix.

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

We bill once for the license. Compute is on your own fal.ai or Decart account.

| Backend (your account) | Model | Cost / sec | 5 min | 1 hour |
|---|---|---|---|---|
| **fal.ai** | Lucy 2.5 realtime | $0.04 | $12 | $144 |
| **Decart** | Lucy 2 realtime | $0.02 | $6 | $72 |

**Badge-free costs double.** fal charges $0.04/sec for Lucy 2.5 — the model that
does identity swap — while Decart bills $0.02/sec direct but stamps its
"✦ AI Generated" badge on every frame.

fal does list a $0.02/sec Lucy 2 endpoint, but it's `lucy2-vton` — the virtual
**try-on** model. It swaps clothing, not faces, so it can't do swap-cli's job.
There is no cheaper badge-free option today.

If cost matters more than the badge, run Decart and turn on `Remove watermark`
in the GUI. If the badge matters more than cost, run fal.

Your fal/Decart dashboard shows the running tally. We're never in that loop.

---

## Commands

| Command | Purpose |
|---|---|
| `swap setup` | Save your license key (backend key goes in ⚙ Settings) |
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

## macOS compatibility

The base install (`pip install swap-cli`) works cleanly on both Apple Silicon
and Intel Macs — every base dependency has a pre-built arm64 wheel, no
compilation needed. The differences are concentrated in the optional voice
features.

| Feature | macOS support |
|---|---|
| Live deepfake (Decart Lucy 2) | ✅ Works — runs in Decart's cloud, no local GPU needed |
| `swap gui` (customtkinter window) | ⚠️ Needs Tcl/Tk ≥ 8.6.9 — system Python's 8.5.9 won't work; use python.org or `brew install python-tk@3.11` |
| Virtual camera output → Zoom / Meet / Discord | ✅ Works via OBS Virtual Camera (install OBS Studio once) |
| Virtual camera output → WhatsApp Desktop | ✅ Windows only, via the DroidCam drivers — OBS's camera is refused by WhatsApp |
| `swap voices add` (one-shot reference voice extraction) | ✅ Works on CPU, ~5 s per add |
| Live RVC voice streaming on **Apple Silicon** | ⚠️ CPU-only — too slow for real-time conversation; doctor reports honestly |
| Live RVC voice streaming on **Intel Mac** | ❌ Unsupported (no NVIDIA, no usable CPU path) |
| First-time voice install (`swap voices install`) | ⚠️ Needs Xcode CLT for `git` + C compiler (`xcode-select --install`) |
| Audio routing into Zoom/Meet | ✅ BlackHole (`brew install blackhole-2ch`) |

### macOS install (one-time setup)

```bash
# 1. Python 3.11 with proper Tcl/Tk
brew install python@3.11 python-tk@3.11

# 2. Xcode Command Line Tools (git + C compiler)
xcode-select --install

# 3. Optional: virtual audio cable for voice routing into Zoom/Meet
brew install blackhole-2ch

# 4. Optional: OBS Studio for the virtual camera driver
# https://obsproject.com/download    (no need to run the OBS app)

# 5. swap-cli itself
pip install swap-cli

# 6. Verify
swap doctor
```

Every row in `swap doctor` should be `✓`. The macOS-specific rows
(`tcl/tk version`, `virtual camera`) tell you exactly what to fix
if they're not.

### What does NOT work well on macOS yet

- **Live voice on Apple Silicon** — RVC streaming is GPU-bound; Apple's
  MPS path is broken upstream for fairseq (the model loader). On M-series
  CPUs you'll get ~4–6× real-time speed which is too slow for live
  conversation. Voice add/extract still works fine.
- **Live voice on Intel Mac** — no GPU path at all.
- **FaceTime / iOS apps** — Apple sandboxes virtual cameras for those.
  Zoom / Meet / Discord / browsers see "OBS Virtual Camera" normally.

---

## Update

```bash
pip install --upgrade swap-cli
```

License keys carry over (stored in user config, not in the package).
Your machine ID stays stable so you don't burn a slot on the seat cap.

---

## Privacy

- Your **backend API key** (fal or Decart) never leaves your machine.
- License validation pings `swap.storelygh.com` once per launch with
  a hashed machine ID. No camera frames, no IP geolocation, no analytics.
- Generated frames stay on your machine unless you opt in to recording.

---

## License

Commercial. See [LICENSE.md](LICENSE.md). Buy a license key at
[swap.storelygh.com](https://swap.storelygh.com).

---

## In one screen

```bash
pip install swap-cli                                     # install
swap setup                                               # paste keys once
swap doctor                                              # verify
swap gui          # OR:  swap run -r face.jpg            # use it
```

That's the whole system.
