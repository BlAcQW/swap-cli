# Prepaid (managed credit) — install & setup

This is the guide for **managed mode**: you buy prepaid **minutes** instead of
holding your own AI-backend API key. There's nothing to configure on the
provider side — the compute is ours, metered against your balance.

> **Version:** these steps assume **swap-cli 0.1.7 or newer** (the managed
> build). `swap version` prints what you have.

## What you'll have been given

| Item | Looks like | What it's for |
|---|---|---|
| **Licence key** | `SWAP-CLI-XXXX-XXXX-XXXX` | Identifies you. Entered once. |
| **Voucher code(s)** | `SWAP-CR-XXXX-XXXX-XXXX` | Each one adds minutes to your balance. |

You do **not** need a Decart or fal API key in managed mode.

---

## Track A — New user (fresh machine)

### 1. Get the app

- **No Python (recommended):** download the latest
  `swap-cli-vX.Y.Z-windows-x64.exe` from the
  [Releases page](https://github.com/BlAcQW/swap-cli/releases) and double-click.
  Nothing to install.
- **Have Python 3.11+:** `pip install swap-cli`

### 2. Enter your licence key

- **App / GUI:** the first launch with no key opens **⚙ Settings** — paste your
  `SWAP-CLI-…` key into **License key** and click **Save**.
- **Command line:** `swap setup --license "SWAP-CLI-XXXX-XXXX-XXXX"`

### 3. Load your minutes

- **GUI:** ⚙ Settings → **Prepaid credit** → paste `SWAP-CR-…` → **Redeem**.
- **Command line:** `swap redeem SWAP-CR-XXXX-XXXX-XXXX`

You'll see the balance as **minutes** (e.g. `Balance: 4m 00s`).

### 4. (WhatsApp only) install a camera driver

Skip this unless you want the swapped video *inside a video call*.

| Calling from | Driver | Download |
|---|---|---|
| **WhatsApp Desktop** | DroidCam **Drivers** | [dev47apps releases](https://github.com/dev47apps/droidcam-obs-virtual-output/releases) → `DroidCam.Drivers.exe` |
| Zoom / Meet / Discord / Teams | OBS Studio | [obsproject.com/download](https://obsproject.com/download) |

Pick it in the GUI under **Camera output**.

### 5. Go live

`swap gui` → pick a face → **Live**. The header shows **Credit: N minutes** and
counts down as you stream. Under 5 seconds it warns; at zero the session closes.

### Check your balance any time

- **GUI:** the credit shows in the window header and in ⚙ Settings.
- **Command line:** `swap doctor` → the **credit remaining** row.

---

## Track B — Existing BYOK user switching to prepaid

You're currently on "bring your own key" and want to move to prepaid minutes.

> **Operator prerequisite (server side — the customer can't do this).**
> Managed vs BYOK is a property of the *licence on the server*. Before the
> steps below, in the admin either:
> - set that licence's **mode = managed** (keeps the same key), **or**
> - issue a **new managed licence** and give them the new key.
>
> The prepaid wallet is created automatically the first time the machine makes
> a managed call; the voucher (step 3) is what puts minutes in it.

### 1. Update the app to 0.1.7+

| How it was installed | Update it |
|---|---|
| Source checkout (`E:\swap-cli`) | `git pull` then `pip install -e .` |
| pip | `pip install --upgrade swap-cli` |
| Moving to the `.exe` | `pip uninstall -y swap-cli`, then download the exe |

*(Do **not** delete `config.toml` — see "About your config" below.)*

### 2. Point the machine at the managed licence

This also forces the app to drop its cached `byok` mode and re-check with the
server. Works whether the key changed or stayed the same:

```
swap setup --license "SWAP-CLI-<the managed key>"
```

### 3. Load your minutes

```
swap redeem SWAP-CR-XXXX-XXXX-XXXX
```

### 4. Verify the switch

```
swap config      # billing → "prepaid credits"
swap doctor      # shows "credit remaining: N"
```

In the GUI, the API-key rows disappear and **Credit: N minutes** appears.

---

## About your config (both tracks)

Your settings live in one file:

- **Windows:** `C:\Users\<you>\AppData\Local\swap-cli\config.toml`
- **macOS:** `~/Library/Application Support/swap-cli/config.toml`
- **Linux:** `~/.config/swap-cli/config.toml`

- **Keep this file.** The new version reads an old config fine, and the file
  holds your licence key and a stable machine id. Deleting it makes you re-enter
  the licence and can register a **new machine seat** against the licence
  (`too_many_machines`).
- **Leftover BYOK keys are harmless.** Any old `decart_api_key` / `fal_api_key`
  are simply ignored in managed mode; you don't need to remove them.
- **Only clean up a stray temp file** if you see one (a leftover from an old
  crash): delete `config.toml.tmp` in that folder — never `config.toml`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| GUI still shows API-key fields, `swap config` says "own API key" | The licence isn't managed on the server yet, or the client is still on its cached `byok`. Confirm the licence is managed in the admin, then re-run `swap setup --license "…"` and relaunch. |
| `too_many_machines` | A drifting machine id burned seats. Release stale machines for that licence in the admin, then relaunch. |
| Crash on **Live** with `PermissionError: [WinError 32]` | Fixed in 0.1.7 — update the app. |
| A licence error on launch | `swap doctor` prints the real reason in plain English. |
| Redeem says the code is used / expired | Vouchers are single-use; check you're pasting an unredeemed one. |
