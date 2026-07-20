# Changelog

All notable changes to swap-cli.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.6] — 2026-07-20 — managed credit + Windows installer

The first release that carries the managed-credits work. Until now the only
published build (0.1.5) was bring-your-own-key; everything below shipped on
`main` but never reached PyPI.

### Added

- **Managed mode** — buy prepaid credit instead of holding a provider API key.
  `swap redeem CODE` claims a voucher; the GUI and `swap` show remaining credit
  as **minutes**, not tokens. Sessions are metered server-side against your
  balance, so a silent client can't run up the bill.
- **Self-serve licence entry in the GUI** — the licence key is now an editable
  field in ⚙ Settings, and a first launch with no key opens Settings straight
  away. This is what makes the Windows `.exe` self-sufficient: a customer with no
  Python (so no `swap setup`) can paste their key and redeem a voucher entirely
  in the window.
- **fal.ai as a switchable realtime engine** (`--provider fal`) alongside Decart,
  selectable before going Live.
- **DroidCam virtual-camera output** — the one virtual camera WhatsApp actually
  accepts, so the swapped feed shows up in a WhatsApp video call.
- **Windows `.exe`** on the [Releases page](https://github.com/BlAcQW/swap-cli/releases)
  — a double-click desktop build for people who don't have Python. Built by the
  new `build-windows.yml` workflow on each tag.

### Changed

- The licence is now enforced **before** a session starts, and revoking a licence
  takes effect immediately.
- Licence failures are shown in plain English instead of a raw server code.
- Managed customers never see the underlying vendor's name in the UI.

### Fixed

- The machine id is now stable across restarts, so a single install no longer
  burns through its machine seats.

## [0.1.1] — 2026-06-XX — first PyPI release

First public release on PyPI. `pip install swap-cli` now works.

### Packaging

- Added PyPI classifiers, project URLs, keywords, author email.
- Bumped version 0.1.0 → 0.1.1.
- Added GitHub Actions workflows:
  - `release.yml` — tag-triggered publish to PyPI via Trusted Publishing (OIDC).
  - `ci.yml` — push/PR matrix tests across Ubuntu/macOS/Windows × Python 3.11/3.12.
- Verified base install is clean on Apple Silicon Mac (all deps have arm64 wheels).

### Documentation

- README: added "macOS users — read this first" callout at the top of the
  Install section pointing at python.org / `brew install python-tk@3.11`.
- README: new "macOS compatibility" section detailing per-feature support,
  install prereqs, and known limitations (Apple Silicon CPU is too slow
  for live RVC voice; Intel Mac is unsupported for voice).
- New `CHANGELOG.md` (this file).
- New `docs/RELEASING.md` documenting the tag → publish workflow.

### Notes

This release reflects the cumulative work through Sprint 14o. See the
Sprint Appendix below for the feature provenance and the commit log for
fine-grained changes.

### Sprint appendix (feature provenance)

| Sprint | What landed |
|---|---|
| 14a | SOLA crossfade for voice streaming (w-okada-inspired) |
| 14b.1–b.3 | Voice engine registry abstraction; RVC engine |
| 14c | Mac install hardening (rvc-python out of extras, Tcl/Tk check) |
| 14d | Install/engine UX hardening |
| 14e | Dropped OpenVoice; voice = RVC only |
| 14f | Curated voice catalog + auto-starter |
| 14g.0–g.5 | RVC cascade fixes (dataclass patch, deps, fork swap, CUDA torch) |
| 14h | Audio plumbing diagnostics |
| 14i | w-okada-inspired voice perf pass |
| 14j | Halved speak-to-hear latency |
| 14k | Virtual camera output (OBS Virtual Camera, no OBS app needed) |
| 14l | In-GUI Settings panel (rotate Decart key without CLI) |
| 14m | Decart reconnect timeout 20→45s + pyvirtualcam doctor visibility |
| 14n | OBS Virtual Camera detection via rglob (modern OBS layouts) |
| 14o | Refuse OBS Virtual Camera as input (feedback-loop prevention) |
| 14p | First PyPI release + Mac compat docs (this release) |

## [0.1.0] — internal

Pre-PyPI development versions. Not published.
