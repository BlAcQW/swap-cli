# PyInstaller spec for the swap-cli Windows desktop build.
#
# Produces a single `swap-cli.exe` that opens the GUI directly (the frozen
# equivalent of `swap gui`) so a user without Python can just double-click it.
#
# Build from the repo root:
#     pyinstaller packaging/swap-gui.spec --noconfirm
#
# Notes on the choices here:
#   * onefile — one downloadable artifact for non-technical users. It unpacks
#     to a temp dir on launch (slower cold start, fine for a desktop app).
#   * console=True — this is an early build; if the GUI fails to construct,
#     swap_cli.gui.launch() prints the traceback and pauses. A console window
#     lets the user screenshot that error instead of the window vanishing.
#     Flip to False once the build is proven in the field.
#   * The `voice` extra (torch/torchaudio/librosa/… ~3 GB) is never part of the
#     base install, so it is excluded explicitly as a guard — the exe stays lean.

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

# customtkinter ships themes/fonts/assets as data files it loads at runtime;
# without this the window fails to build. Bundle our watermark template too.
datas = collect_data_files("customtkinter")
datas += [
    (
        os.path.join(ROOT, "src", "swap_cli", "assets", "watermark_default.png"),
        os.path.join("swap_cli", "assets"),
    ),
]

# Native-backed packages PyInstaller can under-discover via static analysis.
hiddenimports = (
    collect_submodules("av")
    + collect_submodules("aiortc")
    + collect_submodules("pyvirtualcam")
    + ["pygrabber", "pygrabber.dshow_graph"]
)

# Voice stack is opt-in only and must never be dragged into the exe.
excludes = [
    "torch",
    "torchaudio",
    "librosa",
    "soundfile",
    "sounddevice",
    "huggingface_hub",
]

block_cipher = None

a = Analysis(
    [os.path.join(SPECPATH, "entry_gui.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="swap-cli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
