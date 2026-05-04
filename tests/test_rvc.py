"""Tests for sprint 14b.2.b RVC plumbing.

What we CAN test without a GPU or rvc-python installed:
  - voice_ops.add_rvc_voice copies files + writes the right Voice JSON
  - voice_ops.remove_rvc_voice tears down model dir + JSON
  - voice_ops.rvc_model_path_for / rvc_index_path_for resolve correctly
  - RVCEngine.is_available() returns False when rvc_python module is absent
  - RVCEngine.extract_embedding raises with clear add-rvc guidance
  - RVCEngine.make_converter rejects non-RVC voices
  - RVCEngine.make_converter raises when the model file is missing

What needs the user's GPU machine (NOT covered here):
  - Actual rvc-python loading the model
  - Inference quality / latency
  - End-to-end audio through VB-Cable
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def isolated_data_dirs(tmp_path, monkeypatch):
    """Point user_data_dir + user_voices_dir at a fresh tmp_path.

    Doing this via env vars so platformdirs picks them up. Also reload the
    relevant modules so cached values don't leak between tests.
    """
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    import importlib

    from swap_cli import voice_library, voice_prereq

    importlib.reload(voice_prereq)
    importlib.reload(voice_library)
    yield tmp_path


@pytest.fixture
def fake_pth(tmp_path) -> Path:
    """Write a small bytes file masquerading as a .pth (we never load it)."""
    p = tmp_path / "fake_voice.pth"
    p.write_bytes(b"\x00" * 1024)  # 1 KB of nulls — enough to copy + read length
    return p


# ── voice_ops.add_rvc_voice / remove_rvc_voice / path lookups ──────────────


def test_add_rvc_voice_copies_pth_and_writes_record(
    isolated_data_dirs, fake_pth
) -> None:
    from swap_cli import voice_library, voice_ops

    voice = voice_ops.add_rvc_voice(fake_pth, name="Test Voice")

    # Voice id has the rvc- prefix so RVCEngine recognises it.
    assert voice.id.startswith("rvc-")
    assert voice.name == "Test Voice"
    assert voice.embedding == []  # RVC voices use the .pth, not an embedding
    assert voice.source == "library"

    # The Voice was persisted via save_user_voice → load_user_voices picks it up.
    user_voices = voice_library.load_user_voices()
    assert any(v.id == voice.id for v in user_voices)

    # The .pth was copied into rvc_models_dir/<voice_id>/.
    pth = voice_ops.rvc_model_path_for(voice)
    assert pth is not None
    assert pth.exists()
    assert pth.read_bytes() == fake_pth.read_bytes()

    # No .index supplied → rvc_index_path_for returns None.
    assert voice_ops.rvc_index_path_for(voice) is None


def test_add_rvc_voice_with_index(isolated_data_dirs, fake_pth, tmp_path) -> None:
    from swap_cli import voice_ops

    fake_index = tmp_path / "fake.index"
    fake_index.write_bytes(b"\x01" * 512)
    voice = voice_ops.add_rvc_voice(fake_pth, name="With Index", index_path=fake_index)
    idx = voice_ops.rvc_index_path_for(voice)
    assert idx is not None
    assert idx.read_bytes() == fake_index.read_bytes()


def test_add_rvc_voice_rejects_missing_file(isolated_data_dirs) -> None:
    from swap_cli import voice_ops

    with pytest.raises(FileNotFoundError):
        voice_ops.add_rvc_voice(Path("/no/such/file.pth"), name="X")


def test_add_rvc_voice_rejects_wrong_extension(isolated_data_dirs, tmp_path) -> None:
    from swap_cli import voice_ops

    not_pth = tmp_path / "wrong.bin"
    not_pth.write_bytes(b"\x00")
    with pytest.raises(ValueError, match=".pth"):
        voice_ops.add_rvc_voice(not_pth, name="X")


def test_remove_rvc_voice(isolated_data_dirs, fake_pth) -> None:
    from swap_cli import voice_ops

    voice = voice_ops.add_rvc_voice(fake_pth, name="Removable")
    pth = voice_ops.rvc_model_path_for(voice)
    assert pth is not None and pth.exists()

    assert voice_ops.remove_rvc_voice(voice.id) is True
    assert voice_ops.rvc_model_path_for(voice) is None
    # JSON also gone.
    from swap_cli.voice_library import find_voice

    assert find_voice(voice.id) is None


def test_remove_rvc_voice_nonexistent_returns_false(isolated_data_dirs) -> None:
    from swap_cli import voice_ops

    assert voice_ops.remove_rvc_voice("rvc-not-here") is False


def test_rvc_model_path_for_non_rvc_voice_returns_none(isolated_data_dirs) -> None:
    from swap_cli import voice_ops
    from swap_cli.voice_library import Voice

    not_rvc = Voice(
        id="aria",
        name="Aria",
        description="",
        source="library",
        embedding=[0.0] * 256,
        sample_rate=16_000,
        created_at=0,
    )
    assert voice_ops.rvc_model_path_for(not_rvc) is None


# ── RVCEngine ──────────────────────────────────────────────────────────────


def test_rvc_engine_is_available_false_without_rvc_python() -> None:
    """When rvc_python isn't installed, is_available() is False — never raises."""
    from swap_cli import voice_engines

    engine = voice_engines.get_engine("rvc")
    # In this CI environment rvc-python is not installed.
    result = engine.is_available()
    assert isinstance(result, bool)
    assert result is False


def test_rvc_engine_extract_embedding_explains_use_add_rvc() -> None:
    """Calling extract_embedding directs users to add-rvc."""
    from swap_cli import voice_engines

    engine = voice_engines.get_engine("rvc")
    with pytest.raises(RuntimeError, match="add-rvc"):
        engine.extract_embedding("/dev/null")


def test_rvc_engine_make_converter_rejects_non_rvc_voice(isolated_data_dirs) -> None:
    """make_converter checks the voice id has the rvc- prefix."""
    from swap_cli import voice_engines
    from swap_cli.voice_library import Voice

    not_rvc = Voice(
        id="aria",
        name="Aria",
        description="",
        source="library",
        embedding=[0.0] * 256,
        sample_rate=16_000,
        created_at=0,
    )
    engine = voice_engines.get_engine("rvc")
    with pytest.raises(RuntimeError, match="isn't an RVC voice"):
        engine.make_converter(target_voice=not_rvc)


def test_rvc_engine_make_converter_rejects_missing_model(isolated_data_dirs) -> None:
    """If a Voice claims to be RVC but the .pth file is missing on disk,
    make_converter raises a clear RuntimeError."""
    from swap_cli import voice_engines
    from swap_cli.voice_library import Voice

    fake_rvc = Voice(
        id="rvc-not-installed",
        name="Not Installed",
        description="",
        source="library",
        embedding=[],
        sample_rate=16_000,
        created_at=0,
    )
    engine = voice_engines.get_engine("rvc")
    with pytest.raises(RuntimeError, match="model file not found"):
        engine.make_converter(target_voice=fake_rvc)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
