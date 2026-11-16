# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The speaker-ASR endpoint comes from the profile's ``speech_in``; ``voice_gate.yaml`` stays a fallback."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from xr_ai_voicegate import load_voice_gate_config

_ROOT = Path(__file__).resolve().parents[1]
_ENABLED = "mode: conversation\nspeaker:\n  enabled: true\n  backend: required\n  base_url: http://127.0.0.1:8102\n"


def test_default_loading_is_unchanged(tmp_path):
    path = tmp_path / "voice_gate.yaml"
    path.write_text(_ENABLED)
    assert load_voice_gate_config(path)._speaker.base_url == "http://127.0.0.1:8102"


def test_speech_in_endpoint_replaces_the_yaml_key(tmp_path):
    path = tmp_path / "voice_gate.yaml"
    path.write_text(_ENABLED)
    cfg = load_voice_gate_config(path, speaker_base_url="http://10.0.0.5:9000")
    assert cfg._speaker.base_url == "http://10.0.0.5:9000"
    assert cfg._speaker.backend == "required"


def test_speaker_enabled_override_switches_the_backend_with_the_profile(tmp_path):
    path = tmp_path / "voice_gate.yaml"
    path.write_text(_ENABLED)
    assert load_voice_gate_config(path, speaker_enabled=False)._speaker is None
    path.write_text("mode: always_on\n")
    cfg = load_voice_gate_config(path, speaker_enabled=True, speaker_base_url="http://10.0.0.5:9000")
    assert cfg._speaker is not None and cfg._speaker.base_url == "http://10.0.0.5:9000"


def test_speech_in_endpoint_does_not_enable_the_speaker_backend(tmp_path):
    path = tmp_path / "voice_gate.yaml"
    path.write_text("mode: always_on\n")
    assert load_voice_gate_config(path, speaker_base_url="http://10.0.0.5:9000")._speaker is None
    assert load_voice_gate_config(tmp_path / "missing.yaml", speaker_base_url="http://10.0.0.5:9000")._speaker is None


def test_speech_in_endpoint_is_validated(tmp_path):
    path = tmp_path / "voice_gate.yaml"
    path.write_text(_ENABLED)
    with pytest.raises(ValueError, match="HTTP origin"):
        load_voice_gate_config(path, speaker_base_url="https://host/path")


def _explicit_speaker_base_url(path: Path) -> str | None:
    speaker = (yaml.safe_load(path.read_text()) or {}).get("speaker")
    value = speaker.get("base_url") if isinstance(speaker, dict) else None
    return value if isinstance(value, str) else None


def test_every_sample_voice_gate_still_loads_with_its_key():
    for gate in (_ROOT / "agent-samples").glob("*/yaml/voice_gate*.yaml"):
        cfg = load_voice_gate_config(gate)
        explicit = _explicit_speaker_base_url(gate)
        if cfg._speaker is not None:
            assert cfg._speaker.base_url == explicit or explicit is None
