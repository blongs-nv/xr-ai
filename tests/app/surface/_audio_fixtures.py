# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import wave

import numpy as np
from PIL import Image
from xr_ai_app import App, Hub, PerConnection, Voice
from xr_ai_app.devkit import Scenario


def _wav(path=None, *, seconds=0.1):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((np.ones(round(16000 * seconds)) * 4000).astype("<i2").tobytes())
    data = output.getvalue()
    if path is not None:
        path.write_bytes(data)
    return data


def _scenario(tmp_path):
    _wav(tmp_path / "speech.wav")
    Image.new("RGB", (64, 48), "blue").save(tmp_path / "camera.png")
    return Scenario.from_dict(
        {
            "name": "real speech",
            "participant": "p1",
            "duration_s": 1.6,
            "steps": [
                {"join": "p1"},
                {"image": "camera.png"},
                {"at": 0.1, "audio": {"path": "speech.wav", "text": "hello"}},
                {"at": 0.8, "audio": {"path": "speech.wav", "text": "hello again"}},
                {"at": 1.0, "expect": {"said": "answer", "within_s": 1}},
            ],
        },
        base_dir=tmp_path,
    )


def _app(tmp_path, monkeypatch, *, transcribe=True):
    from xr_ai_app._base import _voice
    from xr_ai_voice import _session
    from xr_ai_voice._processors import speaker_stt

    class TTS:
        async def synthesize(self, text, **_kwargs):
            return _wav(seconds=0.04)

        async def health(self):
            return True

        async def close(self):
            pass

    class Speaker:
        def __init__(self, cfg):
            self.started = None
            self.enrolled = False

        async def _feed(self, pid, audio, pts):
            events = []
            if not self.enrolled:
                self.enrolled = True
                events.append({"kind": "enrolled"})
            if any(audio):
                if self.started is None:
                    self.started = pts
                    events.append({"kind": "speech_start"})
            elif self.started is not None:
                events.extend(
                    [{"kind": "transcript", "text": "hello", "pts_us": self.started}, {"kind": "speech_stop"}]
                )
                self.started = None
            return [event for event in events if transcribe or event["kind"] != "transcript"]

        async def _forget(self, *_args):
            pass

        async def _skip(self, *_args):
            return []

        async def _close(self):
            pass

    async def select(_cfg):
        return True

    monkeypatch.setattr(_voice, "make_tts", lambda *_args: TTS())
    monkeypatch.setattr(_session, "_select_speaker_asr", select)
    monkeypatch.setattr(speaker_stt, "_SpeakerClient", Speaker)
    (tmp_path / "models.json").write_text(
        json.dumps(
            {
                "models": {
                    "speech_out": {"adapter": {"preset": "pocket_tts"}, "endpoint": {"base_url": "http://127.0.0.1:9"}}
                }
            }
        )
    )
    (tmp_path / "voice_gate.yaml").write_text(
        "mode: always_on\nlistening_chime: false\nspeaker:\n  enabled: true\n  backend: required\n"
    )
    config = tmp_path / "app.yaml"
    config.write_text("profile: models.json\nvoice:\n  voice_gate_yaml: voice_gate.yaml\n")
    app = App.from_config(config, enrollment=PerConnection(admit=None))
    app.add(Hub())
    voice = app.add(Voice())
    return app, voice
