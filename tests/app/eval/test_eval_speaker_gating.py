# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The speech suite's gating driver against the real speaker-stt app (fake model session) over a socket."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

service = pytest.importorskip("speaker_stt.__main__")
uvicorn = pytest.importorskip("uvicorn")
pytest.importorskip("websockets")

from test_eval_suites import SPEECH, SUITES, _load  # noqa: E402
from xr_ai_eval.cli import main  # noqa: E402
from xr_ai_eval.profiles import Endpoint  # noqa: E402

_CFG = {"host": "127.0.0.1", "port": 0, "asr_model": "stub-asr", "diar_model": "stub-diar"}


class _Session:
    """Enrolls after the first frames, then transcribes: the wire contract is real, the model is not."""

    def __init__(self):
        self.frames = 0

    def _feed(self, audio):
        self.frames += 1
        if self.frames == 1:
            return [{"kind": "enrolled"}]
        return [{"kind": "transcript", "text": f"utterance {self.frames}"}] if self.frames == 3 else []


@pytest.fixture
def speaker_url():
    server = service._Server(SimpleNamespace(_session=lambda cfg, *, audio_origin_us: _Session()))
    config = uvicorn.Config(service._build_app(server, _CFG), host="127.0.0.1", port=0, log_level="error")
    runner = uvicorn.Server(config)
    thread = threading.Thread(target=runner.run, daemon=True)
    thread.start()
    for _ in range(200):
        if runner.started and runner.servers:
            break
        time.sleep(0.05)
    port = runner.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    runner.should_exit = True
    thread.join(5)


def test_driver_streams_clips_and_summarises_events(speaker_url, tmp_path):
    drivers = _load(SPEECH / "drivers.py", "speech_drivers")
    case = SimpleNamespace(
        base_dir=SPEECH / "cases", params={"clips": ["../fixtures/silence_2s.wav"], "tail_silence_s": 0}
    )
    summary = json.loads(asyncio.run(drivers.speaker_gate(Endpoint(speaker_url, "m", "m"), case)))
    assert summary["enrolled"] is True and summary["transcripts"] == ["utterance 3"]
    assert summary["event_kinds"][0] == "enrolled"


def test_gating_case_runs_through_the_runner(speaker_url, tmp_path):
    profile = tmp_path / "models.local.json"
    profile.write_text(
        json.dumps(
            {"models": {"speaker_gate": {"adapter": {"preset": "speaker_stub"}, "endpoint": {"base_url": speaker_url}}}}
        )
    )
    out = tmp_path / "r.jsonl"
    code = main(["run", "--suite", "speech", "--evals-dir", str(SUITES), "--profile", str(profile), "--out", str(out),
                 "--no-gpu-memory", "--tags", "gating"])  # fmt: skip
    (line,) = Path(out).read_text().splitlines()
    record = json.loads(line)
    assert record["case"] == "gate-silence-no-enrollment" and "error" not in record
    # The fake session enrolls on any audio, so the silence case (expects no enrollment) fails by design.
    assert code == 1 and record["pass"] is False and record["detail"]["failure"] == "enrollment mismatch"
    assert record["role"] == "speaker_gate" and record["preset"] == "speaker_stub"
