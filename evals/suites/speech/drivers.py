# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Call driver for the speaker-gating cases: stream recorded audio to the speaker-stt WebSocket.

The wire contract is private to the voice workers (``services/speaker-stt``), so it lives here and not in
the runner: an opening JSON message (``config``, ``audio_origin_us``), then binary 16 kHz mono signed 16-bit
PCM in frames of at most one second; the service answers each frame with ``{"events": [...]}``.
"""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path

_PATH = "/v1/audio/transcriptions/stream"
_RATE = 16000
_FRAME_BYTES = 16000  # half a second; the service accepts at most 32000


def read_pcm(path: Path) -> bytes:
    """Raw PCM of a 16 kHz mono 16-bit WAV; anything else raises ``ValueError``."""
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (_RATE, 1, 2):
            raise ValueError(f"{path.name}: need 16 kHz mono 16-bit PCM WAV")
        return wav.readframes(wav.getnframes())


def ws_url(base_url: str) -> str:
    """The stream URL for an HTTP origin."""
    origin = base_url.rstrip("/").removesuffix("/v1")
    return origin.replace("http://", "ws://", 1).replace("https://", "wss://", 1) + _PATH


async def speaker_gate(endpoint, case) -> str:
    """Stream ``params.clips`` in order and return a JSON summary of what the service emitted.

    ``params``: ``clips`` (WAV paths relative to the case file), optional ``config`` (speaker settings
    merged into the opening message), optional ``tail_silence_s`` (silence appended so the last utterance
    closes; default 2). The summary is ``{"enrolled", "transcripts", "event_kinds"}``.
    """
    from websockets.asyncio.client import connect  # imported late: only gating cases need it

    params = case.params
    pcm = b"".join(read_pcm(case.base_dir / name) for name in params["clips"])
    pcm += bytes(int(params.get("tail_silence_s", 2) * _RATE) * 2)
    opening = {"audio_origin_us": 0, "config": {"mode": "conversation", **params.get("config", {})}}
    enrolled, transcripts, kinds = False, [], []
    async with connect(ws_url(endpoint.base_url), max_size=None) as socket:
        await socket.send(json.dumps(opening))
        hello = json.loads(await asyncio.wait_for(socket.recv(), 10))
        if "error" in hello:
            raise RuntimeError(f"speaker service refused the stream: {hello['error']}")
        for start in range(0, len(pcm), _FRAME_BYTES):
            await socket.send(pcm[start : start + _FRAME_BYTES])
            reply = json.loads(await asyncio.wait_for(socket.recv(), 60))
            if "error" in reply:
                raise RuntimeError(f"speaker service error: {reply['error']}")
            for event in reply.get("events", []):
                kinds.append(event.get("kind"))
                enrolled = enrolled or event.get("kind") == "enrolled"
                if event.get("kind") == "transcript":
                    transcripts.append(event.get("text", ""))
    return json.dumps({"enrolled": enrolled, "transcripts": transcripts, "event_kinds": kinds})
