# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the eval runner tests: a real OpenAI-compatible stub server on an ephemeral port.

Imported by the test files. A `conftest.py` here would shadow `tests/conftest.py` by module name.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


class StubServer:
    """Answers ``/v1/chat/completions``. The first rule whose key is in the last user text wins."""

    def __init__(self) -> None:
        self.rules: dict[str, str] = {}
        self.default = "unknown"
        self.usage: dict[str, int] | None = {"prompt_tokens": 10, "completion_tokens": 3}
        self.fail_with: int | None = None
        self.transcript = "hello world"
        self.uploads: list[dict] = []
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path.endswith("/audio/transcriptions"):
                    outer.uploads.append({"content_type": self.headers["Content-Type"], "bytes": len(raw), "raw": raw})
                    data = json.dumps({"text": outer.transcript}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                body = json.loads(raw)
                body["_path"] = self.path
                body["_auth"] = self.headers.get("Authorization")
                outer.requests.append(body)
                if outer.fail_with:
                    self.send_response(outer.fail_with)
                    self.end_headers()
                    return
                content = body["messages"][-1]["content"]
                text = content if isinstance(content, str) else content[0]["text"]
                reply = next((v for k, v in outer.rules.items() if k in text), outer.default)
                payload = {"choices": [{"message": {"content": reply}}]}
                if outer.usage is not None:
                    payload["usage"] = outer.usage
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def stub() -> StubServer:
    server = StubServer()
    yield server
    server.close()


@pytest.fixture
def evals_dir(tmp_path: Path) -> Path:
    """An evals tree with one suite, ``kettle``, plus a frame fixture and a profile."""
    suite = tmp_path / "evals" / "kettle"
    (suite / "fixtures").mkdir(parents=True)
    (suite / "cases").mkdir()
    (suite / "fixtures" / "frame.jpg").write_bytes(b"\xff\xd8\xff-frame")
    (suite / "cases" / "basic.yaml").write_text(
        """
suite: kettle
version: 1
defaults: {role: perceive, pass_threshold: 1.0}
cases:
  - id: boiling
    input: {text: "Is the kettle boiling?", frames: [../fixtures/frame.jpg]}
    scorer: {type: choice, expect: positive, choices: [positive, negative, unknown]}
    tags: [visual]
  - id: json-fields
    role: extract
    input: {text: "Extract the temp", context: [{role: user, text: earlier}]}
    scorer: {type: json_match, expect: {temp: 90, unit: C}}
    pass_threshold: 0.5
  - id: skipped
    input: {text: "never run"}
    scorer: {type: exact, expect: x}
    skip_reason: needs audio
""",
        encoding="utf-8",
    )
    return tmp_path / "evals"


def write_profile(directory: Path, name: str, base_url: str, **extra: str) -> Path:
    path = directory / f"models.{name}.json"
    doc = {
        "profile": {"name": name, "gpu_profile": "dual_48G_ada"},
        "models": {
            "vlm": {"adapter": {"preset": "cosmos_stub"}, "endpoint": {"base_url": base_url}},
            "stt": {"adapter": {"preset": "stt_stub"}, "endpoint": {"base_url": base_url}},
            "llm": {
                "adapter": {"kind": "openai_compat", "model_name": "big-stub"},
                "endpoint": {"base_url": base_url + "/v1", **extra},
            },
        },
        "roles": {"extract": "llm"},
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path
