# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""App lifecycle and configuration binding for the shared audio runner."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from .._base._voice import _gate_path, _speech
from .._base.profiles import load_profile
from ..voice import Voice

_ROOT = Path(__file__).resolve().parents[4]


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _settings_identity(settings, base):
    opaque = set()

    def encode(value):
        if isinstance(value, Path):
            path = value.resolve()
            relative = (
                path.relative_to(base)
                if path.is_relative_to(base)
                else (path.relative_to(_ROOT) if path.is_relative_to(_ROOT) else path)
            )
            return {"path": str(relative), "sha256": _hash(path.read_bytes()) if path.is_file() else None}
        name = type(value).__module__ + "." + type(value).__qualname__
        opaque.add(name)
        return {"opaque_type": name}

    raw = json.dumps(dataclasses.asdict(settings), default=encode, sort_keys=True).encode()
    return {"settings_sha256": _hash(raw), "unserialized_settings": sorted(opaque)}


def _identity(profile, settings, base, gate, source):
    return {
        "profile_sha256": _hash(profile.path.read_bytes()),
        **_settings_identity(settings, base),
        "models": {
            name: {"served_alias": spec.adapter.model_name, "endpoint_host": urlsplit(spec.endpoint.base_url).hostname}
            for name, spec in profile.models.entries.items()
        },
        "roles": dict(profile.roles),
        "voice_gate_sha256": _hash(gate) if gate is not None else None,
        "app_source_sha256": _hash(source.read_bytes()) if source else "unmeasured",
    }


def _source_identity():
    import xr_ai_hub
    import xr_ai_runtime
    import xr_ai_tools
    import xr_ai_voice

    import xr_ai_app

    checkout = (_ROOT / "app-sdk/xr-ai-app/xr_ai_app/devkit/_targets.py").resolve() == Path(__file__).resolve()
    checkout = checkout and (_ROOT / "AGENTS.md").is_file()
    revision = diff = None
    if checkout:
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, check=False
            )
            diff = subprocess.run(["git", "diff", "HEAD"], cwd=_ROOT, capture_output=True, check=False)
        except FileNotFoundError:
            pass
    digest = hashlib.sha256()
    for package in (xr_ai_app, xr_ai_voice, xr_ai_runtime, xr_ai_tools, xr_ai_hub):
        root = Path(package.__file__).parent
        for path in sorted(root.rglob("*.py")):
            digest.update((package.__name__ + "/" + str(path.relative_to(root))).encode())
            digest.update(path.read_bytes())
    return {
        "revision": revision.stdout.strip() if revision is not None and revision.returncode == 0 else None,
        "tracked_diff_sha256": _hash(diff.stdout) if diff is not None and diff.returncode == 0 else None,
        "sdk_sha256": digest.hexdigest(),
        "host_platform": platform.platform(),
    }


class _AppTarget:
    def __init__(self, app):
        if not app._config.voice.enabled or app._voice_factory is None:
            raise ValueError("real audio requires the app's real voice pipeline")
        self.app = app
        self.trace = app.trace
        gate = _gate_path(app)
        self.gate = gate.read_bytes() if gate.exists() else None
        self.profile = app._load_profile()
        if load_profile(self.profile.path) != self.profile:
            raise ValueError("model profile changed since app import; reload the app before evaluation")
        self.identity = _identity(
            self.profile,
            app._config,
            app._config.config_dir.resolve(),
            self.gate,
            getattr(app, "_source_path", None),
        )
        speech = _speech(app)
        self.identity["speech_roles"] = {key: getattr(speech[key], "name", None) for key in ("stt", "tts")}
        self.identity["text_input"] = speech["text_input"]
        self.tts_entry = getattr(speech["tts"], "name", None) or (
            "speech_out" if "speech_out" in self.profile.resolve_all() else "tts"
        )
        self.tts_entry = self.profile.resolve(self.tts_entry)

    @contextlib.contextmanager
    def configure(self, root, out):
        app = self.app
        original, ready = app._config, app._ready_file
        component = next((item for item in app._components if isinstance(item, Voice)), None)
        with contextlib.ExitStack() as cleanup:
            cleanup.callback(setattr, app, "_config", original)
            cleanup.callback(setattr, app, "_ready_file", ready)
            if component is not None:
                cleanup.callback(setattr, component, "_speech", component._speech)
            snapshot = None
            if self.gate is not None:
                snapshot = root / "voice_gate.yaml"
                snapshot.touch(mode=0o600)
                snapshot.write_bytes(self.gate)
                if component is not None:
                    component._speech = {**component._speech, "gate": snapshot}
            app._config = dataclasses.replace(
                original,
                artifacts_dir=out / "artifacts",
                voice=dataclasses.replace(original.voice, voice_gate_yaml=snapshot) if snapshot else original.voice,
            )
            app._ready_file = root / "app.ready"
            yield

    async def start(self):
        await self.app.start()

    def ready(self):
        voice_tasks = [task for task in self.app._app_tasks if task.get_name() == "app:voice"]
        if not voice_tasks and not self.app._ready_file.exists():
            raise RuntimeError("voice pipeline exited before readiness")
        return self.app._ready_file.exists()

    async def stop(self):
        await self.app.stop()
