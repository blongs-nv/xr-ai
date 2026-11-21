# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare and pin reusable speech/media fixtures before measurement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from xr_ai_models import make_tts

from ._targets import _AppTarget


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(directory, name):
    if not isinstance(name, str) or not name:
        raise ValueError("fixture path must be a nonempty string")
    path = (directory / name).resolve()
    if not path.is_relative_to(directory):
        raise ValueError(f"fixture path escapes output directory: {name}")
    return path



async def _prepare_fixtures(app, scenario, directory, *, images=None, tts_entry=None):
    target = _AppTarget(app)
    try:
        return await _prepare(target, scenario, Path(directory).resolve(), images or {}, tts_entry)
    finally:
        await target.stop()


async def _prepare(target, scenario, directory, images, tts_entry):
    from ._audio import _prepare_audio, _wav_samples  # noqa: PLC0415

    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / ".gitignore").exists():
        (directory / ".gitignore").write_text("*\n")
    manifest_path = directory / "fixtures.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}, "utterances": {}}
    if (
        not isinstance(manifest, dict)
        or set(manifest) - {"files", "utterances"}
        or not isinstance(manifest.get("files"), dict)
        or not isinstance(manifest.get("utterances", {}), dict)
    ):
        raise ValueError("fixture manifest requires files and optional utterances mappings")
    manifest.setdefault("utterances", {})
    for name, digest in manifest["files"].items():
        path = _inside(directory, name)
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"pinned fixture changed or missing: {name}; use a new fixture directory")
    audio, media = {}, set()
    for step in scenario.steps:
        if step.action not in {"audio", "image"}:
            continue
        name = step.value.get("path") if isinstance(step.value, dict) else step.value
        _inside(directory, name)
        media.add(name)
        if step.action == "audio":
            if not isinstance(step.value, dict):
                raise ValueError("preparing audio requires path and nonempty text")
            text = step.value.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("preparing audio requires path and nonempty text")
            if name in audio and audio[name] != text:
                raise ValueError(f"audio path used for different utterances: {name}")
            if name in manifest["utterances"] and manifest["utterances"][name] != text:
                raise ValueError(f"pinned audio has different text: {name}; use a new fixture directory")
            audio[name] = text
    image_names = media - set(audio)
    if set(images) - image_names:
        raise ValueError(f"unknown image bindings: {sorted(set(images) - image_names)}")
    for name in image_names:
        path = _inside(directory, name)
        source = Path(images[name]) if name in images else (path if path.is_file() else scenario.base_dir / name)
        if not source.is_file():
            raise ValueError(f"missing image fixture {name}; supply --image {name}=PATH")
        data = source.read_bytes()
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"fixture already exists with different contents: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest["files"][name] = _sha(path)
    client = None
    try:
        for name, text in audio.items():
            path = _inside(directory, name)
            if not path.exists():
                if client is None:
                    client = make_tts(target.profile.models, tts_entry or target.tts_entry)
                data = await client.synthesize(text, timeout=120)
                _wav_samples(data, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            _wav_samples(path.read_bytes(), name)
            manifest["files"][name] = _sha(path)
            manifest["utterances"][name] = text
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        _prepare_audio(scenario, directory, 0)
    finally:
        if client is not None:
            await client.close()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path
