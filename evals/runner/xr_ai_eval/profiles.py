# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a profile (``PROFILES_SCHEMA.md`` of ``xr-ai-next``) to the endpoint that fills each role.

This is a deliberate, small duplicate of the resolver in ``xr_ai_next._base.profiles``. The runner
does not import ``xr_ai_next``, so evals run without the app SDK installed. It does not check categories or
``budgets``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

# Legacy entry names that fill the job-named roles (PROFILES_SCHEMA.md, backward compatibility).
LEGACY_ROLE_ALIASES = {
    "vlm": "perceive",
    "decision": "decide",
    "llm": "answer",
    "embedding": "embed",
    "stt": "speech_in",
    "tts": "speech_out",
}


class ProfileError(ValueError):
    """The profile cannot be found or read."""


@dataclass(frozen=True)
class Endpoint:
    """One OpenAI-compatible endpoint."""

    base_url: str
    model: str
    preset: str
    api_key_env: str | None = None


@dataclass(frozen=True)
class Profile:
    """A resolved profile."""

    name: str
    gpu_profile: str | None
    entries: dict[str, Endpoint]
    roles: dict[str, str]
    path: Path | None = None

    def entry_key(self, role: str) -> str:
        """Name of the ``models`` entry filling ``role`` (``roles`` map, exact entry, then legacy alias)."""
        entry = self.roles.get(role, role)
        if entry in self.entries:
            return entry
        for legacy, current in LEGACY_ROLE_ALIASES.items():
            if current == role and legacy in self.entries:
                return legacy
        raise ProfileError(f"profile {self.name!r} has no model for role {role!r}")

    def endpoint(self, role: str) -> Endpoint:
        """Return the endpoint filling ``role``."""
        return self.entries[self.entry_key(role)]

    def with_preset(self, preset: str) -> Profile:
        """Return a copy that asks every endpoint for ``preset`` (the matrix preset axis)."""
        return replace(
            self,
            entries={k: replace(v, model=preset, preset=preset) for k, v in self.entries.items()},
        )


def load_profile(value: str, search_dirs: list[Path]) -> Profile:
    """Load a profile by path, or by name from ``models.<name>.json|yaml|yml`` in ``search_dirs``."""
    if "/" in value or value.endswith((".json", ".yaml", ".yml")):
        path = Path(value)
        if not path.is_file():
            raise ProfileError(f"profile file not found: {value}")
        return _parse(path, path.stem.removeprefix("models.") or path.stem)
    candidates = [d / f"models.{value}.{ext}" for d in search_dirs for ext in ("json", "yaml", "yml")]
    for path in candidates:
        if path.is_file():
            return _parse(path, value)
    raise ProfileError(f"profile {value!r} not found; looked for: " + ", ".join(str(c) for c in candidates))


def _parse(path: Path, name: str) -> Profile:
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))  # JSON is valid YAML
    except yaml.YAMLError as exc:
        raise ProfileError(f"{path}: invalid profile: {exc}") from exc
    if not isinstance(doc, dict):
        raise ProfileError(f"{path}: expected a mapping")
    meta = doc.get("profile") if isinstance(doc.get("profile"), dict) else {}
    models = doc["models"] if "models" in doc else {k: v for k, v in doc.items() if isinstance(v, dict)}
    if not isinstance(models, dict) or not models:
        raise ProfileError(f"{path}: no models")
    entries = {str(key): _endpoint(path, str(key), value) for key, value in models.items()}
    roles = doc.get("roles") or {}
    if not isinstance(roles, dict):
        raise ProfileError(f"{path}: 'roles' must be a mapping")
    return Profile(
        name=str(meta.get("name") or name),
        gpu_profile=meta.get("gpu_profile"),
        entries=entries,
        roles={str(k): str(v) for k, v in roles.items()},
        path=path,
    )


def _endpoint(path: Path, key: str, entry: Any) -> Endpoint:
    if not isinstance(entry, dict):
        raise ProfileError(f"{path}: model {key!r} must be a mapping")
    adapter = entry.get("adapter") or {}
    endpoint = entry.get("endpoint") or {}
    base_url = endpoint.get("base_url") or entry.get("base_url")
    if not base_url:
        raise ProfileError(f"{path}: model {key!r} has no endpoint.base_url")
    model = adapter.get("model_name") or adapter.get("preset") or entry.get("model") or key
    preset = adapter.get("preset") or adapter.get("model_name") or str(model)
    return Endpoint(
        base_url=str(base_url),
        model=str(model),
        preset=str(preset),
        api_key_env=endpoint.get("api_key_env"),
    )


__all__ = ["LEGACY_ROLE_ALIASES", "Endpoint", "Profile", "ProfileError", "load_profile"]
