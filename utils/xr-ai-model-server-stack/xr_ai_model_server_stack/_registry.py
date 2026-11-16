# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative launch specs: ``deployment/services.yaml`` is the one registry of model servers and ports."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = ROOT / "deployment" / "services.yaml"
_KINDS = ("process", "container")


@dataclass(frozen=True)
class ServiceSpec:
    """How to start one model server and where it listens."""

    key: str
    project: str
    port: int
    kind: str = "process"
    command: str = ""
    config: str = ""
    aliases: tuple[str, ...] = ()
    image: str = ""
    container_port: int = 0
    container_env: dict[str, str] = field(default_factory=dict)


def parse_registry(raw: object, source: Path | str = REGISTRY_PATH) -> dict[str, ServiceSpec]:
    """Validate a parsed registry; the mapping keeps the file's (launch) order."""
    services = raw.get("services") if isinstance(raw, dict) else None
    if not isinstance(services, dict) or not services:
        raise ValueError(f"{source}: expected a non-empty 'services' mapping")
    specs: dict[str, ServiceSpec] = {}
    names: dict[str, str] = {}
    ports: dict[int, str] = {}
    for key, entry in services.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: {key} must be a mapping")
        kind = entry.get("kind", "process")
        if kind not in _KINDS:
            raise ValueError(f"{source}: {key}: kind must be one of {', '.join(_KINDS)}")
        port = entry.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"{source}: {key}: port must be an integer between 1 and 65535")
        if not isinstance(entry.get("project"), str):
            raise ValueError(f"{source}: {key}: project is required")
        needed = ("command", "config") if kind == "process" else ("image", "container_port")
        missing = [name for name in needed if not entry.get(name)]
        if missing:
            raise ValueError(f"{source}: {key}: {kind} services need {', '.join(missing)}")
        aliases = tuple(entry.get("aliases", ()))
        for name in (key, *aliases):
            if name in names:
                raise ValueError(f"{source}: name {name!r} is used by {names[name]} and {key}")
            names[name] = key
        if port in ports:
            raise ValueError(f"{source}: port {port} is used by {ports[port]} and {key}")
        ports[port] = key
        specs[key] = ServiceSpec(
            key=key, project=entry["project"], port=port, kind=kind,
            command=entry.get("command", ""), config=entry.get("config", ""), aliases=aliases,
            image=entry.get("image", ""), container_port=int(entry.get("container_port", 0)),
            container_env={str(k): str(v) for k, v in (entry.get("container_env") or {}).items()},
        )
    return specs


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, ServiceSpec]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: could not read service registry: {exc}") from exc
    return parse_registry(raw, path)
