# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container-kind services (the OCR server): ``docker run`` and ``docker stop`` from a launch spec."""
from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from ._registry import ServiceSpec

_READY_PATH = "/v1/health/ready"


def container_name(spec: ServiceSpec, namespace: str) -> str:
    return f"xr-ai-{spec.key}-{namespace}".replace("_", "-")


def run_command(spec: ServiceSpec, config: dict, namespace: str, models_dir: Path) -> list[str]:
    """The ``docker run`` command for a container service.

    ``config`` is the registry spec merged with the GPU preset's overrides; ``cuda_visible_devices``
    selects the GPU. The API is published on localhost only. The image must already be built.
    """
    devices = str(config.get("cuda_visible_devices", "all"))
    command = [
        "docker", "run", "-d", "--rm", "--name", container_name(spec, namespace),
        "--gpus", "all" if devices == "all" else f"device={devices}", "--ipc=host",
        "-e", "HF_TOKEN",
    ]
    env = {**spec.container_env, **{str(k): str(v) for k, v in (config.get("container_env") or {}).items()}}
    for name, value in env.items():
        command += ["-e", f"{name}={value}"]
    command += ["-p", f"127.0.0.1:{spec.port}:{spec.container_port}", "-v", f"{models_dir}:/models", spec.image]
    return command


def stop_command(spec: ServiceSpec, namespace: str) -> list[str]:
    return ["docker", "stop", container_name(spec, namespace)]


def _ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{_READY_PATH}", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def start(
    spec: ServiceSpec, config: dict, namespace: str, models_dir: Path, *,
    timeout_s: float = 900.0, runner: Callable[..., object] = subprocess.run,
    ready: Callable[[int], bool] = _ready, sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Start the container and wait for readiness; stop it again if readiness never comes."""
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        runner(run_command(spec, config, namespace, models_dir), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"{spec.key}: docker run failed ({exc}); build the image first: {spec.image}"
        ) from exc
    deadline = time.monotonic() + timeout_s
    while not ready(spec.port):
        if time.monotonic() > deadline:
            stop(spec, namespace, runner=runner)
            raise RuntimeError(f"{spec.key}: not ready on port {spec.port} after {timeout_s:.0f}s")
        sleep(2.0)


def stop(spec: ServiceSpec, namespace: str, *, runner: Callable[..., object] = subprocess.run) -> bool:
    try:
        runner(stop_command(spec, namespace), check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True
