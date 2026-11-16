# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture helpers: GPU memory and the repo commit. Both are injectable so tests never shell out."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

GpuMemoryReader = Callable[[], float | None]


def nvidia_smi_memory_mb() -> float | None:
    """Total used GPU memory across all GPUs in MiB, or ``None`` when ``nvidia-smi`` is unavailable.

    The value is host-wide, not per process, so it includes other users of the GPU.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return float(sum(float(line) for line in out.split() if line.strip()))
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def git_commit(repo: Path) -> str:
    """Short commit hash of ``repo``, or ``unknown`` outside a git checkout."""
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short=8", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"
