# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deprecated task-done banner (dead code; kept so xr_ai_logging.print_task_done_banner still resolves)."""

from __future__ import annotations

import sys

_STATUS_COLORS: dict[str, str] = {
    "done":        "\x1b[1;32m",   # bright green, bold
    "interrupted": "\x1b[1;33m",   # bright yellow, bold
    "error":       "\x1b[1;31m",   # bright red, bold
}


def print_task_done_banner(
    label: str,
    *,
    status: str = "done",
    detail: str | None = None,
    duration_s: float | None = None,
) -> None:
    """One-shot banner on stderr marking the end of an agent task.

    Mirrors :func:`_print_log_dir_banner` styling so begin/end milestones
    bracket the run with matching dim-grey bars. ``status`` selects the
    headline color (``done`` green, ``interrupted`` yellow, ``error`` red);
    unknown statuses fall back to the ``done`` color.
    """
    bar = "─" * 78
    is_tty = sys.stderr.isatty()
    on  = _STATUS_COLORS.get(status, _STATUS_COLORS["done"]) if is_tty else ""
    dim = "\x1b[2m" if is_tty else ""
    off = "\x1b[0m" if is_tty else ""

    head = f"{status} · {label}"
    if duration_s is not None:
        head += f"  ({duration_s:.2f}s)"

    print(f"\n{dim}{bar}{off}",       file=sys.stderr, flush=True)
    print(f"  {on}{head}{off}",       file=sys.stderr, flush=True)
    if detail:
        print(f"  {dim}{detail}{off}", file=sys.stderr, flush=True)
    print(f"{dim}{bar}{off}\n",       file=sys.stderr, flush=True)
