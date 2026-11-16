# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate result files into the role-by-model scorecard (Markdown)."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable

from xr_ai_eval.schema import Record


def _median(values: list[float]) -> str:
    return f"{statistics.median(values):.1f}" if values else "-"


def build_scorecard(records: Iterable[Record]) -> str:
    """One row per ``(suite, role, preset)``: pass rate, median latency and median tokens.

    Output is deterministic: rows are sorted and no timestamp is written.
    """
    groups: dict[tuple[str, str, str], list[Record]] = defaultdict(list)
    for record in records:
        groups[(record.suite, record.role, record.preset)].append(record)
    lines = [
        "# Eval scorecard",
        "",
        "Pass rate counts every record, errors included. Latency, token and memory columns are medians over "
        "records without an error; `-` means no record reported a value.",
        "",
        "| Suite | Role | Model | Profiles | Cases | Pass rate | Errors | Median latency (ms) "
        "| Median tokens in | Median tokens out | Median GPU memory (MiB) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(groups):
        rows = groups[key]
        passed = sum(r.passed for r in rows)
        ok = [r for r in rows if r.error is None]
        tokens_in = [r.tokens["input"] for r in ok if r.tokens.get("input") is not None]
        tokens_out = [r.tokens["output"] for r in ok if r.tokens.get("output") is not None]
        memory = [r.gpu_memory_mb for r in ok if r.gpu_memory_mb is not None]
        profiles = ", ".join(sorted({r.profile for r in rows}))
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {profiles} | {len(rows)} | {passed / len(rows) * 100:.1f}% "
            f"| {sum(r.error is not None for r in rows)} | {_median([r.latency_ms for r in ok])} "
            f"| {_median([float(v) for v in tokens_in])} | {_median([float(v) for v in tokens_out])} "
            f"| {_median(memory)} |"
        )
    return "\n".join(lines) + "\n"
