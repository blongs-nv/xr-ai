# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python scorer for the xr-render scenario cases: the original corpus checker on the driver's tool calls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_DRIVERS = Path(__file__).with_name("drivers.py")
_spec = importlib.util.spec_from_file_location("xr_render_scenes_drivers", _DRIVERS)
_drivers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_drivers)


def scenario(case, result) -> tuple[float, dict]:
    """1.0 when ``check_corpus`` accepts the calls (perception gating, expected mutations, duplicate-add rule)."""
    summary = json.loads(result.text)
    calls = [(name, args) for name, args in summary["calls"]]
    ok, why = _drivers.harness().check_corpus(calls, _drivers.corpus_case(case.params["case"]))
    return (1.0 if ok else 0.0), {"why": why, "reply": summary["reply"], "tools": [name for name, _ in calls]}
