# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the tea steps one at a time on a recording (eval/steps/*.yaml).

Each case runs the real step, taken from procedure.TeaProcedure exactly as
the procedure runs it, on recorded frames and a scripted person, and is
scored by the step type's own scoring (xr_ai_kit.sop.evaluation). From the
sample's folder, with the models in yaml/models.json running:

    uv --config-file ../../uv.toml run python eval/run_steps.py \\
        --frames-index /path/to/frames/index.csv --out runs/steps/

--only heat_water (a step) or --only looked_away (a case) runs some;
--fakes eval/smoke.fakes.yaml runs the plumbing on scripted replies.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from procedure import TeaProcedure  # noqa: E402
from xr_ai_app.devkit import fakes_from_file  # noqa: E402
from xr_ai_kit.sop import Step, evaluation  # noqa: E402


def build(values: Mapping[str, Any], fakes: Any) -> Mapping[str, Step[Any]]:
    """The tea steps by id. A case file's build values (temperature_c,
    display_unit, duration_s) replace the app's settings; with fakes, the
    models give scripted replies (a plumbing run)."""
    import app  # the app's models and settings  # noqa: PLC0415

    unknown = set(values) - {"temperature_c", "display_unit", "duration_s"}
    if unknown:
        raise ValueError(f"unknown build values: {sorted(unknown)}")
    settings = app.SETTINGS.tea
    tea_procedure = app.tea_procedure
    if fakes is not None:
        tea_procedure = TeaProcedure(
            perceive=fakes.role("perceive"),
            decide=fakes.role("decide"),
            answer_with_reasoning=fakes.role("answer_with_reasoning"),
            settings=settings,
        )
    temperature_c = values.get("temperature_c", settings.default_temperature_c)
    display_unit = values.get("display_unit", settings.display_unit)
    duration_s = values.get("duration_s", settings.default_duration_s)
    return {
        "identify": tea_procedure.identify,
        "fill_water": tea_procedure.fill_water,
        "heat_water": tea_procedure.heat_water(temperature_c, display_unit),
        "start_steeping": tea_procedure.start_steeping,
        "steep": tea_procedure.steep(duration_s),
    }


def main(argv: list[str] | None = None) -> int:
    return evaluation.main(
        argv, build=build, cases=HERE / "steps", fakes_from=fakes_from_file
    )


if __name__ == "__main__":
    raise SystemExit(main())
