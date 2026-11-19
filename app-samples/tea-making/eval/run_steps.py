# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the tea steps one at a time on a recording (eval/steps/*.yaml).

Each case runs the real step, taken from procedure.TeaProcedure as the
procedure runs it, with the inputs the case gives (what the procedure would
pass it), on recorded frames and a scripted person, and is scored by the
step type's own scoring (xr_ai_kit.sop.evaluation). From the sample's
folder, with the models in yaml/models.json running:

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


def steps(fakes: Any) -> Mapping[str, Step[Any]]:
    """The tea steps by id: the app's, or on scripted models (a plumbing
    run)."""
    # The app's real roles and settings (importing it builds the app, which is
    # why it happens only here).
    import app  # noqa: PLC0415

    tea_procedure = app.tea_procedure
    if fakes is not None:
        tea_procedure = TeaProcedure(
            perceive=fakes.role("perceive"),
            decide=fakes.role("decide"),
            answer_with_reasoning=fakes.role("answer_with_reasoning"),
            settings=app.SETTINGS.tea,
        )
    return {step.id: step for step in tea_procedure.steps}


def main(argv: list[str] | None = None) -> int:
    return evaluation.main(
        argv, steps=steps, cases=HERE / "steps", fakes_from=fakes_from_file
    )


if __name__ == "__main__":
    raise SystemExit(main())
