# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the tea steps one at a time on a recording (eval/steps/*.yaml).

Each case runs the real step, built by procedure.build_steps exactly as the
procedure builds it, on recorded frames and a scripted person, and is scored
by the step type's own rules (xr_ai_kit.sop.eval). From the sample's folder,
with the models in yaml/models.json running:

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

from procedure import build_steps  # noqa: E402
from xr_ai_app.devkit import fakes_from_file  # noqa: E402
from xr_ai_kit.sop import Roles, Step  # noqa: E402
from xr_ai_kit.sop import eval as step_eval  # noqa: E402


def build(values: Mapping[str, Any], fakes: Any) -> Mapping[str, Step[Any]]:
    """The app's steps; on scripted roles for a plumbing run."""
    import app  # the app's roles and settings  # noqa: PLC0415

    roles = app.ROLES
    if fakes is not None:
        roles = Roles(
            observe=fakes.role("perceive"),
            decide=fakes.role("decide"),
            reason=fakes.role("answer_with_reasoning"),
        )
    return build_steps(roles, app.SETTINGS.tea, **values)


def main(argv: list[str] | None = None) -> int:
    return step_eval.main(
        argv, build=build, cases=HERE / "steps", fakes_from=fakes_from_file
    )


if __name__ == "__main__":
    raise SystemExit(main())
