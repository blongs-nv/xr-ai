# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample 10 on xr_ai_kit.sop: its scenarios with their scripted models, and the runner's controls in code.

Runs on the devkit's virtual clock (a camera step looks once a second; warm_up is a 90 s wait), on sample 10
unchanged.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from xr_ai_app.devkit import Scenario, fakes_from_file, run_scenario, run_virtual
from xr_ai_app.testing import FakeModels, solid_frame

SCENARIOS = Path(__file__).resolve().parents[3] / "app-samples" / "sdk-tour" / "10-guided-procedure" / "scenarios"


@pytest.mark.parametrize("name", ["checklist", "checklist_on_demand"])
def test_sample_scenarios_pass(load_sample, name: str) -> None:  # noqa: ANN001
    pytest.importorskip("cv2")  # checklist.yaml plays a video
    m = load_sample("10-guided-procedure")
    scenario = Scenario.load(SCENARIOS / f"{name}.yaml")
    report = run_virtual(run_scenario(m.app, scenario, models=fakes_from_file(SCENARIOS / f"{name}.fakes.yaml")))
    assert report.passed, report.transcript()


def test_start_skip_and_stop_in_code(load_sample) -> None:  # noqa: ANN001
    m = load_sample("10-guided-procedure")
    models = FakeModels(
        {
            "perceive": ["The print bed is in view and empty."] * 2 + ["The printer door is open."] * 40,
            "decide": ["positive"] * 2 + ["negative"] * 40,
        }
    )

    async def scenario() -> tuple[str, list[tuple[str, str]], str]:
        async with m.app.test_session(models=models) as s:
            await s.join()
            p = s.participant()
            await m.procedure.start(p)  # no turn, no control decision
            run = m.procedure.latest_run(p)
            for _ in range(40):
                await s.show_frame(solid_frame("red"))
                if run.current is not None and run.current.id == "door_closed":
                    break
                await asyncio.sleep(1)
            await m.procedure.skip(p)  # the door never closes: move on
            for _ in range(20):
                if run.current is not None and run.current.id == "warm_up":
                    break
                await asyncio.sleep(0.5)
            await m.procedure.stop(p)
            return run.status, [(step.id, step.status) for step in run.steps], m.procedure.summary(p)

    status, steps, summary = run_virtual(scenario())
    assert status == "stopped"
    assert steps == [("bed_clear", "done"), ("door_closed", "skipped"), ("warm_up", "running")]
    assert "Printer pre-flight (run 1): stopped." in summary
