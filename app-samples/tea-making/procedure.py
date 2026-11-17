# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tea procedure: which steps, in what order, and what passes between.

identify        read the tea's label, derive brewing settings (a read step)
fill_water      water visible in the kettle (a watch step)
heat_water      the display reads at least the target (a watch step, on a
                schema: number, unit, current or setpoint)
start_steeping  tea touching the water in a real cup (a watch step)
steep           a countdown for the recommended time (a wait step)

build_steps makes the steps from steps.yaml and this app's models and
schemas. Each camera step has an escalator beside it (Escalated): an agent
of its own, configured per step, that reads what the step sees and, when it
cannot decide, asks the person or tells them what to show. The evaluation
(eval/run_steps.py) builds them the same way, so the step it evaluates is
the step that runs. Each step says its own lines (steps.yaml's say and
done); the procedure says only what depends on more than one step. The
runner (SopRunner in app.py) owns the run, skips and stops. A skipped
identification falls back to the default settings here, never inside a
step.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from tea import (
    Brewing,
    DisplayFacts,
    TeaLabel,
    Thermometer,
    assessable,
    reconcile,
    target,
)
from xr_ai_app import Role
from xr_ai_kit.sop import (
    Escalated,
    EscalationSettings,
    Escalator,
    ReadStep,
    Run,
    Step,
    StepDefinitions,
    WaitStep,
    WatchSettings,
    WatchStep,
)

DEFINITIONS = StepDefinitions.load(
    Path(__file__).resolve().parent / "steps.yaml"
)
WATCHED = {"fill_water", "heat_water", "start_steeping"}
"""The camera steps, each with its own escalator."""


@dataclass(frozen=True)
class TeaModels:
    """The models the tea steps use, by what each does (app.py's roles)."""

    observe: Role
    """Reads a frame: the label, the kettle, the display."""
    decide: Role
    """Labels an observation against a step's criteria (fast)."""
    reason: Role
    """Derives brewing settings; each escalator reasons with it."""


class TeaSettings(BaseModel):
    """The procedure's settings (app.yaml's settings.tea)."""

    default_temperature_c: float = 85.0
    default_duration_s: float = 180.0
    display_unit: Literal["celsius", "fahrenheit"] | None = None
    """The kettle display's unit, when it prints none."""
    watch: WatchSettings = Field(default_factory=WatchSettings)
    escalation: EscalationSettings = Field(
        default_factory=lambda: EscalationSettings(context=["run", "dialogue"])
    )
    """Every camera step's escalator, unless escalation_per_step says else."""
    escalation_per_step: dict[str, dict[str, Any]] = {}
    """Per step id, settings that differ (heat_water: {context: [dialogue]})."""


def build_steps(
    models: TeaModels,
    settings: TeaSettings,
    *,
    temperature_c: float | None = None,
    duration_s: float | None = None,
    unit: str | None = None,
) -> dict[str, Step[Any]]:
    """The five steps, for this tea's settings (defaults when not given)."""
    temperature_c = temperature_c or settings.default_temperature_c
    duration_s = duration_s or settings.default_duration_s
    unit = unit or settings.display_unit
    watch = settings.watch
    unknown = set(settings.escalation_per_step) - WATCHED
    if unknown:
        raise ValueError(f"escalation_per_step: no step {sorted(unknown)}")

    def watching(step_id: str, **check: Any) -> WatchStep:
        return WatchStep.define(
            DEFINITIONS[step_id],
            observe=models.observe,
            decide=models.decide,
            settings=watch,
            **check,
        )

    def escalated(step: WatchStep) -> Escalated:
        helper = Escalator.define(
            DEFINITIONS[step.id], models.reason, settings.escalation
        )
        changes = settings.escalation_per_step.get(step.id, {})
        return Escalated(step, helper.but(**changes))

    heat = watching(
        "heat_water",
        needs=DisplayFacts,
        observation_model=Thermometer,
        reconcile=reconcile,
        can_assess=assessable,
    )
    return {
        "identify": ReadStep.define(
            DEFINITIONS["identify"],
            observe=models.observe,
            reason=models.reason,
            visual_model=TeaLabel,
            result_model=Brewing,
            ready=lambda tea: tea.tea_type is not None,
            interval_s=watch.interval_s,
            timeout_s=watch.timeout_s,
        ),
        "fill_water": escalated(watching("fill_water")),
        "heat_water": escalated(
            heat.format(
                minimum=target(temperature_c, unit), temperature_c=temperature_c
            ).knowing(DisplayFacts(unit=unit) if unit else None)
        ),
        "start_steeping": escalated(watching("start_steeping")),
        "steep": WaitStep.define(DEFINITIONS["steep"])
        .lasting(duration_s)
        .format(minutes=duration_s / 60),
    }


def make_tea(
    models: TeaModels, settings: TeaSettings
) -> Callable[[Run], Awaitable[None]]:
    """The procedure for SopRunner. Builds every step once first, so a
    mistake in steps.yaml or the settings fails when the app starts."""
    build_steps(models, settings)
    DEFINITIONS.check_read()

    async def procedure(run: Run) -> None:
        # 1. Identify the tea and its settings; defaults when skipped.
        found = await run.step(build_steps(models, settings)["identify"])
        brewing = found.result if found and found.result else Brewing()
        temperature_c = brewing.temperature_c or settings.default_temperature_c
        duration_s = brewing.duration_s or settings.default_duration_s
        tea = found.visual if found else None
        await run.note(
            tea=tea,
            temperature_c=temperature_c,
            duration_s=duration_s,
            settings_from="recommended" if found else "defaults",
        )
        if tea is not None:
            name = " ".join(filter(None, (tea.tea_name, tea.tea_type)))
            await run.say(
                DEFINITIONS.line(
                    "identified",
                    tea=name,
                    temperature_c=temperature_c,
                    minutes=duration_s / 60,
                )
            )
        else:
            await run.say(DEFINITIONS.line("not_identified"))

        # 2-5, for this tea's settings.
        steps = build_steps(
            models, settings, temperature_c=temperature_c, duration_s=duration_s
        )
        await run.step(steps["fill_water"])
        await run.step(steps["heat_water"])
        await run.step(steps["start_steeping"])
        await run.step(steps["steep"])

    return procedure
