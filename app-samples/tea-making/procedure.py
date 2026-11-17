# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tea procedure: which steps, in what order, and what passes between.

identify        read the tea's label, derive brewing settings (a read step)
fill_water      water visible in the kettle (a watch step)
heat_water      the display reads at least the target (a watch step, on a
                schema: number, unit, current or setpoint)
start_steeping  tea touching the water in a real cup (a watch step)
steep           a countdown for the recommended time (a wait step)

build_steps makes the steps from steps.yaml's words and this app's schemas,
and wires each camera step's escalator (the side agent that asks the person
or tells them what to show), each configured on its own. The evaluation
(eval/run_steps.py) builds them the same way, so the step it evaluates is
the step that runs. The runner (SopRunner in app.py) owns the
run, skips and stops. A skipped identification falls back to the default
settings here, never inside a step.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from tea import (
    Brewing,
    DisplayFacts,
    TeaLabel,
    Thermometer,
    assessable,
    reading_of,
    reconcile,
    target,
)
from xr_ai_kit.sop import (
    EscalationSettings,
    Escalator,
    Roles,
    Run,
    Step,
    StepDefinitions,
    WatchSettings,
    WatchStep,
)

DEFINITIONS = StepDefinitions.load(
    Path(__file__).resolve().parent / "steps.yaml"
)
WATCHED = {"fill_water", "heat_water", "start_steeping"}
"""The camera steps, each with its own escalator."""


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
    roles: Roles,
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

    def escalator(step_id: str) -> Escalator:
        changes = settings.escalation_per_step.get(step_id, {})
        base = DEFINITIONS.escalator(step_id, roles.reason, settings.escalation)
        return base.but(**changes)

    def watching(step: WatchStep) -> WatchStep:
        return step.escalating(escalator(step.id))

    heat = DEFINITIONS.watch(
        "heat_water",
        roles,
        watch,
        needs=DisplayFacts,
        observation_model=Thermometer,
        reconcile=reconcile,
        can_assess=assessable,
    )
    return {
        "identify": DEFINITIONS.read(
            "identify",
            roles,
            visual_model=TeaLabel,
            result_model=Brewing,
            ready=lambda tea: tea.tea_type is not None,
            interval_s=watch.interval_s,
            timeout_s=watch.timeout_s,
        ),
        "fill_water": watching(DEFINITIONS.watch("fill_water", roles, watch)),
        "heat_water": watching(
            heat.format(
                minimum=target(temperature_c, unit), temperature_c=temperature_c
            ).knowing(DisplayFacts(unit=unit) if unit else None)
        ),
        "start_steeping": watching(
            DEFINITIONS.watch("start_steeping", roles, watch)
        ),
        "steep": DEFINITIONS.wait("steep")
        .lasting(duration_s)
        .format(minutes=duration_s / 60),
    }


def make_tea(
    roles: Roles, settings: TeaSettings
) -> Callable[[Run], Awaitable[None]]:
    """The procedure for SopRunner."""

    async def procedure(run: Run) -> None:
        # 1. Identify the tea and its settings; defaults when skipped.
        found = await run.step(build_steps(roles, settings)["identify"])
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
            roles, settings, temperature_c=temperature_c, duration_s=duration_s
        )
        if await run.step(steps["fill_water"]):
            await run.say(DEFINITIONS.line("water_seen"))
        if heated := await run.step(steps["heat_water"]):
            reading = reading_of(heated)
            await run.say(DEFINITIONS.line("hot_enough", reading=reading))
        await run.step(steps["start_steeping"])
        if await run.step(steps["steep"]):
            await run.say(DEFINITIONS.line("steeped"))

    return procedure
