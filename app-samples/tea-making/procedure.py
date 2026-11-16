# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tea procedure, front to back: ordinary Python calls and their results.

identify    read the tea's label, then recommend settings (identify)
fill_water  water visible in the kettle, 2 of the last 4 frames (verify)
heat_water  the display reads at least the target temperature (verify,
            on a schema: number, unit, current or setpoint)
start_steeping  tea touching the water in a real cup (verify)
steep       a countdown for the recommended time (timer)

Each step is a call of an xr_ai_kit.sop function with its inputs; the
runner (SopRunner in app.py) owns the run, skips and stops. A skipped
identification falls back to the default settings here, in the procedure,
never inside a step. Each visual step is a fast check (verify: no
reasoning, never asks) and, beside it, a side escalation (gather): every
review_every_s a reasoning model looks at the latest frame and the guide's
state and, following steps.yaml's escalation guidance, may ask the person
something the camera cannot establish. Answers go into the step's context,
which the fast check reads on every frame; the heating check declares the
typed fact it may need (DisplayFacts), seeded from app.yaml when the
kettle's unit is configured.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict
from tea import (
    Brewing,
    DisplayFacts,
    TeaLabel,
    Thermometer,
    assessable,
    fahrenheit,
    reconcile,
)
from xr_ai_app import Participant, Role
from xr_ai_kit.sop import (
    Check,
    Criteria,
    Run,
    StepContext,
    gather,
    identify,
    timer,
    verify,
)


class CheckCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: str
    positive: str
    negative: str
    unknown: str


class CheckSpec(BaseModel):
    """One check's wording in steps.yaml."""

    model_config = ConfigDict(extra="forbid")
    question: str
    criteria: CheckCriteria
    escalation: str | None = None

    def to_check(
        self, values: dict[str, object] | None = None, **code
    ) -> Check:
        """The kit's Check: this wording, placeholders filled from values.

        What only code declares (needs, observation_model, can_assess) is
        passed as keywords.
        """
        c = self.criteria
        criteria = Criteria(c.instructions, c.positive, c.negative, c.unknown)
        check = Check(self.question, criteria, self.escalation, **code)
        return check.format(**(values or {}))


class Identify(BaseModel):
    question: str
    instructions: str


class Steps(BaseModel):
    """steps.yaml: the wording of every check and message."""

    model_config = ConfigDict(extra="forbid")
    identify: Identify
    fill_water: CheckSpec
    heat_water: CheckSpec
    start_steeping: CheckSpec
    say: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> Steps:
        return cls.model_validate(yaml.safe_load(path.read_text("utf-8")))


class TeaSettings(BaseModel):
    """The procedure's settings (app.yaml's settings.tea)."""

    default_temperature_c: float = 85.0
    default_duration_s: float = 180.0
    display_unit: str | None = None  # "celsius" / "fahrenheit": the kettle's
    required: int = 2  # positives needed ...
    window: int = 4  # ... among the last this many decidable frames
    interval_s: float = 2.0
    timeout_s: float = 30.0
    review_every_s: float = 20.0  # the side escalation's pace


def target(temperature_c: float, unit: str | None) -> str:
    """The minimum as the criteria say it, in the display's unit if known."""
    if unit == "fahrenheit":
        return f"{fahrenheit(temperature_c):g} degrees Fahrenheit"
    if unit == "celsius":
        return f"{temperature_c:g} degrees Celsius"
    return (
        f"{temperature_c:g} degrees Celsius or "
        f"{fahrenheit(temperature_c):g} degrees Fahrenheit, matching the"
        " reading's unit"
    )


def checks(
    steps: Steps, *, temperature_c: float, unit: str | None
) -> dict[str, Check]:
    """The three visual checks, as the procedure runs them (the evaluation
    builds them here too): steps.yaml's wording plus what code declares."""
    return {
        "fill_water": steps.fill_water.to_check(),
        "heat_water": steps.heat_water.to_check(
            {"minimum": target(temperature_c, unit)},
            needs=DisplayFacts,
            observation_model=Thermometer,
            reconcile=reconcile,
            can_assess=assessable,
        ),
        "start_steeping": steps.start_steeping.to_check(),
    }


def make_tea(
    *,
    observe: Role,
    answer: Role,
    decide: Role,
    review: Role,
    steps: Steps,
    settings: TeaSettings,
    situation: Callable[[Participant], Awaitable[str]] | None = None,
) -> Callable[[Run], Awaitable[None]]:
    """The procedure for SopRunner, bound to its roles, wording and settings.

    observe and decide run the fast checks; review (a reasoning model that
    takes images) and answer (reasoning, text) run the side escalation;
    situation tells it what is going on (the guide's state, the dialogue).
    """
    say = steps.say

    async def procedure(run: Run) -> None:
        def watched(check: Check, step_id: str, known: Any = None):
            """A step's fast check and its side escalation, sharing one
            context: the check reads it, the escalation adds to it."""
            context = StepContext(known)

            async def what_is_going_on() -> str:
                return await situation(run.p) if situation else ""

            def fast():
                return verify(
                    run.p.frames,
                    check,
                    observe=observe,
                    decide=decide,
                    context=context,
                    required=settings.required,
                    window=settings.window,
                    interval_s=settings.interval_s,
                    timeout_s=settings.timeout_s,
                    on_state=run.report(step_id),
                )

            def side():
                return gather(
                    context,
                    check,
                    run.p.frames,
                    role=review,
                    ask=run.ask,
                    reader=answer,
                    situation=what_is_going_on,
                    every_s=settings.review_every_s,
                )

            return fast, [side]

        # 1. Identify the tea and its settings; defaults when skipped.
        found = await run.step(
            "identify",
            lambda: identify(
                run.p.frames,
                observe=observe,
                answer=answer,
                question=steps.identify.question,
                visual_model=TeaLabel,
                result_model=Brewing,
                instructions=steps.identify.instructions,
                ready=lambda tea: tea.tea_type is not None,
                interval_s=settings.interval_s,
                timeout_s=settings.timeout_s,
                on_state=run.report("identify"),
            ),
            say=say["identify"],
        )
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
        if found and tea is not None:
            name = " ".join(filter(None, (tea.tea_name, tea.tea_type)))
            await run.say(
                say["identified"].format(
                    tea=name,
                    temperature_c=temperature_c,
                    minutes=duration_s / 60,
                )
            )
        else:
            await run.say(say["not_identified"])

        # The visual checks, now that the target temperature is known.
        unit = settings.display_unit
        built = checks(steps, temperature_c=temperature_c, unit=unit)

        # 2. Water in the kettle.
        fast, side = watched(built["fill_water"], "fill_water")
        if await run.step(
            "fill_water", fast, say=say["fill_water"], alongside=side
        ):
            await run.say(say["water_seen"])

        # 3. The display reads at least the target.
        known = DisplayFacts(unit=unit) if unit else None
        fast, side = watched(built["heat_water"], "heat_water", known)
        heated = await run.step(
            "heat_water",
            fast,
            say=say["heat_water"].format(temperature_c=temperature_c),
            alongside=side,
        )
        if heated:
            reading = next(
                (
                    j.parsed
                    for j in reversed(heated.judgments)
                    if j.label == "positive" and j.parsed is not None
                ),
                None,
            )
            shown = "the target"
            if isinstance(reading, Thermometer) and reading.value is not None:
                shown = f"{reading.value:g} {reading.unit_text or ''}".strip()
            await run.say(say["hot_enough"].format(reading=shown))

        # 4. Tea in the water; 5. the countdown starts now either way.
        fast, side = watched(built["start_steeping"], "start_steeping")
        await run.step(
            "start_steeping", fast, say=say["start_steeping"], alongside=side
        )
        if await run.step(
            "steep",
            lambda: timer(duration_s, on_state=run.report("steep")),
            say=say["steep"].format(minutes=duration_s / 60),
        ):
            await run.say(say["steeped"])

    return procedure
