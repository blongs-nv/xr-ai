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
never inside a step. The three visual checks may ask the person when they
get stuck, as steps.yaml's escalation guidance allows; the heating check
also declares what it may need to know that the display does not show
(DisplayFacts), seeded from app.yaml when the kettle's unit is configured.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict
from tea import (
    Brewing,
    DisplayFacts,
    TeaLabel,
    Thermometer,
    assessable,
    fahrenheit,
)
from xr_ai_app import Role
from xr_ai_kit.sop import (
    Check,
    Criteria,
    Escalation,
    Facts,
    Run,
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
            can_assess=assessable,
        ),
        "start_steeping": steps.start_steeping.to_check(),
    }


def known_facts(unit: str | None) -> Facts | None:
    """What the heating check knows at the start: the configured unit."""
    return Facts(DisplayFacts(unit=unit)) if unit else None


def make_tea(
    *,
    observe: Role,
    answer: Role,
    decide: Role,
    steps: Steps,
    settings: TeaSettings,
) -> Callable[[Run], Awaitable[None]]:
    """The procedure for SopRunner, bound to its roles, wording and settings."""
    say = steps.say

    async def procedure(run: Run) -> None:
        look = partial(
            verify,
            run.p.frames,
            observe=observe,
            decide=decide,
            escalation=Escalation(answer, run.ask),
            required=settings.required,
            window=settings.window,
            interval_s=settings.interval_s,
            timeout_s=settings.timeout_s,
        )

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
        if await run.step(
            "fill_water",
            lambda: look(
                built["fill_water"], on_state=run.report("fill_water")
            ),
            say=say["fill_water"],
        ):
            await run.say(say["water_seen"])

        # 3. The display reads at least the target.
        heated = await run.step(
            "heat_water",
            lambda: look(
                built["heat_water"],
                facts=known_facts(unit),
                on_state=run.report("heat_water"),
            ),
            say=say["heat_water"].format(temperature_c=temperature_c),
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
        await run.step(
            "start_steeping",
            lambda: look(
                built["start_steeping"],
                on_state=run.report("start_steeping"),
            ),
            say=say["start_steeping"],
        )
        if await run.step(
            "steep",
            lambda: timer(duration_s, on_state=run.report("steep")),
            say=say["steep"].format(minutes=duration_s / 60),
        ):
            await run.say(say["steeped"])

    return procedure
