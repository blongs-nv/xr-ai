# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tea procedure: its steps, and the order it runs them in.

identify        read the tea's label and derive brewing settings (a read step)
fill_water      water visible in the kettle (a watch step)
heat_water      the display reads at least the target temperature (a watch
                step that reads the display into a Thermometer)
start_steeping  tea touching the water in a real cup (a watch step)
steep           a countdown for the steeping time (a wait step)

Each step's words are in steps.yaml; its type, models and schemas are here.
Each camera step has an escalator beside it (StepWithEscalator): when the
camera alone cannot decide, it asks the person or tells them what to show.
heat_water and steep depend on the brewing settings, so they are built once
the tea is identified. Each step says its own lines; the procedure says only
the settings it decided on. The evaluation (eval/run_steps.py) takes its
steps from TeaProcedure too, so the step it evaluates is the step that runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from tea import (
    Brewing,
    DisplayFacts,
    TeaLabel,
    Thermometer,
    can_judge_reading,
    target_in_display_unit,
    use_known_unit,
)
from xr_ai_app import Role
from xr_ai_kit.sop import (
    EscalationSettings,
    ReadModels,
    ReadSettings,
    ReadStep,
    ReasoningEscalator,
    Run,
    StepDefinitions,
    StepWithEscalator,
    StructuredObservation,
    WaitStep,
    WatchModels,
    WatchSettings,
    WatchStep,
)

DEFINITIONS = StepDefinitions.load(
    Path(__file__).resolve().parent / "steps.yaml"
)


class TeaSettings(BaseModel):
    """The procedure's settings (app.yaml's settings.tea)."""

    default_temperature_c: float = 85.0
    default_duration_s: float = 180.0
    display_unit: Literal["celsius", "fahrenheit"] | None = None
    """The kettle display's unit, when it prints none."""
    watch: WatchSettings = Field(default_factory=WatchSettings)
    """How every camera step watches."""
    escalation: EscalationSettings = Field(
        default_factory=lambda: EscalationSettings(context=["run", "dialogue"])
    )
    """How every camera step's escalator works."""


class TeaProcedure:
    """The tea procedure (make_tea, for SopRunner) and its steps.

    perceive reads camera frames; decide labels what perceive saw (fast);
    answer_with_reasoning derives the brewing settings and is what every
    escalator reasons with. These are app.yaml's model roles.
    """

    def __init__(
        self,
        *,
        perceive: Role,
        decide: Role,
        answer_with_reasoning: Role,
        settings: TeaSettings,
    ) -> None:
        self.settings = settings
        self.camera_check_models = WatchModels(perceive=perceive, decide=decide)
        self.escalation_model = answer_with_reasoning

        self.identify = ReadStep(
            DEFINITIONS["identify"],
            ReadModels(perceive=perceive, reason=answer_with_reasoning),
            reading_schema=TeaLabel,
            derived_schema=Brewing,
            is_enough=lambda label: label.tea_type is not None,
            settings=ReadSettings(
                interval_s=settings.watch.interval_s,
                timeout_s=settings.watch.timeout_s,
            ),
        )
        self.fill_water = StepWithEscalator(
            WatchStep(
                DEFINITIONS["fill_water"],
                self.camera_check_models,
                settings=settings.watch,
            ),
            ReasoningEscalator(
                DEFINITIONS["fill_water"],
                self.escalation_model,
                settings.escalation,
            ),
        )
        self.start_steeping = StepWithEscalator(
            WatchStep(
                DEFINITIONS["start_steeping"],
                self.camera_check_models,
                settings=settings.watch,
            ),
            ReasoningEscalator(
                DEFINITIONS["start_steeping"],
                self.escalation_model,
                settings.escalation,
            ),
        )
        # heat_water and steep are built during the run, once the brewing
        # settings are known. Build them once now too, so a mistake in
        # steps.yaml fails when the app starts, not halfway through the tea.
        self.heat_water(settings.default_temperature_c, settings.display_unit)
        self.steep(settings.default_duration_s)
        DEFINITIONS.check_read()

    def heat_water(
        self,
        temperature_c: float,
        display_unit: Literal["celsius", "fahrenheit"] | None,
    ) -> StepWithEscalator:
        """The heating step for this temperature. The display may print no
        unit: then a known unit (app.yaml, or what the person answers) is
        needed before a reading can be judged."""
        return StepWithEscalator(
            WatchStep(
                DEFINITIONS["heat_water"],
                self.camera_check_models,
                values={
                    "temperature_c": temperature_c,
                    "minimum": target_in_display_unit(
                        temperature_c, display_unit
                    ),
                },
                facts_schema=DisplayFacts,
                known_facts=(
                    DisplayFacts(unit=display_unit) if display_unit else None
                ),
                observation=StructuredObservation(
                    schema=Thermometer,
                    apply_known_facts=use_known_unit,
                    can_judge=can_judge_reading,
                ),
                settings=self.settings.watch,
            ),
            ReasoningEscalator(
                DEFINITIONS["heat_water"],
                self.escalation_model,
                self.settings.escalation,
            ),
        )

    def steep(self, duration_s: float) -> WaitStep:
        """The steeping countdown for this duration."""
        return WaitStep(
            DEFINITIONS["steep"],
            duration_s,
            values={"minutes": duration_s / 60},
        )

    async def make_tea(self, run: Run) -> None:
        """The procedure: identify the tea, decide the settings, brew it."""
        identified = await run.step(self.identify)  # None when skipped

        # The settings: recommended for this tea, or the defaults.
        recommended = (identified and identified.derived) or Brewing()
        temperature_c = (
            recommended.temperature_c or self.settings.default_temperature_c
        )
        duration_s = recommended.duration_s or self.settings.default_duration_s
        settings_from = "recommended" if identified else "defaults"
        await run.note(
            tea=identified.reading if identified else None,
            temperature_c=temperature_c,
            duration_s=duration_s,
            settings_from=settings_from,
        )
        await run.say(
            DEFINITIONS.procedure_line(
                settings_from,
                temperature_c=temperature_c,
                minutes=duration_s / 60,
            )
        )

        await run.step(self.fill_water)
        await run.step(
            self.heat_water(temperature_c, self.settings.display_unit)
        )
        await run.step(self.start_steeping)
        await run.step(self.steep(duration_s))
