# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tea procedure: its steps, what each runs with, and the order.

identify         read the tea's package into a Tea (a read step)
choose_settings  fill in the brewing settings the package does not print
                 (a transform step; it may ask the person what tea it is)
fill_water       water visible in the kettle (a watch step)
heat_water       the display reads at least the target temperature (a watch
                 step that reads the display into a Thermometer)
start_steeping   tea touching the water in a real cup (a watch step)
steep            a countdown for the steeping time (a wait step)

Each step is defined once: its words from steps.yaml, its type, models and
schemas here. A step that needs something an earlier step produced declares
it as its inputs (HeatInputs, SteepInputs), and make_tea passes it on. Each
camera step has an escalator beside it (StepWithEscalator): when the camera
alone cannot decide, it asks the person or tells them what to show. Each
step says its own lines. The evaluation (eval/run_steps.py) takes its steps
from TeaProcedure, so the step it evaluates is the step that runs.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from tea import (
    DisplayFacts,
    Tea,
    Thermometer,
    can_judge_reading,
    fahrenheit,
    use_known_unit,
)
from xr_ai_app import Role
from xr_ai_kit.sop import (
    EscalationSettings,
    ReadSettings,
    ReadStep,
    ReasoningEscalator,
    Run,
    StepDefinitions,
    StepWithEscalator,
    StructuredObservation,
    TransformStep,
    WaitInputs,
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

    read: ReadSettings = Field(default_factory=ReadSettings)
    """How the package is read."""
    watch: WatchSettings = Field(default_factory=WatchSettings)
    """How every camera step watches."""
    escalation: EscalationSettings = Field(
        default_factory=lambda: EscalationSettings(context=["run", "dialogue"])
    )
    """How every camera step's escalator works."""


class HeatInputs(BaseModel):
    """What heat_water runs with."""

    temperature_c: float

    @property
    def minimum(self) -> str:
        """{minimum} in heat_water's criteria: the target in either unit,
        since the display may print none."""
        return (
            f"{self.temperature_c:g} degrees Celsius or "
            f"{fahrenheit(self.temperature_c):g} degrees Fahrenheit, matching"
            " the reading's unit"
        )


class SteepInputs(WaitInputs):
    """What steep runs with: duration_s, and {minutes} for its words."""

    @property
    def minutes(self) -> float:
        return self.duration_s / 60


class TeaProcedure:
    """The tea procedure (make_tea, for SopRunner) and its steps.

    perceive reads camera frames; decide labels what perceive saw (fast);
    answer_with_reasoning fills in the brewing settings and is what every
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
        camera_check_models = WatchModels(perceive=perceive, decide=decide)

        self.identify = ReadStep(
            DEFINITIONS["identify"],
            perceive,
            reading_schema=Tea,
            is_enough=lambda tea: tea.tea_type is not None,
            settings=settings.read,
        )
        self.choose_settings = TransformStep(
            DEFINITIONS["choose_settings"],
            answer_with_reasoning,
            input_schema=Tea,
            output_schema=Tea,
            keep_input_values=True,
            required_fields=("water_temperature", "steep_time"),
        )
        self.fill_water = StepWithEscalator(
            WatchStep(
                DEFINITIONS["fill_water"],
                camera_check_models,
                settings=settings.watch,
            ),
            ReasoningEscalator(
                DEFINITIONS["fill_water"],
                answer_with_reasoning,
                settings.escalation,
            ),
        )
        self.heat_water = StepWithEscalator(
            WatchStep(
                DEFINITIONS["heat_water"],
                camera_check_models,
                inputs_schema=HeatInputs,
                facts_schema=DisplayFacts,
                observation=StructuredObservation(
                    schema=Thermometer,
                    apply_known_facts=use_known_unit,
                    can_judge=can_judge_reading,
                ),
                settings=settings.watch,
            ),
            ReasoningEscalator(
                DEFINITIONS["heat_water"],
                answer_with_reasoning,
                settings.escalation,
            ),
        )
        self.start_steeping = StepWithEscalator(
            WatchStep(
                DEFINITIONS["start_steeping"],
                camera_check_models,
                settings=settings.watch,
            ),
            ReasoningEscalator(
                DEFINITIONS["start_steeping"],
                answer_with_reasoning,
                settings.escalation,
            ),
        )
        self.steep = WaitStep(DEFINITIONS["steep"], inputs_schema=SteepInputs)
        DEFINITIONS.check_read()  # every section of steps.yaml has a reader

    async def make_tea(self, run: Run) -> None:
        """The procedure: read the package, fill in its settings, brew."""
        read = await run.step(self.identify)  # None when skipped
        tea = await run.step(self.choose_settings, read or Tea())
        # choose_settings cannot be skipped, and it always sets both.
        assert tea is not None
        assert tea.water_temperature is not None and tea.steep_time is not None
        temperature_c = tea.water_temperature.celsius
        duration_s = tea.steep_time.minutes * 60
        # For the conversation, until the runner records each step's inputs
        # and result itself.
        await run.note(
            tea=tea.spoken_name,
            temperature_c=temperature_c,
            duration_s=duration_s,
        )

        await run.step(self.fill_water)
        await run.step(self.heat_water, HeatInputs(temperature_c=temperature_c))
        await run.step(self.start_steeping)
        await run.step(self.steep, SteepInputs(duration_s=duration_s))
