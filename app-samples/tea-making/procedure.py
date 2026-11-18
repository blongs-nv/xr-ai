# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tea procedure: make_tea runs the steps in order and passes each the
data it needs; TeaProcedure builds them.

identify         read the tea's package into a Tea (a read step)
choose_settings  fill in the brewing settings the package does not print
                 (a transform step; it may ask the person what tea it is)
fill_water       water visible in the kettle (a watch step)
heat_water       the display reads at least the target temperature (a watch
                 step that describes the display and judges it in code)
start_steeping   tea touching the water in a real cup (a watch step)
steep            a countdown for the steeping time (a wait step)

Each step's folder (steps/<step>/) has its words (words.yaml), how the person
can help when its camera check is stuck (escalation.yaml), and its own types
(schemas.py). What passes between steps is in tea.py.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from steps.heat_water.schemas import (
    DisplayCaption,
    DisplayFacts,
    HeatWaterInputs,
    reached_target,
)
from steps.identify.schemas import TeaPackageLook, tea_from_look
from steps.steep.schemas import SteepInputs
from tea import Tea
from xr_ai_app import Role
from xr_ai_kit.sop import (
    EscalationSettings,
    EscalationWords,
    ReadSettings,
    ReadStep,
    ReadWords,
    ReasoningEscalator,
    Run,
    StepWithEscalator,
    StepWords,
    StructuredObservation,
    TransformStep,
    TransformWords,
    WaitStep,
    WatchModels,
    WatchSettings,
    WatchStep,
    WatchWords,
    load_words,
)

STEPS = Path(__file__).resolve().parent / "steps"


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


class TeaProcedure:
    """The tea procedure (make_tea, for SopRunner) and its steps.

    perceive reads camera frames, without reasoning; decide labels what
    perceive saw (fast); answer_with_reasoning fills in the brewing settings
    and is what every escalator reasons with. These are yaml/models.json's
    roles.
    """

    async def make_tea(self, run: Run) -> None:
        """Read the package, fill in its settings, brew."""
        read = await run.step(self.identify)  # None when skipped
        tea = await run.step(self.choose_settings, read or Tea())
        # choose_settings cannot be skipped, and it always sets both.
        assert tea is not None
        assert tea.water_temperature is not None and tea.steep_time is not None
        duration_s = tea.steep_time.minutes * 60
        # For the conversation, until the runner records each step's inputs
        # and result itself.
        await run.note(
            tea=tea.spoken_name,
            temperature_c=tea.water_temperature.celsius,
            duration_s=duration_s,
        )

        await run.step(self.fill_water)
        await run.step(
            self.heat_water, HeatWaterInputs(target=tea.water_temperature)
        )
        await run.step(self.start_steeping)
        await run.step(self.steep, SteepInputs(duration_s=duration_s))

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
            "identify",
            load_words(STEPS / "identify" / "words.yaml", ReadWords),
            perceive,
            reply_schema=TeaPackageLook,
            extract=tea_from_look,
            is_enough=lambda tea: tea.tea_type is not None,
            settings=settings.read,
        )
        self.choose_settings = TransformStep(
            "choose_settings",
            load_words(
                STEPS / "choose_settings" / "words.yaml", TransformWords
            ),
            answer_with_reasoning,
            input_schema=Tea,
            output_schema=Tea,
            keep_input_values=True,
            required_fields=("water_temperature", "steep_time"),
        )
        self.fill_water = StepWithEscalator(
            WatchStep(
                "fill_water",
                load_words(STEPS / "fill_water" / "words.yaml", WatchWords),
                camera_check_models,
                settings=settings.watch,
            ),
            ReasoningEscalator(
                load_words(
                    STEPS / "fill_water" / "escalation.yaml", EscalationWords
                ),
                answer_with_reasoning,
                settings.escalation,
            ),
        )
        self.heat_water = StepWithEscalator(
            WatchStep(
                "heat_water",
                load_words(STEPS / "heat_water" / "words.yaml", WatchWords),
                camera_check_models,
                inputs_schema=HeatWaterInputs,
                facts_schema=DisplayFacts,
                observation=StructuredObservation(
                    schema=DisplayCaption, judge=reached_target
                ),
                settings=settings.watch,
            ),
            ReasoningEscalator(
                load_words(
                    STEPS / "heat_water" / "escalation.yaml", EscalationWords
                ),
                answer_with_reasoning,
                settings.escalation,
            ),
        )
        self.start_steeping = StepWithEscalator(
            WatchStep(
                "start_steeping",
                load_words(STEPS / "start_steeping" / "words.yaml", WatchWords),
                camera_check_models,
                settings=settings.watch,
            ),
            ReasoningEscalator(
                load_words(
                    STEPS / "start_steeping" / "escalation.yaml",
                    EscalationWords,
                ),
                answer_with_reasoning,
                settings.escalation,
            ),
        )
        self.steep = WaitStep(
            "steep",
            load_words(STEPS / "steep" / "words.yaml", StepWords),
            inputs_schema=SteepInputs,
        )
        self.steps = (
            self.identify,
            self.choose_settings,
            self.fill_water,
            self.heat_water,
            self.start_steeping,
            self.steep,
        )
        """In the order make_tea runs them."""

    @property
    def plan(self) -> list[tuple[str, str]]:
        """(step id, what the person does) in order, for the conversation."""
        return [
            (step.id, step.words.does) for step in self.steps if step.words.does
        ]
