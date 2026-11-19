# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""heat_water: what it runs with (HeatWaterInputs), how each frame of the
kettle's display is described (DisplayCaption), what the person may have to
say (DisplayFacts), and how a description is judged (reached_target).

The vision model describes the display; code compares the number with the
target, since that comparison is exact."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from tea import WaterTemperature
from xr_ai_kit.sop import Label


class HeatWaterInputs(BaseModel):
    """What heat_water runs with: the target, in the unit it was given in."""

    target: WaterTemperature

    @property
    def target_in_both_units(self) -> str:
        """{target_in_both_units} in heat_water's criteria: the target in
        both units, since the display may print none."""
        target = self.target
        if target.unit == "celsius":
            other = f"{target.fahrenheit:g} degrees Fahrenheit"
        else:
            other = f"{target.celsius:g} degrees Celsius"
        return f"{target.spoken} or {other}, matching the reading's unit"


class DisplayFacts(BaseModel):
    """What the heating check may need to know that a display may not show:
    from what the person says (its escalator reads the dialogue as the step
    starts and as it grows), or asked when the check gets stuck."""

    model_config = ConfigDict(extra="ignore")
    unit: Literal["celsius", "fahrenheit"] | None = Field(
        default=None,
        description="the unit the kettle's display uses, as the person says",
    )


# How one frame of the kettle's display is described. The shape guides the
# model (thinking is off), and constrained decoding holds it to the shape: it
# first says whether a lit display is in view (if not, why: printed text on a
# package is not a display), then what the display is on and where it is,
# copies its characters, gives the reading as digits only (a dot after them is
# not a decimal point), and decides whether a unit letter is printed before it
# may give one. Measured on the green-tea recording (xr_ai_kit.sop.step_types.
# watch's README): the number right on every frame, no digits read from a tea
# bag's "100% Organic", and a unit invented once in 75 frames. No class
# docstrings: they would become schema descriptions.
class DisplayReading(BaseModel):
    evidence: str = Field(
        description="what in the image shows it, or that nothing does"
    )
    shown_as: str = Field(
        description="the characters on the display, copied exactly (with any"
        " dot)"
    )
    digits: str | None = Field(
        pattern=r"^[0-9]{1,3}$",
        description="only the digits of the number, without any dot or sign;"
        " null when not all digits are legible",
    )


class UnitPrinted(BaseModel):
    unit_letter_visible: Literal[True]
    letter: Literal["C", "F"]


class NoUnitPrinted(BaseModel):
    unit_letter_visible: Literal[False]


# reached_target does not read this: it is kept because it is part of the
# caption shape that was measured (xr_ai_kit.sop.step_types.watch's README),
# and asking for it made the model look at what the number is.
class MarkedAs(BaseModel):
    evidence: str = Field(
        description="what in the image shows it, or that nothing does"
    )
    answer: str = Field(
        description="the answer, from the image only; 'not shown' when it"
        " does not show it"
    )


class DisplayInView(BaseModel):
    display_in_view: Literal[True]
    display_is_on: Literal["kettle", "thermometer"] = Field(
        description="the device the lit display is part of"
    )
    display_kind: Literal["lit digital digits"]
    where_in_image: Literal[
        "top left",
        "top",
        "top right",
        "left",
        "center",
        "right",
        "bottom left",
        "bottom",
        "bottom right",
    ] = Field(description="where the display is in the image")
    text_on_display: list[str] = Field(
        description="every character on the display, copied exactly"
    )
    reading: DisplayReading = Field(
        description="the number shown on the display"
    )
    unit: UnitPrinted | NoUnitPrinted = Field(
        description="is a unit letter (C or F) printed on the display in this"
        " image?"
    )
    marked_as: MarkedAs = Field(
        description="whether a label or indicator on the display marks the"
        " number as the water's current temperature or as a target setting,"
        " or that nothing marks it"
    )


class NoDisplayInView(BaseModel):
    display_in_view: Literal[False]
    instead: Literal[
        "no display in view",
        "only printed text or a label (a package, a mug), not a lit display",
        "the display is cut off by the image edge",
        "the display is turned away, blurred or covered",
    ] = Field(description="why no lit display can be read")
    what_is_in_view: str


class DisplayCaption(BaseModel):
    look: DisplayInView | NoDisplayInView


_UNIT_LETTERS = {"C": "celsius", "F": "fahrenheit"}


def reached_target(
    observation: DisplayCaption,
    facts: DisplayFacts | None,
    inputs: HeatWaterInputs,
) -> Label:
    """Whether the display shows the target reached: positive or negative
    once a number and its unit are known; unknown otherwise.

    The unit is the one the person stated, else a letter printed on the
    display; a number without a unit is not judged, so the escalator beside
    the step notices and asks for it."""
    look = observation.look
    if isinstance(look, NoDisplayInView) or look.reading.digits is None:
        return "unknown"
    printed = (
        _UNIT_LETTERS[look.unit.letter]
        if isinstance(look.unit, UnitPrinted)
        else None
    )
    unit = (facts.unit if facts is not None else None) or printed
    if unit is None:
        return "unknown"
    target = inputs.target
    minimum = target.celsius if unit == "celsius" else target.fahrenheit
    return "positive" if int(look.reading.digits) >= minimum else "negative"
