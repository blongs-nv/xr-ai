# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tea-specific data: what is read from the camera and what is derived.

The step functions (xr_ai_kit.sop) know nothing about tea; these schemas and
the temperature check are what this app passes them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_UNREADABLE = {
    "",
    "unknown",
    "unreadable",
    "none",
    "null",
    "not visible",
    "n/a",
}


class TeaLabel(BaseModel):
    """The tea's name and type as printed on its label (None: unreadable)."""

    model_config = ConfigDict(extra="ignore")
    tea_name: str | None = Field(default=None, max_length=300)
    tea_type: str | None = Field(
        default=None,
        max_length=300,
        description="The generic tea type printed on the label (green, black,"
        " oolong, ...), not the brand; null when not readable.",
    )

    @field_validator("tea_name", "tea_type", mode="before")
    @classmethod
    def _unreadable_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in _UNREADABLE:
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator("tea_type")
    @classmethod
    def _type_has_letters(cls, value: str | None) -> str | None:
        # A kettle's number read as a "type" is not one.
        if value is None or any(c.isalpha() for c in value):
            return value
        return None


# Recommended brewing settings for the identified tea (None: not
# established). No class docstring on purpose: it would become the schema's
# description, and with it the reasoning model's constrained reply for this
# schema left temperature_c out (6 of 6 tries; 3 of 3 filled without it).
class Brewing(BaseModel):
    model_config = ConfigDict(extra="ignore")
    temperature_c: float | None = Field(
        default=None,
        gt=0,
        lt=101,
        description="the recommended water temperature in degrees Celsius",
    )
    duration_s: float | None = Field(
        default=None,
        gt=0,
        le=1800,
        description="the recommended steeping time in seconds",
    )


class Thermometer(BaseModel):
    """What a temperature display shows; a missing unit is kept as None."""

    model_config = ConfigDict(extra="ignore")
    value: float | None = None
    unit_text: str | None = Field(
        default=None,
        max_length=40,
        description="The unit letter printed beside the number, copied"
        " exactly (°C, °F, C, F); null when the display shows no C or F"
        " letter (a bare number, or only a degree sign).",
    )
    role: Literal["current", "setpoint", "unknown"] = Field(
        default="unknown",
        description="current only when a label or indicator shows a measured"
        " temperature, setpoint only when it shows a target; else unknown.",
    )

    @property
    def unit(self) -> Literal["celsius", "fahrenheit"] | None:
        """The unit from the printed text only, never from the number."""
        text = "".join((self.unit_text or "").split()).casefold()
        return {
            "°c": "celsius",
            "c": "celsius",
            "celsius": "celsius",
            "°f": "fahrenheit",
            "f": "fahrenheit",
            "fahrenheit": "fahrenheit",
        }.get(text)

    @property
    def reading(self) -> str:
        """The reading as spoken ("176 °F"; "the target" without a value)."""
        if not self.value:
            return "the target"
        return f"{self.value:g} {self.unit_text or ''}".strip()


class DisplayFacts(BaseModel):
    """What the heating check may need to know that a display may not show:
    learned from the person when the check gets stuck, or set in app.yaml."""

    model_config = ConfigDict(extra="ignore")
    unit: Literal["celsius", "fahrenheit"] | None = Field(
        default=None,
        description="the unit the kettle's display uses, as the person says",
    )


_UNIT_TEXT = {"celsius": "°C", "fahrenheit": "°F"}


def reconcile(reading: Thermometer, known: DisplayFacts | None) -> Thermometer:
    """The reading with the known unit in place of the unit the model read.

    A unit the person stated (or app.yaml set) wins: on a display that prints
    none, a read unit is invented, and the decision must not see it.
    """
    if known is None or known.unit is None:
        return reading
    return reading.model_copy(update={"unit_text": _UNIT_TEXT[known.unit]})


def assessable(reading: Thermometer, known: DisplayFacts | None) -> bool:
    """Whether a reading can be compared with the target at all.

    A known unit (the person said it, or app.yaml set it) wins over the unit
    the model says it reads: on a display that prints none, a read unit is
    invented. A setpoint, or no unit at all (neither known nor read), is not
    a measurement to judge; the frame then counts as unknown.
    """
    if reading.value is None or reading.role == "setpoint":
        return False
    unit = known.unit if known is not None else None
    return (unit or reading.unit) is not None


def fahrenheit(celsius: float) -> float:
    return celsius * 9 / 5 + 32


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
