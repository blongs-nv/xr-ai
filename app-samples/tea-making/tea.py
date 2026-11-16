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


class Brewing(BaseModel):
    """Brewing settings for the identified tea (None when not established)."""

    model_config = ConfigDict(extra="ignore")
    temperature_c: float | None = Field(default=None, gt=0, lt=101)
    duration_s: float | None = Field(default=None, gt=0, le=1800)


class Thermometer(BaseModel):
    """What a temperature display shows; a missing unit is kept as None."""

    model_config = ConfigDict(extra="ignore")
    value: float | None = None
    unit_text: str | None = Field(
        default=None,
        max_length=40,
        description="The unit exactly as printed next to the number (°C, °F,"
        " C, F); null when no unit can be read.",
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


def assessable(reading: Thermometer, unit: str | None) -> bool:
    """Whether a reading can be compared with the target at all.

    ``unit`` is the display's unit when the app knows it (a setting), used
    when none is printed. A setpoint, a conflicting unit or no unit at all is
    not a measurement to judge; the frame then counts as unknown.
    """
    if reading.value is None or reading.role == "setpoint":
        return False
    if reading.unit is not None and unit is not None and reading.unit != unit:
        return False
    return (reading.unit or unit) is not None


def fahrenheit(celsius: float) -> float:
    return celsius * 9 / 5 + 32
