# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tea-specific data: what is read from the camera and what is filled in.

The step types (xr_ai_kit.sop) know nothing about tea; these schemas and the
temperature checks are what procedure.py gives them.
"""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema

_UNREADABLE = {
    "",
    "unknown",
    "unreadable",
    "none",
    "null",
    "not visible",
    "n/a",
}


TeaType = Literal["green", "black", "oolong", "white", "pu-erh", "herbal"]


# A water temperature, in the unit it is given in. The schemas here carry no
# class docstrings on purpose: a docstring becomes the schema's description,
# and with one the reasoning model's constrained reply for the old brewing
# schema left the temperature out (6 of 6 tries; 3 of 3 filled without it).
# The meaning is in the field descriptions.
class WaterTemperature(BaseModel):
    model_config = ConfigDict(extra="ignore")
    value: float = Field(description="the number")
    unit: Literal["celsius", "fahrenheit"] = Field(
        description="the unit it is given in"
    )

    @model_validator(mode="after")
    def _plausible_for_brewing(self) -> WaterTemperature:
        if not 40 <= self.celsius <= 100:
            raise ValueError(f"{self.spoken} is not a brewing temperature")
        return self

    @property
    def celsius(self) -> float:
        if self.unit == "celsius":
            return self.value
        return round((self.value - 32) * 5 / 9, 1)

    @property
    def spoken(self) -> str:
        """As said aloud ("175 degrees Fahrenheit")."""
        return f"{self.value:g} degrees {self.unit.capitalize()}"


# A steeping time as a range in minutes, as packages print it ("2-3
# minutes"); a single time has both ends the same.
class SteepTime(BaseModel):
    model_config = ConfigDict(extra="ignore")
    minimum_minutes: float = Field(
        gt=0, le=20, description="the shortest steeping time, in minutes"
    )
    maximum_minutes: float = Field(
        gt=0,
        le=20,
        description="the longest steeping time, in minutes; the same as the"
        " shortest for a single time",
    )

    @model_validator(mode="after")
    def _ordered(self) -> SteepTime:
        if self.maximum_minutes < self.minimum_minutes:
            raise ValueError("the longest steeping time is under the shortest")
        return self

    @property
    def minutes(self) -> float:
        """The middle of the range."""
        return (self.minimum_minutes + self.maximum_minutes) / 2

    @property
    def spoken(self) -> str:
        """As said aloud ("2 to 3 minutes")."""
        if self.minimum_minutes == self.maximum_minutes:
            return f"{self.minimum_minutes:g} minutes"
        return f"{self.minimum_minutes:g} to {self.maximum_minutes:g} minutes"


# What is known about the tea being brewed: what its package says (the read
# step) and the brewing settings it does not print (the transform step).
# Every field may be empty until then; a value that does not fit (a misread
# 1755 degrees, a type that is a brand) counts as empty, so it is filled in
# rather than failing the whole reading.
class Tea(BaseModel):
    model_config = ConfigDict(extra="ignore")
    brand: str | None = Field(
        default=None, max_length=100, description="the brand, e.g. Bigelow"
    )
    name: str | None = Field(
        default=None,
        max_length=200,
        description="the product name, e.g. Premium Green Tea",
    )
    tea_type: TeaType | None = Field(
        default=None, description="the kind of tea"
    )
    water_temperature: WaterTemperature | None = Field(
        default=None, description="the water temperature to brew at"
    )
    steep_time: SteepTime | None = Field(
        default=None, description="how long to steep"
    )

    @field_validator("brand", "name", mode="before")
    @classmethod
    def _unreadable_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip().lower() in _UNREADABLE:
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator("tea_type", mode="before")
    @classmethod
    def _kind_of_tea(cls, value: Any) -> Any:
        # "Green tea" is green; a word that is no kind of tea is empty.
        if not isinstance(value, str):
            return value
        kind = value.strip().lower().removesuffix(" tea").strip()
        return kind if kind in TeaType.__args__ else None

    @field_validator("water_temperature", "steep_time", mode="wrap")
    @classmethod
    def _implausible_is_none(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> Any:
        try:
            return handler(value)
        except ValidationError:
            return None

    @property
    def spoken_name(self) -> str:
        """The tea as said aloud ("Bigelow Premium Green Tea")."""
        what = self.name or (f"{self.tea_type} tea" if self.tea_type else None)
        return " ".join(filter(None, (self.brand, what))) or "an unnamed tea"


# How the package is read: each field says where its value comes from, and
# the schema allows a value only when the source is the package's own text.
# Constrained decoding then leaves the model two shapes per field,
# {source: tea_package_text, value: ...} or {source: other_text |
# own_knowledge | not_visible, value: null}: deciding the source is a small
# chain of thought the schema forces (thinking is off), and a number on the
# kettle's display cannot be written as a printed brewing temperature.
# No class docstrings: they would become schema descriptions (see above).
V = TypeVar("V")


class OnPackage(BaseModel, Generic[V]):
    source: Literal["tea_package_text"] = Field(
        description="the value is legibly printed on the tea's own box, bag"
        " or tag"
    )
    value: V


class NotOnPackage(BaseModel):
    source: Literal["other_text", "own_knowledge", "not_visible"] = Field(
        description="other_text: printed on something else (a kettle's"
        " display or label, another object); own_knowledge: known, not read;"
        " not_visible: not shown"
    )
    value: None = None


_PRINTED_FIELDS = (
    "brand",
    "name",
    "tea_type",
    "water_temperature",
    "steep_time",
)


class TeaPackageReading(BaseModel):
    model_config = ConfigDict(extra="ignore")
    brand: OnPackage[str] | NotOnPackage | SkipJsonSchema[None] = Field(
        description="the brand, e.g. Bigelow"
    )
    name: OnPackage[str] | NotOnPackage | SkipJsonSchema[None] = Field(
        description="the product name, e.g. Premium Green Tea"
    )
    tea_type: OnPackage[TeaType] | NotOnPackage | SkipJsonSchema[None] = Field(
        description="the kind of tea"
    )
    water_temperature: (
        OnPackage[WaterTemperature] | NotOnPackage | SkipJsonSchema[None]
    ) = Field(description="the water temperature to brew at")
    steep_time: OnPackage[SteepTime] | NotOnPackage | SkipJsonSchema[None] = (
        Field(description="how long to steep")
    )

    @field_validator(*_PRINTED_FIELDS, mode="wrap")
    @classmethod
    def _nothing_read_is_none(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> Any:
        # A field with no value (not on the package, or not plausible) is
        # None, so merging frames keeps what an earlier frame read.
        try:
            read = handler(value)
        except ValidationError:
            return None
        return read if isinstance(read, OnPackage) else None

    @property
    def tea(self) -> Tea:
        """What was read, as a Tea."""
        values: dict[str, Any] = {}
        for field in _PRINTED_FIELDS:
            read = getattr(self, field)
            values[field] = read.value if read is not None else None
        return Tea(**values)


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
    learned from the person when the check gets stuck (its escalator asks)."""

    model_config = ConfigDict(extra="ignore")
    unit: Literal["celsius", "fahrenheit"] | None = Field(
        default=None,
        description="the unit the kettle's display uses, as the person says",
    )


_UNIT_TEXT = {"celsius": "°C", "fahrenheit": "°F"}


def use_known_unit(
    reading: Thermometer, known: DisplayFacts | None
) -> Thermometer:
    """The reading with the known unit in place of the unit the model read.

    A unit the person stated wins: on a display that prints none, a read unit
    is invented, and the decision must not see it.
    """
    if known is None or known.unit is None:
        return reading
    return reading.model_copy(update={"unit_text": _UNIT_TEXT[known.unit]})


def can_judge_reading(reading: Thermometer, known: DisplayFacts | None) -> bool:
    """Whether a reading can be compared with the target at all.

    A unit the person stated wins over the unit the model says it reads: on a
    display that prints none, a read unit is invented. A setpoint, or no unit
    at all (neither known nor read), is not a measurement to judge; the frame
    then counts as unknown.
    """
    if reading.value is None or reading.role == "setpoint":
        return False
    unit = known.unit if known is not None else None
    return (unit or reading.unit) is not None


def fahrenheit(celsius: float) -> float:
    return celsius * 9 / 5 + 32
