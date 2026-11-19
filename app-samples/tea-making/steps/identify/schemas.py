# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How identify reads the tea's package: the reply the model fills in for one
frame (TeaPackageLook), and how a reply becomes the Tea kept across frames
(tea_from_look)."""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
)
from tea import SteepTime, Tea, TeaType, WaterTemperature

# No class docstrings in this file: they would become schema descriptions,
# and with one a model left a field out of its reply.
V = TypeVar("V")


class Quoted(BaseModel, Generic[V]):
    quote: str = Field(
        description="the exact words on the tea's package it comes from, copied"
    )
    value: V


# Each choice below is a gate: the model writes its yes/no first, and on no
# it cannot write any of the fields the yes allows.
class BrewingDirectionsPrinted(BaseModel):
    brewing_directions_visible: Literal[True]
    quote: str = Field(
        description="the brewing directions as printed, copied word for word"
    )
    water_temperature: WaterTemperature | None = None
    steep_time: SteepTime | None = None

    # The schema the model fills in has no bounds, so it can write a misread
    # value (1755 degrees). Such a value counts as empty, as in Tea, so the
    # brand and name read from the same frame are kept.
    @field_validator("water_temperature", "steep_time", mode="wrap")
    @classmethod
    def _implausible_is_none(
        cls, value: Any, handler: ValidatorFunctionWrapHandler
    ) -> Any:
        try:
            return handler(value)
        except ValidationError:
            return None


class NoBrewingDirections(BaseModel):
    brewing_directions_visible: Literal[False]


# A tea package in view: its words, copied first; then the brand, name and
# kind of tea, each with the words it comes from (the kind may be clear from
# the words: Premium Green Tea is green); then whether brewing directions are
# printed, before any may be given.
class PackageInView(BaseModel):
    tea_package_in_view: Literal[True]
    text_on_package: list[str] = Field(
        description="every legible word or phrase on the tea's own package,"
        " copied"
    )
    brand: Quoted[str] | None = Field(
        default=None, description="the brand, e.g. Bigelow"
    )
    name: Quoted[str] | None = Field(
        default=None, description="the product name, e.g. Premium Green Tea"
    )
    tea_type: Quoted[TeaType] | None = Field(
        default=None,
        description="the kind of tea, printed or clear from the words",
    )
    directions: BrewingDirectionsPrinted | NoBrewingDirections


class NoPackage(BaseModel):
    tea_package_in_view: Literal[False]
    what_is_in_view: str = Field(
        description="what the camera shows instead, in a few words"
    )


# How one frame of the tea's package is read. The shape guides the model
# (thinking is off), and constrained decoding holds it to the shape: it first
# decides whether a tea package is in view at all (if not, nothing can be
# read), copies the package's words, gives the brand, name and kind of tea
# each with the words it comes from, and decides whether brewing directions
# are printed before it may give any. Measured on the green-tea recording
# (xr_ai_kit.sop.step_types.read's README): no value from frames without a
# package, and no brewing settings invented from what the model knows.
# tea_from_look turns a reply into the Tea kept across frames. No class
# docstrings (see the top of the file).
class TeaPackageLook(BaseModel):
    look: PackageInView | NoPackage


def tea_from_look(reply: TeaPackageLook) -> Tea:
    """What one look at the package read, as a Tea (empty when no package was
    in view)."""
    seen = reply.look
    if isinstance(seen, NoPackage):
        return Tea()
    read: dict[str, Any] = {
        field: quoted.value if (quoted := getattr(seen, field)) else None
        for field in ("brand", "name", "tea_type")
    }
    if isinstance(seen.directions, BrewingDirectionsPrinted):
        read["water_temperature"] = seen.directions.water_temperature
        read["steep_time"] = seen.directions.steep_time
    return Tea(**read)
