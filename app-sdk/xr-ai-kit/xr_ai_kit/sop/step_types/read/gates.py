# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gates for a reply schema: a yes/no the model decides first, before it may fill anything the yes allows.

A gate is a choice between two shapes, ``{flag: false, ...otherwise}`` or ``{flag: true, ...then's fields}``; with
constrained decoding the model writes the flag first, and on false it cannot write any of ``then``'s fields. Gate on
something plainly visible (is the package in view, are directions printed), not on the model's judgement of its own
reading (see this package's README for what was measured)::

    TeaPackageLook = gated_reply("tea_package_in_view", PackageText, otherwise={"what_is_in_view": str})
    # in PackageText:  directions: gate("brewing_directions_visible", BrewingDirections)

A reply's value is read back by its flag (``reply.look.tea_package_in_view``), not by its class.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, create_model


def gate(flag: str, then: type[BaseModel], *, otherwise: Mapping[str, Any] | None = None) -> Any:
    """The type of a field whose value is ``{flag: true, ...then's fields}`` or ``{flag: false, ...otherwise}``.

    ``otherwise`` maps field names to a type (or a ``(type, Field(...))`` pair) the false shape also has, such as
    what is in view instead. The flag comes first in both shapes, so the model decides it before anything else.
    """
    shown = create_model(  # type: ignore[call-overload]
        f"{then.__name__}Shown",
        **{flag: (Literal[True], ...)},
        **{name: (info.annotation, info) for name, info in then.model_fields.items()},
    )
    absent = create_model(  # type: ignore[call-overload]
        f"{then.__name__}Absent",
        **{flag: (Literal[False], ...)},
        **{name: (value if isinstance(value, tuple) else (value, ...)) for name, value in (otherwise or {}).items()},
    )
    return shown | absent


def gated_reply(
    flag: str, then: type[BaseModel], *, otherwise: Mapping[str, Any] | None = None, field: str = "look"
) -> type[BaseModel]:
    """A reply schema whose one field (``field``) is :func:`gate`\\ ``(flag, then, otherwise=...)``: the gate at the
    top of what the model writes for a frame."""
    return create_model(f"{then.__name__}Reply", **{field: (gate(flag, then, otherwise=otherwise), ...)})  # type: ignore[call-overload]
