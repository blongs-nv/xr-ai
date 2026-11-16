# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``@tool`` decorator with participant injection."""

from __future__ import annotations

import inspect
import typing
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar, overload

import pydantic_core
from pydantic import BaseModel, ConfigDict, create_model

from .participant import Participant, Turn

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])

_MARK = "__xr_ai_tool__"


@overload
def tool(fn: F, /) -> F: ...


@overload
def tool(
    *, name: str | None = None, description: str | None = None, return_direct: bool = False
) -> Callable[[F], F]: ...


def tool(
    fn: F | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    return_direct: bool = False,
) -> Any:
    """Declare an async function a model-callable tool.

    A parameter named ``p`` annotated :class:`Participant` is injected and hidden from the
    model's schema; so is a parameter named ``turn`` annotated :class:`Turn`, which gives the
    tool the current turn (its text and history). The call runs in the turn's scope, so cancelling
    the turn cancels the tool. Parameters may be annotated ``Annotated[T, Field(description=...)]``
    to describe them to the model.

    What the tool returns:

    - text, as it is; a pydantic model or any other value, as JSON; ``None``, as empty text;
    - or an async iterable of text (a stream, e.g. ``role.stream(...)``): the model reads it
      joined; a direct tool's stream is spoken as it arrives.

    ``return_direct=True``: the result is the answer to the turn. The tool loop speaks it and does
    not ask the model again. An empty result (``""`` or ``None``) ends the turn with nothing more
    said: return it when the tool has already spoken for itself, for example to everyone in the
    room. A tool that raises is not an answer: the model reads the error, as for any tool.
    """

    def mark(function: F) -> F:
        _spec(function, name=name, description=description)  # fails here, not on the first model call
        setattr(function, _MARK, (name, description, return_direct))
        return function

    return mark(fn) if fn is not None else mark


@dataclass(frozen=True, slots=True)
class _ToolSpec:
    name: str
    description: str
    function: Callable[..., Awaitable[Any]]
    request_model: type[BaseModel]
    parameters: tuple[str, ...]
    wants_p: bool
    wants_turn: bool
    return_direct: bool


def _injected(parameter: inspect.Parameter, hint: Any, wanted: type) -> bool:
    """A parameter named ``p`` or ``turn`` is injected unless it is annotated with something else."""
    if hint is inspect.Parameter.empty or hint is wanted:
        return True
    return isinstance(hint, str) and hint.rsplit(".", 1)[-1] == wanted.__name__


def _spec(
    function: Callable[..., Awaitable[Any]], *, name: str | None = None, description: str | None = None
) -> _ToolSpec:
    if not inspect.iscoroutinefunction(function):
        raise TypeError(f"tool {getattr(function, '__name__', function)!r} must be an async function")
    marked = getattr(function, _MARK, (None, None, False))
    tool_name = name or marked[0] or function.__name__
    text = description or marked[1] or inspect.getdoc(function)
    if not text:
        raise ValueError(f"tool {tool_name!r} needs a description: add a docstring or pass description=")
    try:
        hints = typing.get_type_hints(function, include_extras=True)
    except Exception:  # noqa: BLE001 - unresolved forward references fall back to the raw annotations
        hints = {}
    fields: dict[str, Any] = {}
    wants_p = wants_turn = False
    for parameter_name, parameter in inspect.signature(function).parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ValueError(f"tool {tool_name!r}: *args and **kwargs cannot be described to a model")
        hint = hints.get(parameter_name, parameter.annotation)
        if parameter_name == "p" and _injected(parameter, hint, Participant):
            wants_p = True
        elif parameter_name == "turn" and _injected(parameter, hint, Turn):
            wants_turn = True
        else:
            default: Any = ... if parameter.default is inspect.Parameter.empty else parameter.default
            fields[parameter_name] = (Any if hint is inspect.Parameter.empty else hint, default)
    model = create_model(
        f"{tool_name}_arguments",
        __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=True),
        **fields,
    )
    return _ToolSpec(tool_name, text, function, model, tuple(fields), wants_p, wants_turn, marked[2])


def _render(value: Any) -> str:
    """The text a model reads for a tool's return value: strings as they are, everything else as JSON."""
    if isinstance(value, str):
        return value
    return pydantic_core.to_json(value, fallback=str).decode()


__all__ = ["tool"]
