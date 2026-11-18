# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a step is and what it may touch: the contract every step type, the runner and the evaluation share.

A step (:class:`Step`) is defined once and run with inputs, like a function: ``await run.step(heat_water,
HeatInputs(temperature_c=80))``. It has an ``id``; ``inputs_schema``, the pydantic model of what each run of it
starts with (``None``: it takes none); the line spoken as it starts (``start_line(inputs)``) and when it finishes
(``done_line(inputs, result)``); and ``run(environment, inputs)``. It touches the world only through its
:class:`StepEnvironment`: the camera, speaking to the person, asking them something, and reporting its progress.
Live, the environment is the procedure's :class:`~..runner.Run`; in an evaluation it is a scripted one
(:mod:`..evaluation`) that plays recorded frames, so the evaluation runs exactly the code that ships.

A step type (the kit's ``watch``, ``read``, ``transform``, ``wait``, or an app's own) is any class with that shape.
The kit's types are built from the step's definition (:class:`~.step_definitions.StepDefinition`, its words from the
procedure's YAML), the models it uses and its ``inputs_schema``. The ``{placeholders}`` in its words name fields or
properties of its inputs (``{temperature_c}``, ``{minimum}``) or, in its done line, what its result offers; they are
checked when the step is built (:func:`check_placeholders`) and filled when it runs (:func:`fill_placeholders`).

Optionally a step type has ``skippable`` (default ``True``; ``False`` for a step that does not wait on the person,
so "skip" has nothing to skip) and ``scoring``, naming the module that scores its evaluation cases (``Expect`` and
``score(trace, expect)``), imported only by the evaluation.

Escalation (an agent beside the step that helps the person when the step cannot decide) is not part of this
contract: a step type opts into it separately (``..escalation``).
"""

from __future__ import annotations

import re
import string
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel
from xr_ai_app import Frame

T_co = TypeVar("T_co", covariant=True)


class Frames(Protocol):
    """The camera frames a step reads (``Participant.frames`` live, a recorded scene in an evaluation)."""

    def every(self, interval_s: float, *, timeout_s: float = 5.0) -> AsyncIterator[Frame]: ...

    async def current(self, *, max_age_s: float = 2.0, timeout_s: float = 5.0) -> Frame | None: ...


class StepEnvironment(Protocol):
    """Everything a step may touch. :class:`~..runner.Run` is one; :mod:`..evaluation` has a scripted one."""

    @property
    def frames(self) -> Frames:
        """The person's camera."""
        ...

    async def say(self, text: str) -> None:
        """Tell the person something."""
        ...

    async def ask(self, question: str) -> str | None:
        """Ask the person something and wait for the answer; ``None`` when none came."""
        ...

    async def report_progress(self, step_id: str, state: Any) -> None:
        """Record the step's latest state (its ``summary()``, if it has one), for the conversation and the record."""
        ...


@runtime_checkable
class Step(Protocol[T_co]):
    """One step of a procedure: ``await run.step(step, inputs)`` live, a case file in an evaluation."""

    id: str
    inputs_schema: type[BaseModel] | None
    """What each run of the step starts with (``None``: nothing)."""

    def start_line(self, inputs: Any = None) -> str | None:
        """Spoken as the step starts (``None``: nothing)."""
        ...

    def done_line(self, inputs: Any, result: Any) -> str | None:
        """Spoken when the step finishes with ``result`` (``None``: nothing); a skipped step says nothing."""
        ...

    async def run(self, environment: StepEnvironment, inputs: Any = None) -> T_co: ...


def check_inputs(step_id: str, inputs_schema: type[BaseModel] | None, inputs: Any) -> None:
    """Raise when ``inputs`` is not what a step with ``inputs_schema`` runs with (an instance of it, or ``None``)."""
    if inputs_schema is None:
        if inputs is not None:
            raise TypeError(f"step {step_id!r} takes no inputs; it got {type(inputs).__name__}")
    elif not isinstance(inputs, inputs_schema):
        raise TypeError(f"step {step_id!r} runs with {inputs_schema.__name__}; it got {type(inputs).__name__}")


def placeholder_names(template: str | None) -> Iterator[str]:
    """The names a template's ``{placeholders}`` start with (``observation`` for ``{observation.reading:g}``)."""
    for _, field, _, _ in string.Formatter().parse(template or ""):
        if field:
            yield re.split(r"[.\[]", field, maxsplit=1)[0]


def check_placeholders(
    step_id: str, where: str, template: str | None, inputs_schema: type[BaseModel] | None, *, also: tuple[str, ...] = ()
) -> None:
    """Raise when ``template`` names a placeholder that is neither a field or property of ``inputs_schema`` nor one
    of ``also`` (what a done line gets from the result): a misspelled name fails when the step is built."""
    for name in placeholder_names(template):
        known = name in also or (inputs_schema is not None and hasattr(inputs_schema, name))
        known = known or (inputs_schema is not None and name in inputs_schema.model_fields)
        if not known:
            offered = sorted({*also, *(inputs_schema.model_fields if inputs_schema else ())})
            raise ValueError(
                f"step {step_id!r}: its {where} uses {{{name}}}, which it is not given (it has: {offered or 'none'})"
            )


class _Placeholders(Mapping[str, Any]):
    """The values a template's placeholders name: ``extra`` first, then the inputs' fields and properties."""

    def __init__(self, inputs: Any, extra: Mapping[str, Any]) -> None:
        self._inputs, self._extra = inputs, extra

    def __getitem__(self, name: str) -> Any:
        if name in self._extra:
            return self._extra[name]
        if self._inputs is not None and hasattr(self._inputs, name):
            return getattr(self._inputs, name)
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._extra)

    def __len__(self) -> int:
        return len(self._extra)


def fill_placeholders(template: str | None, inputs: Any = None, **extra: Any) -> str | None:
    """``template`` with its ``{placeholders}`` filled from ``extra`` and the fields and properties of ``inputs``
    (``None`` stays ``None``)."""
    return None if template is None else template.format_map(_Placeholders(inputs, extra))
