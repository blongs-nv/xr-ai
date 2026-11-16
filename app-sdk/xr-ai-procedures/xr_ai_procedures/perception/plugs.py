# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The plugs: how a look observes and how it judges.

An **observer** turns a frame into an observation (any type: a description, a model instance, detections from a
plain vision model), or ``None`` when it cannot. A **judge** turns an observation into ``"yes"``, ``"no"``,
``"unknown"`` or :class:`~.outcomes.Missing`. Both are plain sync or async functions, so any visual or decision
intelligence plugs in. :func:`read` and :func:`describe` build observers from a perception role, :func:`by_choices`
a judge from a decision role, :func:`all_of` combines judges, and a :class:`Tier` is an observer to escalate to.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ._util import call
from .outcomes import Missing, Verdict

if TYPE_CHECKING:
    from xr_ai_app import Frame, Role

__all__ = ["Criteria", "Judge", "Observer", "Tier", "all_of", "by_choices", "describe", "read", "text_of"]

Observer = Callable[[Any], Any]
"""``(frame) -> observation | None``, sync or async."""

Judge = Callable[[Any], Any]
"""``(observation) -> "yes" | "no" | "unknown" | Missing``, sync or async."""


def describe(
    perceive: Role, question: str, system: str = "", *, thinking: bool = False, max_tokens: int | None = 128
) -> Callable[[Frame], Awaitable[str]]:
    """Observer: the perception role answers ``question`` about the frame in words."""

    async def observe(frame: Frame) -> str:
        answer = await perceive.ask(frame, question, system=system or None, thinking=thinking, max_tokens=max_tokens)
        return str(answer)

    return observe


def read(
    perceive: Role,
    question: str,
    model: type[BaseModel],
    system: str = "",
    *,
    thinking: bool = False,
    max_tokens: int | None = 128,
) -> Callable[[Frame], Awaitable[BaseModel]]:
    """Observer: the perception role answers ``question`` into ``model`` (the schema is the answer's contract)."""

    async def observe(frame: Frame) -> BaseModel:
        return await perceive.ask(
            frame, question, response_format=model, system=system or None, thinking=thinking, max_tokens=max_tokens
        )

    return observe


@dataclass(frozen=True, slots=True)
class Criteria:
    """What counts as yes, no and unknown for an observation in words (for :func:`by_choices`)."""

    positive: str
    negative: str
    unknown: str


def by_choices(decide: Role, instructions: str, criteria: Criteria) -> Judge:
    """Judge: the decision role picks yes, no or unknown for an observation."""

    async def judge(seen: Any) -> Verdict:
        verdict = await decide.decide(
            instructions,
            choices={"yes": criteria.positive, "no": criteria.negative, "unknown": criteria.unknown},
            context=f"Observation: {text_of(seen)}",
        )
        return str(verdict)  # type: ignore[return-value]

    return judge


def all_of(*judges: Judge) -> Judge:
    """Judge: yes only if every judge says yes; the first that does not decides (no, unknown or Missing)."""

    async def judge(seen: Any) -> Verdict | Missing:
        for one in judges:
            verdict = await call(one, seen)
            if verdict != "yes":
                return verdict
        return "yes"

    return judge


@dataclass(frozen=True, slots=True)
class Tier:
    """An escalation: after ``after_s`` seconds without a usable look, look this way instead (and keep doing so).

    ``judge=None`` keeps the look's judge. A tier is any observer: a careful call with reasoning on, a stronger
    model, a plain-code path.
    """

    after_s: float
    observe: Observer
    judge: Judge | None = None
    name: str = "fallback"


def text_of(seen: Any) -> str:
    """An observation as one line of text (a model as JSON)."""
    return seen.model_dump_json() if isinstance(seen, BaseModel) else str(seen)
