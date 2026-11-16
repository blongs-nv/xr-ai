# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Verify``: watch the camera until a condition holds; ``read_check``: the common camera check.

The loop is the common part: fresh frames, pacing, per-look timeouts, the confirmation window, escalation, and
giving up (``Stuck``). What looks and what judges are plugged in::

    Verify(eyes=..., observe=read(perceive, "Read the display", Reading),
           judge=lambda r: "yes" if r.value >= 90 else "no", watching=...)
    Verify(eyes=..., observe=describe(perceive, "Is there water?"), judge=by_choices(decide, ..., criteria), ...)
    Verify(eyes=..., observe=my_detector, judge=lambda hit: "yes" if hit else "no", ...)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from xr_ai_app import StructuredOutputError, clock, trace

from ._util import call
from .confirmation import Confirmation
from .eyes import Eyes, FrameSource, Watching
from .outcomes import Done, Missing, Needs, Report, Stuck, Verdict
from .plugs import Judge, Observer, Tier, read, text_of

if TYPE_CHECKING:
    from pydantic import BaseModel
    from xr_ai_app import Frame, FrameStream, Role

__all__ = ["Verify", "read_check"]


class Verify:
    """Watch until ``required_positives`` of the last ``window`` usable looks say yes.

    An ``unknown`` look (no frame, nothing observed, a slow, failed or malformed model reply) takes no place in the
    window. While looks keep being unusable it moves up its ``escalation`` tiers (by seconds without a usable
    look) and stays on the tier that worked; after ``watching.stuck_after_s`` it gives up with ``Stuck``. A judge
    answering ``Missing(fact)`` ``watching.missing_after`` looks in a row ends it with ``Needs(fact)``. ``Done``
    carries the last observation. ``timeout_s`` bounds one look, end to end (``None``: the role's own timeout).
    """

    def __init__(
        self,
        *,
        eyes: FrameSource,
        observe: Observer,
        judge: Judge,
        watching: Watching,
        timeout_s: float | None = None,
        escalation: tuple[Tier, ...] = (),
        text: Callable[[Any], str] = text_of,
        now: Callable[[], float] = clock.now,
    ) -> None:
        self.eyes = eyes
        self.judge = judge
        self.tiers = (Tier(0.0, observe, judge, "primary"), *sorted(escalation, key=lambda t: t.after_s))
        self.watching = watching
        self.timeout_s = timeout_s
        self.text = text
        self._clock = now

    async def run(self, report: Report) -> Done | Needs | Stuck:
        w = self.watching
        confirmation = Confirmation(w.required_positives, w.window)
        useful_at = self._clock()
        lacking, last_lack = 0, None
        why: Any = "unseen"
        level = 0
        while True:
            unable_for = self._clock() - useful_at
            while level + 1 < len(self.tiers) and unable_for >= self.tiers[level + 1].after_s:
                level += 1
            tier = self.tiers[level]
            frame = await self.eyes.next()
            if frame is None:
                why = "unseen"
            else:
                verdict, seen, observation = await self._look(frame, tier)
                if isinstance(verdict, Missing):
                    lacking = lacking + 1 if verdict.fact == last_lack else 1
                    last_lack = verdict.fact
                    if lacking >= w.missing_after:
                        return Needs(verdict.fact)
                else:
                    lacking, last_lack = 0, None
                    if verdict in ("yes", "no"):
                        confirmation.add(verdict)
                        useful_at = self._clock()
                    else:
                        why = "unclear" if seen else "unreadable"
                if seen:
                    await report(seen + (f" ({tier.name})" if level else ""))
                if confirmation.confirmed:
                    return Done(observation)
            if self._clock() - useful_at > w.stuck_after_s:
                return Stuck(why)

    async def _look(self, frame: Frame, tier: Tier) -> tuple[Verdict | Missing, str, Any]:
        """One look, end to end. A slow, failed or malformed reply is ``unknown``."""
        try:
            async with asyncio.timeout(self.timeout_s):
                seen = await call(tier.observe, frame)
                if seen is None:
                    return "unknown", "", None
                verdict = await call(tier.judge or self.judge, seen)
                return verdict, self.text(seen), seen
        except (TimeoutError, StructuredOutputError) as exc:
            trace.emit("perception", "look_failed", tier=tier.name, error=type(exc).__name__)
        except Exception as exc:  # noqa: BLE001 - a failed look is unknown; watching goes on
            trace.emit("error", tier.name, where="perception look", error=type(exc).__name__)
        return "unknown", "", None


def read_check(
    *,
    frames: FrameStream,
    perceive: Role,
    question: str,
    reads: type[BaseModel],
    judge: Judge,
    watching: Watching,
    text: Callable[[Any], str] = text_of,
    timeout_s: float | None = None,
    system: str = "",
    careful_after_s: float | None = None,
) -> Verify:
    """The common camera check: a structured read, then a judge in plain code.

    The perception role answers ``question`` into ``reads`` on frames from ``frames`` (``p.frames``), and
    ``judge(reading)`` turns the reading into yes, no, unknown or ``Missing(fact)``. ``text(reading)`` is what the
    check reports it sees. The answer is not constrained by the model server: ``Role.ask(response_format=reads)``
    puts the schema in the prompt and parses the reply, asking once more when it is not valid JSON for the schema;
    a second malformed reply is an ``unknown`` look. One model call per look (two after a malformed reply). With
    ``careful_after_s``, looks that keep failing escalate to a careful one (same schema, reasoning on, and the
    profile's token limit, since a per-call cap would also cap the reasoning).
    """
    escalation: tuple[Tier, ...] = ()
    if careful_after_s is not None:
        careful = read(perceive, question, reads, system, thinking=True, max_tokens=None)  # room to reason
        escalation = (Tier(careful_after_s, careful, name="careful look"),)
    return Verify(
        eyes=Eyes.watching(frames, watching),
        observe=read(perceive, question, reads, system),
        judge=judge,
        text=text,
        escalation=escalation,
        watching=watching,
        timeout_s=timeout_s,
    )
