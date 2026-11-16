# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``VisionRetrieval``: read something from the camera, look it up in documents, and fill the rest from a model."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, create_model
from xr_ai_next import StructuredOutputError, clock

from ._util import call
from .eyes import FrameSource, Watching
from .outcomes import Done, Report, Stuck
from .plugs import Observer

if TYPE_CHECKING:
    from xr_ai_next import Role

__all__ = ["Retrieved", "Source", "VisionRetrieval"]


@dataclass(frozen=True, slots=True)
class Source:
    """Where one value came from. Only a document has a quote."""

    kind: Literal["user", "document", "generated"]
    quote: str = ""


@dataclass(frozen=True, slots=True)
class Retrieved:
    """A :class:`VisionRetrieval` result: what the camera showed, the values, and where each came from."""

    seen: BaseModel
    values: dict[str, Any]
    sources: dict[str, Source]


def extraction_model(result: type[BaseModel]) -> type[BaseModel]:
    """The extraction schema: the result's own fields, plus the passage they came from and the words quoted."""
    return create_model(
        "Extraction",
        __base__=result,
        passage=(int | None, Field(None, description="Index of the passage the values came from.")),
        quote=(str, Field("", description="The exact words copied from that passage.")),
    )


class VisionRetrieval:
    """Read something from the camera, look it up in documents, and fill the rest from a reasoning model.

    1. Look at fresh frames with ``observe`` until ``usable(seen)`` (skipped when ``known`` is given, e.g. the user
       said it); with nothing usable for ``watching.stuck_after_s``, end in ``Stuck``.
    2. Retrieve passages with ``retrieve(query(seen))``.
    3. Extract the missing fields from the passages with ``extract``; the model must quote the passage it used, and
       the quote is checked here, in code.
    4. Generate only what is still missing with ``generate``.

    Every field of ``result`` must default to ``None``. Fields in ``given`` (told by the user) are never looked up.
    ``Done`` carries a :class:`Retrieved` saying where each value came from.
    """

    def __init__(
        self,
        *,
        eyes: FrameSource,
        observe: Observer,
        extract: Role,
        generate: Role,
        retrieve: Callable[[str], Awaitable[list[str]]],
        result: type[BaseModel],
        usable: Callable[[Any], bool],
        query: Callable[[Any], str],
        extract_prompt: str,
        generate_prompt: str,
        watching: Watching,
        timeout_s: float | None = None,
        known: BaseModel | None = None,
        given: dict[str, Any] | None = None,
        now: Callable[[], float] = clock.now,
    ) -> None:
        self.eyes, self.observe = eyes, observe
        self.extract, self.generate, self.retrieve = extract, generate, retrieve
        self.result = result
        self.extraction = extraction_model(result)
        self.usable, self.query = usable, query
        self.extract_prompt, self.generate_prompt = extract_prompt, generate_prompt
        self.watching, self.timeout_s = watching, timeout_s
        self.known, self.given = known, dict(given or {})
        self._clock = now

    async def run(self, report: Report) -> Done | Stuck:
        seen = self.known
        started, why = self._clock(), "unseen"
        while seen is None:
            frame = await self.eyes.next()
            if frame is not None:
                seen = await self._read(frame)
                why = "unreadable"
                await report("reading what the camera shows")
            if seen is None and self._clock() - started > self.watching.stuck_after_s:
                return Stuck(why)  # type: ignore[arg-type]

        values = dict(self.given)
        sources = {name: Source("user") for name in values}
        missing = [n for n in self.result.model_fields if n not in values]
        if missing:
            await report("looking it up")
            passages = await self.retrieve(self.query(seen))
            found, source = await self._extract(seen, missing, passages)
            for name, value in found.items():
                values[name], sources[name] = value, source
            missing = [n for n in missing if n not in values]
        if missing:
            await report("filling in what the documents do not say")
            guess = await self.generate.ask(
                None,
                json.dumps({"seen": seen.model_dump(), "missing": missing}, default=str),
                system=self.generate_prompt,
                response_format=self.result,
                thinking=True,  # the profile's token limit: an explicit cap would also cap the reasoning
            )
            for name in missing:
                value = getattr(guess, name)
                if value is not None:
                    values[name], sources[name] = value, Source("generated")
        return Done(Retrieved(seen, values, sources))

    async def _read(self, frame: Any) -> Any | None:
        try:
            async with asyncio.timeout(self.timeout_s):
                got = await call(self.observe, frame)
        except (TimeoutError, StructuredOutputError):
            return None
        return got if got is not None and self.usable(got) else None

    async def _extract(self, seen: BaseModel, missing: list[str], passages: list[str]) -> tuple[dict[str, Any], Source]:
        """Values the passages state, with a quote that really is in them."""
        try:
            async with asyncio.timeout(self.timeout_s):
                got = await self.extract.ask(
                    None,
                    json.dumps({"seen": seen.model_dump(), "fields": missing, "passages": passages}, default=str),
                    system=self.extract_prompt,
                    response_format=self.extraction,
                    thinking=False,
                    max_tokens=512,
                )
        except (TimeoutError, StructuredOutputError):
            return {}, Source("document")
        quoted = (
            got.passage is not None
            and 0 <= got.passage < len(passages)
            and bool(got.quote)
            and got.quote in passages[got.passage]
        )
        if not quoted:
            return {}, Source("document")
        values = {name: getattr(got, name) for name in missing if getattr(got, name) is not None}
        return values, Source("document", got.quote)
