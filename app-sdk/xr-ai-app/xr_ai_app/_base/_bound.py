# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Roles bound to real ``xr_ai_models`` clients, and the pool that builds and closes them.

Cancelling a call cancels the underlying request; cancelling or closing a stream closes the model
stream exactly once (LIFECYCLE_CONTRACT R5.1 and R5.3).
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from xr_ai_models import (
    ChatMessage,
    DecisionQuestion,
    make_decision,
    make_embedding,
    make_llm,
    make_vlm,
)

from . import _tracing, _virtual
from .profiles import (
    SPEECH_CATEGORIES,
    Decision,
    Profile,
    Role,
    StructuredOutputError,
    _category,
    speech_role_error,
)

if TYPE_CHECKING:
    from .frames import Frame

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
_MAKERS: dict[str, Callable[[Any, str], Any]] = {
    "llm": make_llm,
    "vlm": make_vlm,
    "decision": make_decision,
    "embedding": make_embedding,
}


class ClientPool:
    """Build each model client once per entry; the app closes them in reverse order of creation."""

    def __init__(self, profile: Profile, register: Callable[[Any], None]) -> None:
        self._profile = profile
        self._register = register
        self._clients: dict[str, Any] = {}
        self._limits: dict[str, asyncio.Semaphore | None] = {}

    def client(self, entry: str) -> Any:
        """Return the client for ``entry``, building and registering it on first use.

        A speech entry (``stt``, ``tts``) has no chat client: it returns ``None``, and a role bound to it
        raises on ``ask``, ``stream`` and ``decide``.
        """
        client = self._clients.get(entry)
        if client is None:
            category = _category(self._profile.models.entries[entry])
            if category in SPEECH_CATEGORIES:
                return None
            maker = _MAKERS.get(category)
            if maker is None:
                raise TypeError(f"entry {entry!r} is a {category!r} model; no text role can use it")
            client = self._clients[entry] = maker(self._profile.models, entry)
            self._register(client)
        return client

    def limiter(self, entry: str, role: str) -> asyncio.Semaphore | None:
        """One limiter per entry, sized by the role's ``max_concurrency`` budget (``None``: unlimited)."""
        if entry not in self._limits:
            conc = self._profile.budget(role).max_concurrency
            self._limits[entry] = None if conc is None else asyncio.Semaphore(conc)
        return self._limits[entry]


class _BoundStream:
    """Async iterator over a model stream that closes the underlying stream once, however it ends."""

    def __init__(
        self,
        name: str,
        opener: Callable[[], AsyncIterator[str]],
        limiter: asyncio.Semaphore | None,
        *,
        timeout_s: float | None = None,
    ) -> None:
        self._name = name
        self._timeout_s = timeout_s
        self._deadline: float | None = None
        self._opener = opener
        self._limiter = limiter
        self._it: AsyncIterator[str] | None = None
        self._held = False
        self._closed = False

    def __aiter__(self) -> _BoundStream:
        return self

    async def __anext__(self) -> str:
        if self._closed:
            raise StopAsyncIteration
        try:
            if self._timeout_s is not None and self._deadline is None:
                self._deadline = asyncio.get_running_loop().time() + self._timeout_s
            return await self._within_deadline(self._step())
        except BaseException:  # noqa: BLE001 - finish, then re-raise (end of stream, error or cancellation)
            await self.aclose()
            raise

    async def _step(self) -> str:
        if self._it is None:
            if self._limiter is not None:
                await self._limiter.acquire()
                self._held = True
            self._it = self._opener().__aiter__()
        return await self._it.__anext__()

    async def _within_deadline(self, step: Any) -> str:
        if self._deadline is None:
            return await step
        remaining = self._deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            step.close()
            raise TimeoutError(f"role {self._name!r} stream exceeded {self._timeout_s}s")
        return await asyncio.wait_for(step, remaining)

    async def aclose(self) -> None:
        """Close the stream; later calls do nothing."""
        if self._closed:
            return
        self._closed = True
        try:
            closer = getattr(self._it, "aclose", None)
            if closer is not None:
                await closer()
        finally:
            if self._held:
                self._held = False
                self._limiter.release()  # type: ignore[union-attr]


def _schema_prompt(prompt: str, response_format: type[BaseModel]) -> str:
    schema = json.dumps(response_format.model_json_schema())
    return f"{prompt}\n\nAnswer with one JSON object only, matching this JSON schema:\n{schema}"


def _parse(text: str, response_format: type[BaseModel]) -> Any:
    body = text.strip()
    if match := _FENCE.match(body):
        body = match.group(1)
    elif (start := body.find("{")) > 0 and (end := body.rfind("}")) > start:
        body = body[start : end + 1]
    try:
        return response_format.model_validate_json(body)
    except ValueError as exc:  # pydantic's ValidationError is a ValueError
        error = exc
    # Some models (Cosmos3-Nano) fill in the schema itself: the answer is the object under "properties", next to
    # copies of the schema's other keys ("title", "required", ...).
    try:
        data = json.loads(body)
    except ValueError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("properties"), dict):
        try:
            return response_format.model_validate(data["properties"])
        except ValueError:
            pass
    raise StructuredOutputError(f"reply is not a valid {response_format.__name__}: {error}") from error


def _frames(frame: Frame | Sequence[Frame] | None) -> list[Frame]:
    if frame is None:
        return []
    if isinstance(frame, Sequence):
        return list(frame)
    return [frame]


def _kwargs(**options: Any) -> dict[str, Any]:
    """The options a caller set; unset ones are left out so the client's own defaults apply."""
    return {key: value for key, value in options.items() if value is not None}


class _SpeechClient:
    """Stands in for the client of a role bound to a speech entry: any chat call on it raises (it has none)."""

    def __init__(self, name: str, category: str) -> None:
        self._name = name
        self._category = category

    def __getattr__(self, method: str) -> Any:
        if method.startswith("_") or method == "capabilities":
            raise AttributeError(method)
        raise speech_role_error(self._name, self._category, method)


class BoundRole(Role):
    """A role whose calls go to the real client of the profile entry that fills it."""

    def __init__(
        self, name: str, *, entry: str, category: str, client: Any, limiter: asyncio.Semaphore | None = None
    ) -> None:
        self._name = name
        self._entry = entry
        self._category = category
        self._client = _SpeechClient(name, category) if category in SPEECH_CATEGORIES else client
        self._limiter = limiter

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"<Role {self._name!r} -> {self._entry!r} ({self._category})>"

    def _call(
        self,
        frame: Frame | Sequence[Frame] | None,
        prompt: str,
        *,
        system: str | None,
        thinking: bool | None,
        max_tokens: int | None,
        temperature: float | None,
        stream: bool,
        schema: Mapping[str, Any] | None = None,
    ) -> Callable[[], Any]:
        """Build the client call for this role's category; validates the frames first.

        ``schema`` (a JSON Schema) is sent for the server to constrain the reply to; only pass it when
        :meth:`_constrains` says the endpoint declares that.
        """
        frames = _frames(frame)
        if self._category == "vlm":
            if not frames:
                raise ValueError(f"role {self._name!r} is a vision model and needs a frame")
            options = _kwargs(
                system_prompt=system, max_tokens=max_tokens, temperature=temperature, response_schema=schema
            )
            images = [f.image for f in frames]
            if len(images) == 1:
                method = self._client.stream if stream else self._client.ask_image
                return lambda: method(images[0], prompt, **options)
            method = self._client.stream_images if stream else self._client.ask_images
            return lambda: method(images, prompt, **options)
        if frames:
            raise ValueError(f"role {self._name!r} is a text model and cannot take a frame; pass None")
        messages = [ChatMessage("user", prompt)]
        if system:
            messages.insert(0, ChatMessage("system", system))
        options = _kwargs(
            enable_thinking=thinking, max_tokens=max_tokens, temperature=temperature, response_schema=schema
        )
        method = self._client.stream if stream else self._client.chat
        return lambda: method(messages, **options)

    def _constrains(self) -> bool:
        """Whether the endpoint declares JSON Schema output (``capabilities: {json_schema: true}`` in models.json)."""
        capabilities = getattr(self._client, "capabilities", None)
        return bool(getattr(capabilities, "json_schema", False))

    @_tracing.traced_call("ask")
    @_virtual.held_call
    async def ask(
        self,
        frame: Frame | Sequence[Frame] | None,
        prompt: str,
        *,
        response_format: type[BaseModel] | None = None,
        system: str | None = None,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        self._require_text_model("ask")
        text = _schema_prompt(prompt, response_format) if response_format is not None else prompt
        schema = response_format.model_json_schema() if response_format is not None and self._constrains() else None
        call = self._call(
            frame,
            text,
            system=system,
            thinking=thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            schema=schema,
        )

        async def run() -> str:
            if self._limiter is None:
                return (await call()).content
            async with self._limiter:
                return (await call()).content

        reply = await run() if timeout_s is None else await asyncio.wait_for(run(), timeout_s)
        if response_format is None:
            return reply
        try:
            return _parse(reply, response_format)
        except StructuredOutputError:
            # One retry: models that ignore a prompt-only schema often comply when told what went wrong (an
            # endpoint that declares json_schema constrains the reply itself; its retry covers truncation).
            text = f"{text}\n\nYour previous reply was not valid JSON for this schema. Reply with the JSON object only."
            call = self._call(
                frame,
                text,
                system=system,
                thinking=thinking,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                schema=schema,
            )
            reply = await run() if timeout_s is None else await asyncio.wait_for(run(), timeout_s)
            return _parse(reply, response_format)

    @_tracing.traced_stream
    def stream(
        self,
        frame: Frame | Sequence[Frame] | None,
        prompt: str,
        *,
        system: str | None = None,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[str]:
        self._require_text_model("stream")
        opener = self._call(
            frame, prompt, system=system, thinking=thinking, max_tokens=max_tokens, temperature=temperature, stream=True
        )
        return _virtual.HeldStream(_BoundStream(self._name, opener, self._limiter, timeout_s=timeout_s))

    @_tracing.traced_call("decide")
    @_virtual.held_call
    async def decide(
        self,
        question: str,
        *,
        choices: Sequence[str] | Mapping[str, str],
        context: str = "",
        frames: Sequence[Frame] = (),
    ) -> Decision:
        if self._category in SPEECH_CATEGORIES:
            raise speech_role_error(self._name, self._category, "decide")
        if self._category != "decision":
            raise TypeError(
                f"role {self._name!r} is backed by a {self._category!r} model; decide needs a decision model"
            )
        criteria = dict(choices) if isinstance(choices, Mapping) else {c: f"Select {c!r}." for c in choices}
        labels = list(criteria)
        if isinstance(choices, Mapping):
            if len(labels) < 2 or not all(c.strip() and str(t).strip() for c, t in criteria.items()):
                raise ValueError("decide needs at least two non-empty choices with non-empty criteria")
        elif len(labels) < len(choices) or len(labels) < 2 or not all(c.strip() for c in labels):
            raise ValueError("decide needs at least two distinct, non-empty choices")
        questions = {"q": DecisionQuestion(question, criteria)}
        state = context or question
        options = {"images": [f.image for f in frames]} if frames else {}
        if self._limiter is None:
            response = await self._client.decide(state, questions, **options)
        else:
            async with self._limiter:
                response = await self._client.decide(state, questions, **options)
        answer = response.answers["q"]
        if answer.choice not in labels:
            raise ValueError(f"decision model answered {answer.choice!r}, not one of {labels}")
        return Decision(
            answer.choice,
            confidence=getattr(answer, "confidence", None),
            probabilities=getattr(answer, "probabilities", None),
        )

    def _require_text_model(self, method: str) -> None:
        if self._category in SPEECH_CATEGORIES:
            raise speech_role_error(self._name, self._category, method)
        if self._category not in ("llm", "vlm"):
            raise TypeError(f"role {self._name!r} is backed by a {self._category!r} model; {method} needs llm or vlm")
