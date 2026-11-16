# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tools as data: what a model is told about a tool, and how to call it.

``@tool`` (``xr_ai_next.tool``) marks a function; :func:`describe` turns it (or a plain async function, or a
:class:`Tool`: a tool defined by its argument model and a handler, as an adapter builds one) into a
:class:`ToolSpec`: its name, its description, the model of its arguments (``p`` and ``turn`` parameters are
injected, not shown to the model) and whether its result is the answer (``return_direct``). ``await spec.call(p,
arguments)`` runs it: the arguments checked against that model, ``p`` and the current turn injected, a ``tool``
trace span, model calls inside it tagged with the tool, the result rendered to the text a model reads
(:func:`render`). ``spec.stream(...)`` is the same call with the result as it arrives (a tool that returns a stream
of text). A tool loop (the kit's ``Conversation``) and framework adapters (offering the app's tools to another
framework's agent) call tools this way; a tool returns a value and the loop speaks the answer.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, AsyncIterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from xr_ai_tools import Tool
from xr_ai_tools.tools import _json_result as _default_tool_renderer

from . import trace
from ._base.participant import current_turn
from ._base.tools import _render, _spec, _ToolSpec

if TYPE_CHECKING:
    from ._base.participant import Participant, Turn

__all__ = ["Tool", "ToolSpec", "describe", "render"]


def render(value: Any) -> str:
    """The text a model reads for a tool's result: a string as it is, anything else as JSON."""
    return _render(value.model_dump() if isinstance(value, BaseModel) else value)


@dataclass(frozen=True)
class ToolSpec:
    """A tool as a model sees it, and the way to call it (:func:`describe`)."""

    name: str
    description: str
    arguments: type[BaseModel]
    """The arguments the model gives (without ``p`` and ``turn``)."""
    _impl: _ToolSpec | Tool[Any, Any]
    return_direct: bool = False
    """The result is the answer to the turn: a tool loop says it instead of asking the model again."""

    async def call(self, p: Participant, arguments: Mapping[str, Any], *, turn: Turn | None = None) -> str:
        """Run the tool for ``p`` with the model's ``arguments``; ``turn`` defaults to the turn being answered."""
        return "".join([part async for part in self.stream(p, arguments, turn=turn)])

    async def stream(
        self, p: Participant, arguments: Mapping[str, Any], *, turn: Turn | None = None
    ) -> AsyncGenerator[str, None]:
        """:meth:`call`, the result as it arrives: a tool that returns a stream of text yields each part; any other
        result is one part. Invalid arguments and a failure before the first part raise before anything is yielded.
        """
        async with trace.span(self.name, kind="tool") as record:
            record["arguments"] = {k: v.model_dump() if isinstance(v, BaseModel) else v for k, v in arguments.items()}
            with trace.in_tool(self.name):
                value = await self._run(p, arguments, turn if turn is not None else current_turn())
            if not isinstance(value, AsyncIterable):
                record["result"] = trace.brief(text := self._text(value))
                yield text
                return
            parts: list[str] = []
            iterator = aiter(value)
            try:
                while True:
                    with trace.in_tool(self.name):  # the stream's model call runs as it is read
                        try:
                            part = await anext(iterator)
                        except StopAsyncIteration:
                            break
                    parts.append(str(part))
                    yield str(part)
            finally:
                record["result"] = trace.brief("".join(parts))
                if (close := getattr(iterator, "aclose", None)) is not None:
                    await close()

    def _text(self, value: Any) -> str:
        """A result as the model reads it: a foundation ``Tool`` given its own renderer renders a non-text result."""
        if value is None:
            return ""
        renderer = getattr(self._impl, "_render_result", None)  # no public name
        if isinstance(self._impl, Tool) and not isinstance(value, str) and renderer is not _default_tool_renderer:
            return renderer(value)
        return render(value)

    async def _run(self, p: Participant, arguments: Mapping[str, Any], turn: Turn | None) -> Any:
        impl = self._impl
        if isinstance(impl, Tool):
            value = impl.handler(impl.request_model.model_validate(dict(arguments)))
            return await value if inspect.isawaitable(value) else value
        # Checked against the arguments model (pydantic's coercion, no unknown names); only the arguments the model
        # gave are passed, so the function's own defaults fill the rest.
        request = impl.request_model.model_validate(dict(arguments))
        kwargs = {name: getattr(request, name) for name in impl.parameters if name in arguments}
        if impl.wants_p:
            kwargs["p"] = p
        if impl.wants_turn:
            kwargs["turn"] = turn
        return await impl.function(**kwargs)


def describe(tool: Any) -> ToolSpec:
    """The :class:`ToolSpec` of a ``@tool`` function, a plain async function or an ``xr_ai_tools.Tool``."""
    impl = tool if isinstance(tool, Tool) else _spec(tool)
    return ToolSpec(impl.name, impl.description, impl.request_model, impl, impl.return_direct)
