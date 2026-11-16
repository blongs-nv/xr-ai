# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``GraphBackend``: a compiled LangGraph graph as an ``xr_ai_kit.Delegate`` back end."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

from xr_ai_kit.delegate import Job, JobUpdate
from xr_ai_next.tools import render

from ._chat import _text

__all__ = ["GraphBackend"]


def _default_input(request: str) -> dict[str, Any]:
    return {"request": request}


def _default_result(state: Any) -> str:
    """The last message's text for a message graph, else ``state["answer"]``, else the state as JSON."""
    if isinstance(state, Mapping):
        history = state.get("messages")
        if isinstance(history, list) and history:
            last = history[-1]
            return _text(getattr(last, "content", last))
        if "answer" in state:
            return render(state["answer"])
    return render(state)


class GraphBackend:
    """Runs ``graph`` (a compiled LangGraph graph) as one job per request.

    - ``input(request)`` builds the graph input (default ``{"request": request}``).
    - Each custom event (``get_stream_writer()(...)`` in a node) that is a dict with ``"progress"`` is a progress
      update with that text; other custom events are ignored. A ``Delegate`` speaks progress, so write it for the
      user.
    - A node that finishes is a progress update only when the app gives ``progress(node, update)``: its text, or
      ``None`` to stay quiet for that node. By default finished nodes say nothing: node names are internal and must
      not reach the user. Internal entries such as ``__interrupt__`` are skipped.
    - When the graph ends, ``result(state)`` on the final state is the result: a string, or a ``JobUpdate`` (to set
      ``speak``). The default is the last message's text for a message graph, else ``state["answer"]``.
    - ``config`` (a mapping, or ``config(job)``) is passed to the graph run, e.g. a ``thread_id`` for a checkpointer.

    The graph runs with ``stream_mode=["updates", "custom", "values"]``; ``values`` gives the final state with the
    graph's own reducers applied. Cancelling the job (``Delegate.cancel``) cancels the graph run: the cancellation
    reaches the running node, and the stream is closed. A graph error fails the job.
    """

    def __init__(
        self,
        graph: Any,
        *,
        input: Callable[[str], Any] = _default_input,
        result: Callable[[Any], str | JobUpdate] = _default_result,
        progress: Callable[[str, Any], str | None] | None = None,
        config: Mapping[str, Any] | Callable[[Job], Mapping[str, Any]] | None = None,
    ) -> None:
        if not hasattr(graph, "astream"):
            raise TypeError("GraphBackend needs a compiled LangGraph graph (an object with astream)")
        self._graph = graph
        self._input = input
        self._result = result
        self._progress = progress
        self._config = config

    async def run(self, request: str, job: Job) -> AsyncIterator[JobUpdate]:
        config = self._config(job) if callable(self._config) else self._config
        stream = self._graph.astream(
            self._input(request), dict(config) if config else None, stream_mode=["updates", "custom", "values"]
        )
        final: Any = None
        seen = False
        async with contextlib.aclosing(stream):
            async for mode, chunk in stream:
                if mode == "values":
                    final, seen = chunk, True
                elif mode == "updates" and self._progress is not None and isinstance(chunk, Mapping):
                    for node, update in chunk.items():
                        if str(node).startswith("__"):
                            continue
                        text = self._progress(str(node), update)
                        if text:
                            yield JobUpdate("progress", text, data={"node": node})
                elif mode == "custom" and isinstance(chunk, Mapping) and chunk.get("progress"):
                    yield JobUpdate("progress", str(chunk["progress"]), data=dict(chunk))
        if not seen:
            yield JobUpdate("error", "the graph ended without a state")
            return
        value = self._result(final)
        yield value if isinstance(value, JobUpdate) else JobUpdate("result", render(value), data=final)
