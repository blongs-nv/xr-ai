# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace events into NeMo Relay, so Relay's exporters (ATOF, ATIF, OpenTelemetry) see an XR-AI app.

:class:`RelaySink` maps the trace onto Relay's model:

==========================  ===============================================================================
trace event                 Relay
==========================  ===============================================================================
``turn`` start / end        an ``Agent`` scope named ``turn``, input the user's text, output what was said
``role`` (one model call)   an ``Llm`` scope (start and end at the call's times), input and output summarized
``tool`` (a tool span)      a ``Tool`` scope with the arguments and the result
everything else             a mark ``xr_ai_app.<kind>`` with the event's ids and data
==========================  ===============================================================================

A model call or tool inside a turn is that turn's child. Each turn has its own forked Relay context, so concurrent
turns never share a scope stack, and the caller's own Relay stack is never touched.
:func:`relay_exporters` registers Relay's ATOF and ATIF exporters for a run and writes their files; the devkit
uses it (``--atof``/``--atif``), and an app can too.
"""

from __future__ import annotations

import contextvars
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._tracing import Event

__all__ = ["RelaySink", "relay_exporters"]


def _json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _at(t: float) -> datetime:
    return datetime.fromtimestamp(t, tz=UTC)


class RelaySink:
    """A trace sink that writes the trace into NeMo Relay as scopes and marks (see the module docstring).

    It never touches the caller's Relay scope stack. Relay keeps that stack in a context variable whose stack
    object every child task shares, and it only pops a scope from the top; two participants' turns pushed onto
    one shared stack would end out of order and fail. So each turn gets its own forked Relay context (seeded with
    the caller's current scope as parent), the turn's model calls, tools and marks are written in that context,
    and everything outside a turn goes into a fresh forked context.
    """

    def __init__(self, prefix: str = "xr_ai_app") -> None:
        import nemo_relay  # noqa: PLC0415 - optional at import time

        self._relay = nemo_relay
        self._prefix = prefix
        self._turns: dict[str, tuple[Any, contextvars.Context]] = {}
        self._said: dict[str, list[str]] = {}

    def __call__(self, event: Event) -> None:
        relay = self._relay
        if event.kind == "turn" and event.data.get("phase") == "start":
            turn_id = str(event.data.get("id"))
            ctx = self._fork()
            handle = ctx.run(
                relay.scope.push,
                "turn",
                relay.ScopeType.Agent,
                input={"text": event.data.get("text", ""), "participant": event.participant},
                metadata={"turn": turn_id, "session": event.session},
                timestamp=_at(event.t),
            )
            self._turns[turn_id] = (handle, ctx)
            self._said[turn_id] = []
            return
        if event.kind == "turn" and event.data.get("phase") == "end":
            turn_id = str(event.data.get("id"))
            self._end(turn_id, event.data.get("outcome"))
            return
        if event.kind == "output" and event.turn in self._said and event.data.get("text"):
            self._said[event.turn].append(str(event.data["text"]))
        ctx = self._turns[event.turn][1] if event.turn in self._turns else self._fork()
        if event.kind == "role" and event.duration_s is not None:
            details = {k: v for k, v in event.data.items() if k not in ("answer", "prompt")}
            inputs = {"prompt": event.data.get("prompt", ""), **details}
            ctx.run(self._closed_scope, event, relay.ScopeType.Llm, inputs, {"answer": event.data.get("answer")})
            return
        if event.kind == "tool" and event.duration_s is not None:
            ctx.run(
                self._closed_scope,
                event,
                relay.ScopeType.Tool,
                {"arguments": event.data.get("arguments")},
                {"result": event.data.get("result")},
            )
            return
        ctx.run(relay.scope.event, f"{self._prefix}.{event.kind}", data=self._mark(event), timestamp=_at(event.t))

    @property
    def open_turns(self) -> list[str]:
        """Turns whose ``Agent`` scope is still open (a start seen, no end yet)."""
        return list(self._turns)

    def close(self) -> None:
        """End every turn scope still open, with outcome ``closed`` (the sink is removed mid-turn)."""
        for turn_id in list(self._turns):
            self._end(turn_id, "closed")

    def _end(self, turn_id: str, outcome: Any) -> None:
        entry = self._turns.pop(turn_id, None)
        said = self._said.pop(turn_id, [])
        if entry is not None:
            handle, ctx = entry
            ctx.run(self._relay.scope.pop, handle, output={"said": " ".join(said), "outcome": outcome})

    def _fork(self) -> contextvars.Context:
        # Forking reads the caller's current scope, which creates a stack if there is none; do that in a
        # throwaway copy so the caller's context is left as it was.
        return contextvars.copy_context().run(self._relay.fork_asyncio_context)

    def _closed_scope(self, event: Event, scope_type: Any, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        relay = self._relay
        handle = relay.scope.push(
            event.name or event.kind,
            scope_type,
            input=_json(inputs),
            metadata={"turn": event.turn, "task": event.task},
            timestamp=_at(event.t),
        )
        relay.scope.pop(handle, output=_json(outputs), timestamp=_at(event.t + (event.duration_s or 0.0)))

    def _mark(self, event: Event) -> dict[str, Any]:
        return {
            "name": event.name,
            "session": event.session,
            "participant": event.participant,
            "turn": event.turn,
            "task": event.task,
            "parent": event.parent,
            "duration_s": event.duration_s,
            **_json(dict(event.data)),
        }


@asynccontextmanager
async def relay_exporters(
    *,
    atof_dir: str | Path | None = None,
    atif_dir: str | Path | None = None,
    session: str = "xr-ai",
    agent: str = "xr-ai-app",
) -> AsyncIterator[None]:
    """Register Relay's ATOF (raw events, JSONL) and/or ATIF (trajectory, JSON) exporters for the block.

    ATOF goes to ``<atof_dir>/`` as Relay names it; ATIF is written to ``<atif_dir>/<session>.json`` on exit.
    """
    import nemo_relay  # noqa: PLC0415

    atof = atif = None
    if atof_dir is not None:
        Path(atof_dir).mkdir(parents=True, exist_ok=True)
        config = nemo_relay.AtofExporterConfig()
        config.sink_type = "file"
        config.output_directory = str(atof_dir)
        config.mode = nemo_relay.AtofExporterMode.Overwrite
        config.filename = f"{session}.atof.jsonl"
        atof = nemo_relay.AtofExporter(config)
        atof.register(f"xr-ai-atof-{session}")
    if atif_dir is not None:
        atif = nemo_relay.AtifExporter(session, agent, "next")
        atif.register(f"xr-ai-atif-{session}")
    try:
        yield
    finally:
        await nemo_relay.subscribers.flush_async()
        if atof is not None:
            atof.force_flush()
            atof.deregister(f"xr-ai-atof-{session}")
            atof.shutdown()
        if atif is not None:
            out = Path(atif_dir)  # type: ignore[arg-type]
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{session}.json").write_text(atif.export_json(), encoding="utf-8")
            atif.deregister(f"xr-ai-atif-{session}")
