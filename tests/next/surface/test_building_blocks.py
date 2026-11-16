# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr_ai_next building blocks on their own: context channels, choose and the Router, delegate backends, trace."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from xr_ai_kit import Ack, Admission, DestinationResult, Router, RouterDestination
from xr_ai_kit.routing import choose
from xr_ai_next import Component, ContextLog, ContextValue, Participant, PerConnection, TextSink, Turn
from xr_ai_next._base.app import App, AppConfig
from xr_ai_next.testing import FakeModels
from xr_ai_next.trace import Event, JsonlSink, MemorySink, Tracer


def _app() -> App:
    return App(AppConfig(config_dir=Path(".")), enrollment=PerConnection(admit=None))


def _turn(text: str) -> Turn:
    return Turn(text=text, timestamp_us=0, id="t")


# ------------------------------------------------------------------------------------- context


async def test_context_value_scopes_and_budget() -> None:
    p, q = Participant("p"), Participant("q")
    value = ContextValue("alert", scope="session", budget_bytes=10)
    await value.set(p, "a very long alert text")
    assert value.get(p).endswith("...") and len(value.get(p).encode()) <= 10
    assert value.get(q) == ""
    shared = ContextValue("room", scope="app", budget_bytes=100)
    await shared.set(None, "everyone sees this")
    assert await shared.view(q) == "everyone sees this"
    with pytest.raises(ValueError, match="pass the participant"):
        await value.set(None, "x")


async def test_context_value_session_resets_on_reconnect() -> None:
    p = Participant("p")
    value = ContextValue("v", scope="session", budget_bytes=100)
    p._add_connection("c1")  # type: ignore[arg-type]
    await value.set(p, "first connection")
    p._remove_connection("c1")  # type: ignore[arg-type]
    p._add_connection("c2")  # type: ignore[arg-type]
    assert value.get(p) == ""


async def test_context_log_summarizes_in_the_background() -> None:
    models = FakeModels({"sum": ["the user likes tea"]})
    log = ContextLog("notes", scope="participant", budget_bytes=60, summarize=models.role("sum"))
    p = Participant("p")
    for i in range(6):
        await log.publish(p, f"note number {i}")
    await log.settle(p)
    view = await log.view(p)
    assert view.startswith("(earlier, summarized) the user likes tea")
    assert "note number 5" in view
    assert len(models.calls("sum")) >= 1


async def test_context_log_without_summarize_drops_oldest() -> None:
    log = ContextLog("notes", scope="participant", budget_bytes=40)
    p = Participant("p")
    for i in range(6):
        await log.publish(p, f"entry {i}")
    view = await log.view(p)
    assert "earlier entries dropped" in view and "entry 5" in view and "entry 0" not in view


# ------------------------------------------------------------------------------- choose / Router


class Two(Component):
    """Two custom destinations built in ``__init__`` from the component's own methods (focus on a, admission on b)."""

    def __init__(self) -> None:
        super().__init__()
        self.ran: list[str] = []
        self.open = False
        self.a = RouterDestination("a", "Option A.", self._a, focused=self.a_focus)
        self.b = RouterDestination("b", "Option B.", self._b, admit=self.b_admit)

    async def _a(self, turn: Turn, p: Participant) -> DestinationResult | None:
        self.ran.append("a")
        return DestinationResult("pass", reason="not mine")

    def a_focus(self, p: Participant) -> bool:
        return self.open

    async def _b(self, turn: Turn, p: Participant) -> None:
        self.ran.append("b")

    async def b_admit(self, turn: Turn, p: Participant) -> Admission:
        return Admission("decline", reason="no camera") if "decline" in turn.text else Admission()


async def test_choose_with_a_mapping_and_exclusions() -> None:
    models = FakeModels({"decide": ["y"]})
    p = Participant("p")
    choice = await choose(_turn("hi"), p, {"x": "X", "y": "Y"}, decide=models.role("decide"), reasons=["z: no"])
    assert (choice.name, choice.via, choice.destination) == ("y", "model", None)
    assert "Already tried" in models.calls("decide")[0].context
    only = await choose(_turn("hi"), p, {"x": "X", "y": "Y"}, decide=models.role("decide"), exclude=["y"])
    assert (only.name, only.via) == ("x", "only")
    with pytest.raises(ValueError):
        await choose(_turn("hi"), p, {"x": "X"}, decide=models.role("decide"), exclude=["x"])


async def test_choose_focus_skips_the_model() -> None:
    two = Two()
    two.open = True
    models = FakeModels()
    choice = await choose(_turn("hi"), Participant("p"), [two.a, two.b], decide=models.role("decide"))
    assert (choice.name, choice.via) == ("a", "focus")
    assert models.calls("decide") == []


async def test_router_retries_after_pass_and_decline_then_default() -> None:
    two = Two()
    sink = TextSink()

    async def default(turn: Turn, p: Participant) -> None:
        await p.say("default", to=sink)

    models = FakeModels({"decide": ["a"]})
    router = Router(
        decide=models.role("decide"),
        destinations=[two.a, two.b],
        default=RouterDestination("d", "default", default),
    )
    p = Participant("p")
    await router.handle(_turn("please decline"), p)
    # a passed; b (the only one left) declined; the default answered
    assert two.ran == ["a"]
    assert sink.said["p"] == ["default"]


async def test_router_ack_when_slow() -> None:
    sink = TextSink()
    models = FakeModels({"ack": ["One sec."]})

    async def slow(turn: Turn, p: Participant) -> None:
        await asyncio.sleep(0.1)
        await p.reply("done", to=sink)

    only = RouterDestination("only", "the only one", slow)
    router = Router(
        decide=models.role("decide"),
        destinations=[only],
        default=only,
        output=sink,
        ack=Ack(role=models.role("ack"), delay_s=0.02),
    )
    await router.handle(_turn("hi"), Participant("p"))
    assert sink.said["p"] == ["One sec.", "done"]


async def test_one_shot_destination_routes_a_plain_handler() -> None:
    from xr_ai_kit import OneShotDestination, RouterDestination

    sink = TextSink()

    async def answer(turn: Turn, p: Participant) -> None:
        await p.reply(f"answer: {turn.text}", to=sink)

    chat = OneShotDestination("chat", "Chat.", answer)
    assert isinstance(chat, RouterDestination) and chat.describe(Participant("p")) == "Chat."
    router = Router(decide=FakeModels().role("decide"), destinations=[chat], default=chat)
    await router.handle(_turn("hi"), Participant("p"))
    assert sink.said["p"] == ["answer: hi"]


def _mode(models: FakeModels, sink: TextSink, took: list[str]):  # noqa: ANN202
    from xr_ai_kit import OneShotDestination, StartStopDestination

    async def take(turn: Turn, p: Participant) -> None:
        took.append(turn.text)
        await p.reply("Noted.", to=sink)

    async def answer(turn: Turn, p: Participant) -> None:
        await p.reply("an answer", to=sink)

    mode = StartStopDestination(
        "notes",
        "Note taking.",
        take,
        decide=models.role("decide"),
        output=sink,
        start="Start notes.",
        stop="Stop notes.",
        content="A note.",
        started="Taking notes.",
        stopped="Notes stopped.",
    )
    chat = OneShotDestination("chat", "Chat.", answer)
    return mode, Router(decide=models.role("decide"), destinations=[chat, mode], default=chat)


async def test_start_stop_destination_is_a_mode_with_focus() -> None:
    sink, took = TextSink(), []
    # pick notes, start | (focus) content | (focus) other -> declined, chat is left | (focus) stop | pick chat
    models = FakeModels({"decide": ["notes", "start", "content", "other", "stop", "chat"]})
    mode, router = _mode(models, sink, took)
    p = Participant("p")
    for text in ["start taking notes", "buy milk", "what time is it?", "stop taking notes", "hello"]:
        await router.handle(_turn(text), p)
        if text == "start taking notes":
            assert mode.is_on(p)
    assert sink.said["p"] == ["Taking notes.", "Noted.", "an answer", "Notes stopped.", "an answer"]
    assert took == ["buy milk"]  # the handler saw only the content
    assert not mode.is_on(p)
    calls = models.calls("decide")
    assert [tuple(c.choices) for c in calls] == [
        ("chat", "notes"),
        ("start", "content", "other"),  # off
        ("stop", "content", "other"),  # on: no routing call, it holds focus
        ("stop", "content", "other"),
        ("stop", "content", "other"),
        ("chat", "notes"),
    ]
    assert json.loads(calls[2].context) == {"request": "buy milk", "destination": {"name": "notes", "focused": True}}


async def test_start_stop_destination_content_starts_the_mode_and_other_is_declined() -> None:
    sink, took = TextSink(), []
    models = FakeModels({"decide": ["notes", "content", "notes", "other"]})
    mode, router = _mode(models, sink, took)
    p, q = Participant("p"), Participant("q")
    await router.handle(_turn("take a note: buy milk"), p)
    assert took == ["take a note: buy milk"] and mode.is_on(p) and sink.said["p"] == ["Noted."]
    await router.handle(_turn("what is the weather?"), q)  # picked by mistake while off: declined, chat answers
    assert not mode.is_on(q) and sink.said["q"] == ["an answer"]


def test_router_validation() -> None:
    two = Two()
    models = FakeModels()
    with pytest.raises(ValueError, match="output"):
        Router(decide=models.role("d"), destinations=[two.a], default=two.b, ack=Ack(role=models.role("a")))
    with pytest.raises(ValueError, match="duplicate"):
        Router(decide=models.role("d"), destinations=[two.a, two.a], default=two.b)


# ------------------------------------------------------------------------------------ delegate


# --------------------------------------------------------------------------------------- trace


async def test_trace_ids_follow_spans_and_tasks(tmp_path: Path) -> None:
    tracer = Tracer()
    memory = tracer.add(MemorySink())
    tracer.add(JsonlSink(tmp_path))
    failing_calls = []

    def bad(event: Event) -> None:
        failing_calls.append(event)
        raise RuntimeError("sink broke")

    tracer.add(bad)
    async with tracer.span("outer", k=1) as extra:
        extra["more"] = 2
        tracer.emit("custom", "inside")
    inner, outer = memory.all
    assert inner.parent == outer.data["span"]
    assert outer.data["k"] == 1 and outer.data["more"] == 2 and outer.duration_s is not None
    assert len(failing_calls) == 2  # a failing sink is called again, only logged once
    (file,) = tmp_path.glob("trace-*.jsonl")
    assert [json.loads(line)["kind"] for line in file.read_text().splitlines()] == ["custom", "span"]


def _turn_event(kind_phase: str, turn: str, t: float, **data: object) -> Event:
    return Event(
        kind="turn",
        name="",
        id=f"e-{turn}-{kind_phase}",
        t=t,
        turn=turn,
        data={"id": turn, "phase": kind_phase, **data},
    )


def test_relay_sink_writes_a_turn_and_leaves_the_callers_stack_alone() -> None:
    """A turn is a scope in its own forked Relay context: the caller's stack (here, pytest's) is left alone."""
    import nemo_relay
    from xr_ai_next.trace import RelaySink

    before = nemo_relay._scope_stack_var.get(None)
    sink = RelaySink()
    sink(_turn_event("start", "t1", 0.0, text="hi"))
    assert sink.open_turns == ["t1"]
    sink(Event(kind="role", name="answer", id="e2", t=0.1, turn="t1", duration_s=0.1, data={"prompt": "p"}))
    sink(Event(kind="output", name="", id="e3", t=0.2, turn="t1", data={"text": "hello"}))
    sink(Event(kind="context", name="notes", id="e4", t=0.3, data={}))  # outside a turn: a mark
    sink(_turn_event("end", "t1", 0.4))
    assert sink.open_turns == []
    assert nemo_relay._scope_stack_var.get(None) is before  # this context's stack neither created nor replaced


def test_relay_sink_turns_may_end_out_of_order() -> None:
    """Two participants' turns overlap and the first one started ends first. On one shared Relay stack the
    first pop would fail ("not at the top of the stack"); each turn has its own context, so both close."""
    from xr_ai_next.trace import RelaySink

    sink = RelaySink()
    sink(_turn_event("start", "a", 0.0))
    sink(_turn_event("start", "b", 0.1))
    sink(_turn_event("end", "a", 0.2))
    sink(_turn_event("end", "b", 0.3))
    assert sink.open_turns == []


def test_relay_sink_close_ends_open_turns() -> None:
    """A turn whose end never came (the sink removed mid-turn) is closed, so no Relay scope outlives the sink."""
    from xr_ai_next.trace import RelaySink

    sink = RelaySink()
    sink(_turn_event("start", "t1", 0.0))
    sink(_turn_event("start", "t2", 0.1))
    sink.close()
    assert sink.open_turns == []
