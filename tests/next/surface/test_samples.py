# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The xr-ai-next samples run end to end on fake models (xr_ai_next)."""

from __future__ import annotations

import asyncio

from xr_ai_kit.delegate import FunctionBackend, JobUpdate
from xr_ai_next.testing import FakeModels, ToolCall, solid_frame


async def until(s, condition, *, timeout_s: float = 3.0) -> None:
    """Settle repeatedly until ``condition()`` holds (background work that waits on time, not on models)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        assert loop.time() < deadline, "condition not met in time"
        await asyncio.sleep(0.01)
        await s.settle()


async def test_hello_voice_remembers_the_dialogue(load_sample) -> None:
    m = load_sample("01-hello-voice")
    models = FakeModels({"answer_lightweight": ["Hi there!", "You said hello."]})
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("hello")
        await s.settle()
        await s.say("what did I say?")
        await s.settle()
        assert s.spoken() == ["Hi there!", "You said hello."]
        second = models.calls("answer_lightweight")[-1]
        assert "user: hello" in second.system and "assistant: Hi there!" in second.system
        assert len(s.trace.turns()) == 2
        assert [e.name for e in s.trace.events(kind="role")] == ["answer_lightweight"] * 2


async def test_vision_qa_looks_through_the_tool(load_sample) -> None:
    m = load_sample("02-vision-qa")
    models = FakeModels(
        {
            "answer_with_reasoning": [ToolCall("look", {"question": "what color is the mug?"}), "It's a red mug."],
            "perceive": ["A red mug on a desk."],
        }
    )
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.show_frame(solid_frame("red"))
        await s.say("What color is my mug?")
        await s.settle()
        assert s.spoken() == ["It's a red mug."]
        turn = s.trace.turns()[0]
        (look,) = s.trace.events(kind="tool", turn=turn)
        (seen,) = s.trace.calls("perceive")
        assert seen.parent == look.data["span"]
        assert seen.data["tool"] == "look"
        assert look.data["result"] == "A red mug on a desk."


async def test_vision_qa_without_a_camera(load_sample) -> None:
    m = load_sample("02-vision-qa")
    m.SETTINGS.frame_timeout_s = 0.05
    models = FakeModels(
        {"answer_with_reasoning": [ToolCall("look", {"question": "what is this?"}), "I can't see anything."]}
    )
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("What is this?")
        await until(s, lambda: s.spoken())
        assert models.calls("perceive") == []
        assert s.trace.events(kind="tool")[0].data["result"] == "The camera is not streaming."


async def test_camera_monitor_speaks_after_three_looks(load_sample) -> None:
    m = load_sample("03-camera-monitor")
    m.SETTINGS.look_every_s = 0.01
    yes = {"summary": "A stove with a flame.", "stove_unattended": "yes", "door_open": "no"}
    models = FakeModels({"perceive": [yes] * 6})
    async with m.app.test_session(models=models) as s:
        await s.join()
        for _ in range(12):
            await s.show_frame(solid_frame("orange"))
            await asyncio.sleep(0.03)
            if s.spoken():
                break
        await s.leave()
    assert s.spoken() == ["The stove is on and nobody is near it."]
    assert len(models.calls("perceive")) >= 3


async def test_camera_monitor_on_demand_asks_for_no_pictures_while_paused(load_sample) -> None:
    m = load_sample("03-camera-monitor")
    m.SETTINGS.look_every_s = 0.01
    no = {"summary": "A kitchen.", "stove_unattended": "no", "door_open": "no"}
    models = FakeModels({"perceive": [no] * 200})
    async with m.app.test_session(models=models) as s:
        await s.join()
        s.hub.hold_picture(solid_frame("orange"))  # on-demand camera mode: nothing streamed

        def captures() -> int:
            return [e.kind for e in s.hub.events].count("capture")

        for _ in range(300):  # the first tick waits for a stream (2 s), then a picture per tick
            if captures() >= 3:
                break
            await asyncio.sleep(0.02)
        assert captures() >= 3 and len(models.calls("perceive")) >= 2
        p = s.participant()
        await m.pause_monitor(p)
        await asyncio.sleep(0.05)
        paused_at = captures()
        await asyncio.sleep(0.2)
        assert captures() == paused_at  # paused: it left every(), so no picture is asked for
        await m.resume_monitor(p)
        for _ in range(300):
            if captures() > paused_at:
                break
            await asyncio.sleep(0.02)
        assert captures() > paused_at
        await s.leave()


async def test_notes_mode_holds_focus_and_passes_questions(load_sample) -> None:
    m = load_sample("04-notes-mode")
    models = FakeModels(
        {
            # route the first turn to notes; later turns are notes' (focus) until a pass
            "decide": ["notes", "start", "content", "other", "stop"],
            "answer_with_reasoning": ["Paris is the capital of France."],
        }
    )
    async with m.app.test_session(models=models) as s:
        await s.join()
        for text in ["start taking notes", "buy milk", "what is the capital of France?", "stop"]:
            await s.say(text)
            await s.settle()
        assert s.spoken() == [
            "Okay, I'm taking notes.",
            "Noted.",
            "Paris is the capital of France.",
            "Okay, notes stopped.",
        ]
        routes = [(e.name, e.data["via"]) for e in s.trace.events(kind="route")]
        assert routes == [
            ("notes", "model"),
            ("notes", "focus"),
            ("notes", "focus"),
            ("conversation", "only"),  # notes declined it: the only one left
            ("notes", "focus"),
        ]
        p = s.participant()
        assert m.notes.entries(p) == ["buy milk"]
        assert "buy milk" in models.calls("answer_with_reasoning")[0].system


async def test_manual_dispatch_two_level_menu(load_sample) -> None:
    m = load_sample("05-manual-dispatch")
    models = FakeModels({"decide": ["lights", "dim"]})
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("make it a bit darker")
        await s.settle()
        await s.say("stop")  # a code rule: no model call
        await s.settle()
        assert s.spoken() == ["Dimmed.", "Stopped."]
        assert [c.choices for c in models.calls("decide")] == [("lights", "music", "chat"), ("on", "off", "dim")]


async def test_shared_room_tells_the_others(load_sample) -> None:
    m = load_sample("06-shared-room")
    models = FakeModels(
        {"answer_with_reasoning": [ToolCall("check_item", {"item": "valve closed"}), "Got it, valve checked."]}
    )
    async with m.app.test_session(models=models) as s:
        await s.join("ana")
        await s.join("ben")
        await s.say("the valve is closed", participant_id="ana")
        await s.settle()
        assert s.spoken("ana") == ["Got it, valve checked."]
        assert s.spoken("ben") == ["ana checked valve closed."]
        assert m.app.state.checklist["valve closed"] == "ana"
        assert "valve closed: done by ana" in m.checklist.get(None)


async def test_delegate_reports_progress_and_result(load_sample) -> None:
    m = load_sample("07-delegate-a2a")
    gate = asyncio.Event()

    async def research(request, job):
        yield "searching maps"
        await gate.wait()
        yield JobUpdate("result", "Cafe A, Cafe B, Cafe C.", speak="I found three cafes.")

    m.research._backend = FunctionBackend(research)
    m.research._progress_every_s = None
    models = FakeModels(
        {
            "answer_with_reasoning": [
                ToolCall("start_research", {"request": "quiet cafe near the station"}),
                "I'm on it.",
                "It's still searching maps.",
            ]
        }
    )
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("find me a quiet cafe near the station")
        await s.settle()
        assert s.spoken() == ["I'm on it."]
        await s.say("how is it going?")
        await s.settle()
        assert "searching maps" in models.calls("answer_with_reasoning")[-1].system
        gate.set()
        for _ in range(50):
            await asyncio.sleep(0.01)
            if len(s.spoken()) == 3:
                break
        assert s.spoken()[-1] == "I found three cafes."
        p = s.participant()
        assert "Cafe A" in await m.research.results.view(p)
        (job,) = m.research.jobs(p)
        assert job.state == "done"
        phases = [e.data["phase"] for e in s.trace.events(kind="job")]
        assert phases == ["start", "progress", "done"]
        # the job's events name the turn that started it
        assert {e.turn for e in s.trace.events(kind="job")} == {s.trace.turns()[0]}


async def test_delegate_cancel(load_sample) -> None:
    m = load_sample("07-delegate-a2a")

    async def forever(request, job):
        yield "working"
        await asyncio.Event().wait()

    m.research._backend = FunctionBackend(forever)
    models = FakeModels(
        {
            "answer_with_reasoning": [
                ToolCall("start_research", {"request": "x"}),
                "Started.",
                ToolCall("cancel_research", {}),
                "Cancelled it.",
            ]
        }
    )
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("research x")
        await s.settle()
        await s.say("never mind")
        await s.settle()
        (job,) = m.research.jobs(s.participant())
        assert job.state == "cancelled"
        assert s.spoken() == ["Started.", "Cancelled it."]


async def test_tracing_sample_span_and_sink(load_sample) -> None:
    m = load_sample("11-tracing")
    from xr_ai_next.trace import JsonlSink

    for sink in [s for s in m.app.trace._sinks if isinstance(s, JsonlSink)]:
        m.app.trace.remove(sink)  # the sample's app.yaml writes traces/ next to it; not from tests
    models = FakeModels({"answer_lightweight": ["Sure."]})
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("  hello  ")
        await s.settle()
        (span,) = s.trace.events(kind="span", name="spell-check")
        assert span.data["text_len"] == 9 and span.duration_s is not None
        assert models.calls("answer_lightweight")[0].prompt == "hello"
        assert s.trace.events(kind="first_output")
