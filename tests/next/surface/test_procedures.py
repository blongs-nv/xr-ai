# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr_ai_procedures: the steps on their own (fake time), and guided procedures end to end on fake models."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, Field
from xr_ai_next import App, PerConnection, Voice
from xr_ai_next._base.app import AppConfig
from xr_ai_next.testing import FakeModels, solid_frame
from xr_ai_procedures import Facts, GuidedProcedure, Run, Sop, StepDef, Timer
from xr_ai_procedures.perception import Done, Missing, Needs, Stuck, Tier, Verify, Watching, read_check
from xr_ai_procedures.run import fill, told_model

SAMPLE = Path(__file__).resolve().parents[3] / "xr-ai-next" / "samples" / "sdk-tour" / "10-guided-procedure"
START = "Okay, starting printer pre-flight."
BED = "First, make sure the print bed is clear."
CLEAR = {"seen": "An empty print bed.", "state": "clear"}
CLOSED = {"seen": "The door is shut.", "state": "closed"}


# ------------------------------------------------------------------------------- helpers


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


class ScriptedEyes:
    """``Verify``'s frame source: one scripted frame (or ``None``) per look; each look takes a second of fake time."""

    def __init__(self, frames: list[object | None], clock: FakeClock) -> None:
        self._frames = list(frames)
        self._clock = clock

    async def next(self) -> object | None:
        self._clock.now += 1.0
        return self._frames.pop(0) if self._frames else None


def _report_into(lines: list[str]):
    async def report(text: str) -> None:
        lines.append(text)

    return report


async def until(s, condition, *, timeout_s: float = 3.0, frames: bool = False) -> None:
    """Settle until ``condition()`` holds; with ``frames``, the camera shows a new frame every round."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        assert loop.time() < deadline, f"not in time; spoken: {s.spoken()}; errors: {s.trace.events(kind='error')}"
        if frames:
            await s.show_frame(solid_frame("white"))
        await asyncio.sleep(0.01)
        await s.settle()


def _sample(load_sample, *, stuck_after_s: float = 20.0):
    """Sample 10, with fast looks and a short warm-up."""
    m = load_sample("10-guided-procedure")
    m.WATCHING = Watching(required_positives=2, window=3, stuck_after_s=stuck_after_s, interval_s=0.01)
    m.procedure.defs["warm_up"] = StepDef("warm_up", lambda p, run, spec: (Timer(0.05, interval_s=0.01), ()))
    return m


def _events(s, name: str) -> list[dict]:
    return [dict(e.data) for e in s.trace.events(kind="procedure", name=name)]


# ------------------------------------------------------------------------------- the steps on their own


async def test_verify_needs_required_positives_in_the_window() -> None:
    clock = FakeClock()
    verdicts = iter(["yes", "unknown", "no", "no", "yes", "yes"])
    verify = Verify(
        eyes=ScriptedEyes(["f"] * 6, clock),
        observe=lambda frame: "a look",
        judge=lambda seen: next(verdicts),
        watching=Watching(required_positives=2, window=3, stuck_after_s=60),
        now=clock,
    )
    lines: list[str] = []
    outcome = await verify.run(_report_into(lines))
    # yes | (unknown: no place) | no | no -> [yes, no, no] | yes -> [no, no, yes] | yes -> [no, yes, yes]: done
    assert outcome == Done("a look")
    assert len(lines) == 6


async def test_verify_asks_for_a_missing_fact_and_gets_stuck() -> None:
    clock = FakeClock()
    answers = iter([Missing("unit"), Missing("other"), Missing("unit"), Missing("unit")])
    verify = Verify(
        eyes=ScriptedEyes(["f"] * 4, clock),
        observe=lambda frame: "a reading",
        judge=lambda seen: next(answers),
        watching=Watching(stuck_after_s=60, missing_after=2),
        now=clock,
    )
    assert await verify.run(_report_into([])) == Needs("unit")

    never = ScriptedEyes([], clock)
    unseen = Verify(eyes=never, observe=str, judge=lambda s: "yes", watching=Watching(stuck_after_s=5), now=clock)
    assert await unseen.run(_report_into([])) == Stuck("unseen")

    unclear = Verify(
        eyes=ScriptedEyes(["f"] * 99, clock),
        observe=lambda frame: "blurry",
        judge=lambda seen: "unknown",
        watching=Watching(stuck_after_s=5),
        now=clock,
    )
    assert await unclear.run(_report_into([])) == Stuck("unclear")


async def test_verify_escalates_to_a_tier_and_tolerates_failed_looks() -> None:
    clock = FakeClock()

    def quick(frame: object) -> object:
        raise ConnectionError("model down")  # a failed look is unknown, not a failed step

    verify = Verify(
        eyes=ScriptedEyes(["f"] * 10, clock),
        observe=quick,
        judge=lambda seen: "yes",
        watching=Watching(required_positives=2, window=2, stuck_after_s=60),
        escalation=(Tier(2.5, lambda frame: "careful answer", name="careful"),),
        now=clock,
    )
    lines: list[str] = []
    assert await verify.run(_report_into(lines)) == Done("careful answer")
    assert lines == ["careful answer (careful)"] * 2


async def test_timer_counts_down_on_fake_time() -> None:
    clock = FakeClock()
    lines: list[str] = []
    timer = Timer(10, interval_s=4, now=clock, sleep=clock.sleep)
    assert await timer.run(_report_into(lines)) == Done(10)
    assert lines == ["10 seconds left", "6 seconds left", "2 seconds left", "time is up"]
    assert clock.now == 10
    with pytest.raises(ValueError):
        Timer(0)


def test_sop_defaults_and_wording() -> None:
    sop = Sop.load(SAMPLE / "sop.yaml")
    assert sop.name == "printer pre-flight"
    assert sop.messages["start"] == "Okay, starting {procedure}."
    custom = Sop.model_validate({"steps": [{"id": "a", "instruction": "i", "enter": "e", "done": "d"}],
                                 "messages": {"start": "Go."}})  # fmt: skip
    assert custom.messages["start"] == "Go." and custom.messages["stopped"] == "Okay, I've stopped."
    assert fill("Heat to {temperature} for {minutes}.", {"temperature": 93}) == "Heat to 93 for {minutes}."
    assert fill("the unit is {}", {}, "celsius") == "the unit is celsius"
    with pytest.raises(ValueError, match="confirm"):
        Sop.model_validate({"steps": [{"id": "a", "instruction": "i", "enter": "e", "done": "d", "unable": "confirm"}]})


def test_two_steps_cannot_own_one_fact() -> None:
    class A(Facts):
        unit: str | None = None

    class B(Facts):
        unit: str | None = None

    model, owner = told_model({"a": A})
    assert owner == {"unit": "a"} and set(model.model_fields) == {"unit", "step_done"}
    with pytest.raises(ValueError, match="unit"):
        told_model({"a": A, "b": B})


# ------------------------------------------------------------------------------- sample 10, end to end


async def test_sample_runs_start_to_finish(load_sample) -> None:
    m = _sample(load_sample)
    models = FakeModels({"decide": ["procedure", "start"], "perceive": [CLEAR, CLEAR, CLOSED, CLOSED]})
    async with m.app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        assert "It has not been started." in m.procedure.channel.get(p)
        await s.say("let's do the pre-flight checklist")
        await until(s, lambda: len(s.spoken()) == 5, frames=True)
        assert s.spoken() == [
            START,
            BED,
            "The bed is clear. Close the printer door.",
            "The door is closed. Warming up for ninety seconds.",
            "Warm-up done. You can start the print. That was the last step.",
        ]
        run = m.procedure.run_of(p)
        await until(s, lambda: run.finished.is_set())
        assert run.status == "completed"
        assert run.outcomes == {"bed_clear": "done", "door_closed": "done", "warm_up": "done"}
        assert run.results["bed_clear"].state == "clear"
        assert [e["step"] for e in _events(s, "step_done")] == ["bed_clear", "door_closed", "warm_up"]
        assert _events(s, "run_end")[-1]["status"] == "completed"
        assert "Status: completed." in m.procedure.channel.get(p)
        assert [c.response_format.__name__ for c in models.calls("perceive")] == ["Bed", "Bed", "Door", "Door"]


async def test_stop(load_sample) -> None:
    m = _sample(load_sample)
    models = FakeModels({"decide": ["procedure", "start", "procedure", "stop"]})
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("start the checklist")
        await until(s, lambda: len(s.spoken()) == 2)
        await s.say("stop")
        await s.settle()
        assert s.spoken() == [START, BED, "Okay, I've stopped."]
        p = s.participant()
        assert m.procedure.run_of(p).status == "stopped"
        assert "Status: stopped." in m.procedure.channel.get(p)
        assert [(e["op"]) for e in _events(s, "control")] == ["start", "stop"]


async def test_next_and_back(load_sample) -> None:
    m = _sample(load_sample)
    models = FakeModels({"decide": ["procedure", "start", "procedure", "next", "procedure", "back"]})
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("start")
        await until(s, lambda: len(s.spoken()) == 2)
        await s.say("skip this one")
        await until(s, lambda: len(s.spoken()) == 4)
        run = m.procedure.run_of(s.participant())
        assert run.index == 1 and run.outcomes == {"bed_clear": "skipped"}
        await s.say("go back")
        await until(s, lambda: len(s.spoken()) == 6)
        assert s.spoken() == [START, BED, "Okay, moving on.", "Close the printer door.", "Okay, going back.", BED]
        assert run.index == 0 and run.outcomes == {}
        assert [e["step"] for e in _events(s, "step_start")] == ["bed_clear", "door_closed", "bed_clear"]


async def test_a_step_that_cannot_see_asks_and_takes_yes(load_sample) -> None:
    m = _sample(load_sample, stuck_after_s=0.05)
    models = FakeModels({"decide": ["procedure", "start", "info"], "answer_with_reasoning": [{"step_done": True}]})
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("start")
        await until(s, lambda: len(s.spoken()) == 3)  # no camera: Stuck("unseen"), then the confirm question
        question = (
            "I can't verify this with the camera. Is the print bed clear? Say yes or no, or say next to skip this step."
        )
        assert s.spoken()[2] == question
        p = s.participant()
        assert f"The procedure asked: {question}" in m.procedure.channel.get(p)
        await s.say("yes")
        await until(s, lambda: len(s.spoken()) >= 5)
        assert s.spoken()[3:5] == ["Got it: that step is done.", "The bed is clear. Close the printer door."]
        assert m.procedure.run_of(p).outcomes == {"bed_clear": "confirmed"}
        assert [e.data["via"] for e in s.trace.events(kind="route")] == ["model", "focus"]  # the answer: no routing
        assert _events(s, "step_stuck")[0] == {
            "procedure": "procedure", "step": "bed_clear", "reason": "unseen", "policy": "confirm"
        }  # fmt: skip
        (facts_call,) = models.calls("answer_with_reasoning")
        assert "Is the print bed clear?" in facts_call.prompt and facts_call.response_format.__name__ == "Told"


async def test_the_conversation_reads_the_procedure_channel(load_sample) -> None:
    m = _sample(load_sample)
    models = FakeModels(
        {
            "decide": ["procedure", "start", "conversation"],
            "answer_with_reasoning": ["You're on the first step: clear the print bed."],
        }
    )
    async with m.app.test_session(models=models) as s:
        await s.join()
        await s.say("start")
        await until(s, lambda: len(s.spoken()) == 2)
        await s.say("what step am I on?")
        await s.settle()
        assert s.spoken()[-1] == "You're on the first step: clear the print bed."
        system = models.calls("answer_with_reasoning")[0].system
        assert "## printer pre-flight" in system
        assert "Steps: bed_clear (current), door_closed, warm_up." in system
        assert "Current step: bed_clear: the user should now clear everything off the print bed." in system
        assert "warm_up (a 90-second timer) has not started" in system  # the sample's describe= line


async def test_the_summary_names_the_current_step_and_who_reported(load_sample) -> None:
    m = _sample(load_sample)
    run = Run(facts={}, index=2, outcomes={"bed_clear": "done", "door_closed": "done"}, seen="70 seconds left")
    summary = m.procedure.summary(run)
    assert "Current step: warm_up: the user should now wait while the printer warms up." in summary
    assert "warm_up reported: 70 seconds left" in summary
    assert "last reported" not in summary


# ------------------------------------------------------------------------------- facts, on a small app


class Unit(Facts):
    unit: Literal["celsius", "fahrenheit"] | None = Field(None, description="The unit the kettle display shows.")


class Reading(BaseModel):
    value: float | None = None
    unit: Literal["celsius", "fahrenheit"] | None = None


KETTLE_SOP = {
    "name": "kettle check",
    "steps": [
        {"id": "fill", "instruction": "fill the kettle", "enter": "Fill the kettle.", "done": "Filled."},
        {
            "id": "heat",
            "instruction": "heat the water",
            "enter": "Heat it to {target} degrees.",
            "done": "Hot enough.",
            "asks": {"unit": "Which unit does the display show?"},
        },
    ],
}


def _kettle_app(tmp_path: Path, models: FakeModels) -> tuple[App, GuidedProcedure]:
    app = App(AppConfig(config_dir=tmp_path), enrollment=PerConnection(admit=None))
    voice = app.add(Voice())
    watching = Watching(required_positives=2, window=2, stuck_after_s=20, missing_after=2, interval_s=0.01)

    def heat(p, run, spec):
        told = run.facts["heat"].unit

        def judge(r: Reading):
            unit = told or r.unit
            if unit is None:
                return Missing("unit")
            celsius = (r.value - 32) * 5 / 9 if unit == "fahrenheit" else r.value
            return "yes" if celsius >= 90 else "no"

        step = read_check(
            frames=p.frames, perceive=models.role("perceive"), question="Read the display.", reads=Reading,
            judge=judge, text=lambda r: f"the display reads {r.value:g}", watching=watching,
        )  # fmt: skip
        return step, (told,)

    procedure = app.add(
        GuidedProcedure(
            sop=Sop.model_validate(KETTLE_SOP),
            steps=[
                StepDef("fill", lambda p, run, spec: (Timer(30), ())),
                StepDef(
                    "heat",
                    heat,
                    facts=Unit,
                    phrases={"unit": "the display shows {}"},
                    needs=frozenset({"unit"}),
                    words=lambda run: {"target": 90},
                ),
            ],
            output=voice,
            decide=models.role("decide"),
            reason=models.role("reason"),
            name="kettle",
        )
    )
    app.on_turn(procedure.control)
    return app, procedure


async def test_a_fact_told_early_reaches_the_step_that_owns_it(tmp_path: Path) -> None:
    hot = {"value": 200, "unit": None}  # 200 with no unit shown: only the told unit makes it judgeable
    models = FakeModels(
        {"decide": ["start", "info", "next"], "reason": [{"unit": "fahrenheit"}], "perceive": [hot, hot]}
    )
    app, procedure = _kettle_app(tmp_path, models)
    async with app.test_session(models=models) as s:
        await s.join()
        await s.say("start")
        await until(s, lambda: len(s.spoken()) == 2)
        await s.say("by the way the kettle shows fahrenheit")
        await s.settle()
        run = procedure.run_of(s.participant())
        assert s.spoken()[-1] == "Got it: the display shows fahrenheit."
        assert run.index == 0 and run.facts["heat"].unit == "fahrenheit"  # held by its owner, step 2
        assert len(_events(s, "step_start")) == 1  # the running timer was not rebuilt
        await s.say("next")
        await until(s, lambda: run.finished.is_set(), frames=True)
        assert s.spoken()[3:] == ["Okay, moving on.", "Heat it to 90 degrees.", "Hot enough. That was the last step."]
        assert run.outcomes == {"fill": "skipped", "heat": "done"}


async def test_a_step_that_needs_a_fact_asks_for_it(tmp_path: Path) -> None:
    unitless = {"value": 95, "unit": None}
    models = FakeModels(
        {
            "decide": ["start", "next", "info"],
            "reason": [{"unit": "celsius"}],
            "perceive": [unitless, unitless, unitless, unitless],
        }
    )
    app, procedure = _kettle_app(tmp_path, models)
    async with app.test_session(models=models) as s:
        await s.join()
        await s.say("start")
        await until(s, lambda: len(s.spoken()) == 2)
        await s.say("next")
        await until(s, lambda: len(s.spoken()) == 5, frames=True)
        assert s.spoken()[3:] == ["Heat it to 90 degrees.", "Which unit does the display show?"]
        run = procedure.run_of(s.participant())
        assert run.pending == "unit"
        await s.say("celsius")
        await until(s, lambda: run.finished.is_set(), frames=True)
        assert s.spoken()[5:] == ["Got it: the display shows celsius.", "Hot enough. That was the last step."]
        assert _events(s, "step_needs")[0]["fact"] == "unit"


# ------------------------------------------------------------------------------- when a step cannot observe


class Seen(BaseModel):
    seen: str
    state: Literal["there", "gone", "cannot_tell"]


def _look_app(tmp_path: Path, models: FakeModels, unable: str) -> tuple[App, GuidedProcedure]:
    app = App(AppConfig(config_dir=tmp_path), enrollment=PerConnection(admit=None))
    voice = app.add(Voice())
    watching = Watching(required_positives=2, window=2, stuck_after_s=0.05, interval_s=0.01)

    def look(p, run, spec):
        step = read_check(
            frames=p.frames, perceive=models.role("perceive"), question="Is it there?", reads=Seen,
            judge=lambda r: "yes" if r.state == "there" else "no", watching=watching,
        )  # fmt: skip
        return step, ()

    look_spec = {"id": "look", "instruction": "point the camera at it", "enter": "Show it.", "done": "Seen."}
    sop = {
        "steps": [
            {**look_spec, "unable": unable},
            {"id": "wait", "instruction": "wait", "enter": "Wait.", "done": "Done waiting."},
        ]
    }
    procedure = app.add(
        GuidedProcedure(
            sop=Sop.model_validate(sop),
            steps=[StepDef("look", look), StepDef("wait", lambda p, run, spec: (Timer(30), ()))],
            output=voice,
            decide=models.role("decide"),
            reason=models.role("reason"),
        )
    )
    app.on_turn(procedure.control)
    return app, procedure


async def test_unable_skip_says_so_and_moves_on(tmp_path: Path) -> None:
    models = FakeModels({"decide": ["start"]})
    app, procedure = _look_app(tmp_path, models, "skip")
    async with app.test_session(models=models) as s:
        await s.join()
        await s.say("start")
        await until(s, lambda: len(s.spoken()) == 3)  # no camera: stuck, skipped
        assert s.spoken()[1:] == ["Show it.", "I couldn't verify that step, so I'm moving on. Wait."]
        assert procedure.run_of(s.participant()).outcomes == {"look": "skipped"}


async def test_unable_show_asks_to_show_it_and_keeps_looking(tmp_path: Path) -> None:
    there = {"seen": "It is there.", "state": "there"}
    models = FakeModels({"decide": ["start"], "perceive": [there, there]})
    app, procedure = _look_app(tmp_path, models, "show")
    async with app.test_session(models=models) as s:
        await s.join()
        await s.say("start")
        await until(s, lambda: len(s.spoken()) == 3)  # no camera: stuck, so it asks to see it
        assert s.spoken()[2] == (
            "I can't tell what the camera shows. Please point the camera at it, with it in view of the camera, "
            "or say next to skip this step."
        )
        run = procedure.run_of(s.participant())
        await until(s, lambda: run.outcomes.get("look") == "done", frames=True)  # rebuilt, and it sees it now
        assert _events(s, "step_stuck")[0]["policy"] == "show"
        await until(s, lambda: s.spoken()[-1] == "Seen. Wait.")
