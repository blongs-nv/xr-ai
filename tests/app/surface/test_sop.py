# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr_ai_kit.sop: the step functions on scripted models, and the runner's run, skip, stop and controls."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel
from xr_ai_app import App, Participant, Turn
from xr_ai_app._base.app import AppConfig
from xr_ai_app.testing import FakeModels, solid_frame
from xr_ai_kit import DestinationResult
from xr_ai_kit.sop import Criteria, Run, SopRunner, identify, judge, timer, verify

WATER = Criteria(
    "Is there water in the kettle?",
    positive="Water is visible inside the kettle.",
    negative="The kettle is visibly empty.",
    unknown="No kettle, or its inside cannot be seen.",
)


class Reading(BaseModel):
    value: float | None = None
    unit: str | None = None


class Tea(BaseModel):
    tea_type: str | None = None


class Settings(BaseModel):
    temperature_c: float | None = None
    duration_s: float | None = None


def _app() -> App:
    return App(AppConfig(config_dir=Path(".")))


async def _until(s, condition, *, frames: bool = False, timeout_s: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        assert loop.time() < deadline, f"not in time; spoken: {s.spoken()}; errors: {s.trace.events(kind='error')}"
        if frames:
            await s.show_frame(solid_frame("white"))
        await asyncio.sleep(0.01)
        await s.settle()


# ------------------------------------------------------------------------------- one frame


async def test_judge_describes_the_frame_then_decides_against_the_criteria() -> None:
    models = FakeModels({"observe": ["The kettle is half full of water."], "decide": ["positive"]})
    app = _app()
    async with app.test_session(models=models) as s:
        await s.join()
        frame = solid_frame("white")
        j = await judge(
            frame, observe=app.role("observe"), decide=app.role("decide"), question="Water?", criteria=WATER
        )
    assert (j.label, j.description) == ("positive", "The kettle is half full of water.")
    assert "half full" in models.calls("decide")[0].context


async def test_judge_with_a_schema_skips_the_decision_when_the_reading_cannot_be_assessed() -> None:
    models = FakeModels({"observe": [{"value": 68}, ValueError("not json")], "decide": []})
    app = _app()
    async with app.test_session(models=models) as s:
        await s.join()
        frame = solid_frame("white")
        args = dict(observe=app.role("observe"), decide=app.role("decide"), question="Read it.", criteria=WATER)
        no_unit = await judge(frame, **args, observation_model=Reading, can_assess=lambda r: r.unit is not None)
    assert (no_unit.label, no_unit.reason, no_unit.parsed) == ("unknown", "not assessable", Reading(value=68))
    assert models.calls("decide") == []


# ------------------------------------------------------------------------------- steps


async def test_verify_completes_on_n_of_the_last_m_decidable_frames_and_unknowns_take_no_slot() -> None:
    models = FakeModels(
        {
            "observe": ["empty", "can't see", "water", "water"],
            "decide": ["negative", "unknown", "positive", "positive"],
        }
    )
    app = _app()
    states = []

    async def on_state(state) -> None:
        states.append(state)

    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        check = asyncio.ensure_future(
            verify(
                p.frames,
                observe=app.role("observe"),
                decide=app.role("decide"),
                question="Water?",
                criteria=WATER,
                required=2,
                window=3,
                interval_s=0.01,
                on_state=on_state,
            )
        )
        await _until(s, check.done, frames=True)
        result = check.result()
    assert result.status == "completed"
    assert result.window == ("negative", "positive", "positive")
    assert [st.positives for st in states] == [0, 0, 1, 2]
    assert len({j.frame_us for j in result.judgments}) == 4  # never the same frame twice


async def test_timer_counts_down_and_reports() -> None:
    states = []

    async def on_state(state) -> None:
        states.append(state)

    final = await timer(0.05, interval_s=0.02, on_state=on_state)
    assert final.status == "completed" and final.remaining_s == 0
    assert states[0].status == "running" and states[-1] is final


async def test_identify_keeps_looking_until_the_reading_is_enough_then_derives_the_result() -> None:
    models = FakeModels(
        {
            "observe": [{}, {"tea_type": "green tea"}],
            "answer": [{"temperature_c": 80, "duration_s": 120}],
        }
    )
    app = _app()

    async def retrieve(tea: Tea) -> list[str]:
        return [f"{tea.tea_type}: 80 C for 2 minutes"]

    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        job = asyncio.ensure_future(
            identify(
                p.frames,
                observe=app.role("observe"),
                answer=app.role("answer"),
                question="Read the label.",
                visual_model=Tea,
                result_model=Settings,
                instructions="Give brewing settings.",
                ready=lambda tea: tea.tea_type is not None,
                retrieve=retrieve,
                interval_s=0.01,
            )
        )
        await _until(s, job.done, frames=True)
        result = job.result()
    assert result.result == Settings(temperature_c=80, duration_s=120)
    assert result.attempts == 2 and result.passages == ("green tea: 80 C for 2 minutes",)
    assert "80 C for 2 minutes" in models.calls("answer")[0].prompt


# ------------------------------------------------------------------------------- the runner


def _runner(app: App, procedure, models: FakeModels) -> SopRunner:
    runner = SopRunner("tea", procedure, title="Tea guidance", decide=models.role("decide"))
    app.add(runner)
    return runner


async def test_a_run_goes_through_its_steps_and_the_conversation_sees_its_summary() -> None:
    app = _app()

    async def procedure(run: Run) -> None:
        await run.note(temperature_c=80)
        await run.step("heat", lambda: timer(0.02, interval_s=0.01, on_state=run.report()), say="Heat the water.")
        await run.step("steep", lambda: timer(0.02, interval_s=0.01, on_state=run.report()))

    models = FakeModels()
    runner = _runner(app, procedure, models)
    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        await runner.start(p)
        await _until(s, lambda: not runner.running(p))
        summary = runner.context.get(p)
        run = runner.run(p)
    assert s.spoken() == ["Starting Tea guidance.", "Heat the water.", "Tea guidance is complete."]
    assert run.status == "completed"
    assert [(st.id, st.status) for st in run.steps] == [("heat", "done"), ("steep", "done")]
    assert "completed" in summary and '"temperature_c": 80' in summary and "heat: done" in summary


async def test_skip_cancels_only_the_current_step_and_the_procedure_continues() -> None:
    app = _app()
    results = []

    async def procedure(run: Run) -> None:
        results.append(await run.step("wait", lambda: asyncio.Event().wait()))
        results.append(await run.step("next", lambda: asyncio.sleep(0, result="ok")))

    models = FakeModels()
    runner = _runner(app, procedure, models)
    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        await runner.start(p)
        await _until(s, lambda: runner.run(p).current is not None)
        assert await runner.skip(p) == "Skipping this step."
        await _until(s, lambda: not runner.running(p))
        run = runner.run(p)
    assert results == [None, "ok"]
    assert [(st.id, st.status) for st in run.steps] == [("wait", "skipped"), ("next", "done")]


async def test_stop_cancels_the_run_and_its_step_and_restart_starts_a_new_run() -> None:
    app = _app()
    cancelled = []

    async def blocked() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append("step")
            raise

    async def procedure(run: Run) -> None:
        await run.step("wait", blocked)

    models = FakeModels()
    runner = _runner(app, procedure, models)
    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        await runner.start(p)
        await _until(s, lambda: runner.run(p).current is not None)
        assert await runner.start(p) == "Tea guidance is already running."
        await runner.stop(p)
        assert (runner.run(p).status, cancelled) == ("stopped", ["step"])
        assert await runner.stop(p) == "Tea guidance is not running."
        await runner.restart(p)
        await _until(s, lambda: runner.run(p).current is not None)
        assert runner.run(p).number == 2
        await runner.stop(p)


async def test_a_failing_step_fails_the_run_and_says_so() -> None:
    app = _app()

    async def broken() -> None:
        raise ValueError("camera service down")

    async def procedure(run: Run) -> None:
        await run.step("look", broken)

    models = FakeModels()
    runner = _runner(app, procedure, models)
    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        await runner.start(p)
        await _until(s, lambda: runner.run(p).status == "failed")
        run = runner.run(p)
    assert s.spoken()[-1].startswith("Tea guidance could not continue")
    assert run.steps[0].status == "failed"


async def test_with_a_plan_the_summary_says_what_comes_next() -> None:
    models = FakeModels()
    app = _app()
    plan = [("heat", "Heat the water."), ("steep", "Steep the tea.")]

    async def procedure(run: Run) -> None:
        await run.step("heat", lambda: asyncio.Event().wait())

    runner = SopRunner("tea", procedure, title="Tea guidance", decide=models.role("decide"), plan=plan)
    app.add(runner)
    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        before = runner.summary(p)
        await runner.start(p)
        await _until(s, lambda: runner.run(p).current is not None)
        during = runner.summary(p)
        await runner.stop(p)
    assert "Its steps, in order, once started:" in before and "- steep: Steep the tea." in before
    assert "- heat (Heat the water.): running." in during
    assert "- steep (Steep the tea.): ahead." in during


async def test_classify_names_the_control_without_performing_it() -> None:
    models = FakeModels({"decide": ["stop"]})
    app = _app()

    async def procedure(run: Run) -> None:
        await run.step("wait", lambda: asyncio.Event().wait())

    runner = _runner(app, procedure, models)
    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        await runner.start(p)
        assert await runner.classify("stop it", p) == "stop"
        assert runner.running(p)  # classified, not performed
        assert "Tea guidance (run 1): running" in models.calls("decide")[0].context
        await runner.stop(p)


@pytest.mark.parametrize(("choice", "kind"), [("start", "answer"), ("none", "pass")])
async def test_the_control_destination_runs_controls_and_passes_everything_else(choice: str, kind: str) -> None:
    models = FakeModels({"decide": [choice]})
    app = _app()

    async def procedure(run: Run) -> None:
        await run.step("wait", lambda: asyncio.Event().wait())

    runner = _runner(app, procedure, models)
    async with app.test_session(models=models) as s:
        await s.join()
        p: Participant = s.participant()
        result = await runner.control(Turn("let's make tea", 0, "t1"), p)
        assert isinstance(result, DestinationResult) and result.kind == kind
        assert runner.running(p) is (choice == "start")
        if runner.running(p):
            await runner.stop(p)
    assert "let's make tea" in models.calls("decide")[0].context
