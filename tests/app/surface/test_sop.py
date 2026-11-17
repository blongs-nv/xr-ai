# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr_ai_kit.sop: the step functions on scripted models, and the runner's run, skip, stop and controls."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import BaseModel
from xr_ai_app import App, Participant, Turn, clock
from xr_ai_app._base.app import AppConfig
from xr_ai_app.testing import FakeModels, solid_frame
from xr_ai_kit import DestinationResult
from xr_ai_kit.sop import (
    Check,
    Criteria,
    EscalationSettings,
    Facts,
    Judgment,
    Link,
    ReportEscalator,
    Run,
    SopRunner,
    StepContext,
    escalation_for,
    identify,
    judge,
    timer,
    verify,
)
from xr_ai_kit.sop.step_types.watch import outbound

WATER = Criteria(
    "Is there water in the kettle?",
    positive="Water is visible inside the kettle.",
    negative="The kettle is visibly empty.",
    unknown="No kettle, or its inside cannot be seen.",
)


WATER_CHECK = Check("Water?", WATER)


class Reading(BaseModel):
    value: float | None = None
    unit: str | None = None


class Display(BaseModel):
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
        j = await judge(frame, WATER_CHECK, observe=app.role("observe"), decide=app.role("decide"))
    assert (j.label, j.description) == ("positive", "The kettle is half full of water.")
    assert "half full" in models.calls("decide")[0].context


async def test_judge_with_a_schema_skips_the_decision_when_the_reading_cannot_be_assessed() -> None:
    models = FakeModels({"observe": [{"value": 68}, ValueError("not json")], "decide": []})
    app = _app()
    async with app.test_session(models=models) as s:
        await s.join()
        frame = solid_frame("white")
        check = Check(
            "Read it.",
            WATER,
            needs=Display,
            observation_model=Reading,
            can_assess=lambda r, known: (r.unit or (known and known.unit)) is not None,
        )
        args = dict(observe=app.role("observe"), decide=app.role("decide"))
        no_unit = await judge(frame, check, **args)
    assert (no_unit.label, no_unit.reason, no_unit.parsed) == ("unknown", "not assessable", Reading(value=68))
    assert models.calls("decide") == []


async def test_facts_reach_the_check_in_code_and_the_decision_not_the_observation() -> None:
    models = FakeModels({"observe": [{"value": 176}], "decide": ["positive"]})
    app = _app()
    check = Check(
        "Read it.",
        WATER,
        needs=Display,
        observation_model=Reading,
        can_assess=lambda r, known: (r.unit or (known and known.unit)) is not None,
    )
    facts = Facts(Display(unit="fahrenheit"), ('Asked the person: "Unit?" They answered: "Fahrenheit"',))
    async with app.test_session(models=models) as s:
        await s.join()
        j = await judge(solid_frame(), check, observe=app.role("observe"), decide=app.role("decide"), facts=facts)
    assert j.label == "positive"
    assert "fahrenheit" in models.calls("decide")[0].context
    assert "Fahrenheit" not in models.calls("observe")[0].prompt


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
                WATER_CHECK,
                observe=app.role("observe"),
                decide=app.role("decide"),
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


def _escalating_check() -> Check:
    return Check(
        "Read the display.",
        WATER,
        needs=Display,
        observation_model=Reading,
        can_assess=lambda r, known: (r.unit or (known and known.unit)) is not None,
    )


async def test_the_fast_loop_reads_context_added_while_it_runs() -> None:
    models = FakeModels({"observe": [{"value": 176}] * 8, "decide": ["positive", "positive"]})
    app = _app()
    context = StepContext()
    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        job = asyncio.ensure_future(
            verify(
                p.frames,
                _escalating_check(),
                observe=app.role("observe"),
                decide=app.role("decide"),
                context=context,
                interval_s=0.01,
            )
        )
        await _until(s, lambda: len(models.calls("observe")) >= 3, frames=True)
        assert models.calls("decide") == []  # no unit known: nothing to decide
        context.add('Asked the person: "Unit?" They answered: "F"', Display(unit="fahrenheit"))
        await _until(s, job.done, frames=True)
        result = job.result()
    assert result.status == "completed" and result.facts.known == Display(unit="fahrenheit")


def _reports(*labels: str) -> list[Judgment]:
    return [Judgment(label, '{"value": 75}', i, i) for i, label in enumerate(labels)]  # type: ignore[arg-type]


UNIT_GUIDANCE = "Some displays print no unit; the person knows which unit theirs uses."


class _IO:
    """An escalator's EscalationIO: the person, and what is going on."""

    def __init__(self, *, ask=None, say=None, context: dict[str, str] | None = None) -> None:
        self._ask, self._say, self._context = ask, say, context or {}
        self.sources: list[list[str]] = []

    async def ask(self, question: str) -> str | None:
        return await self._ask(question)

    async def say(self, text: str) -> None:
        await self._say(text)

    async def situation(self, sources) -> str:
        self.sources.append(list(sources))
        return "\n\n".join(self._context[s] for s in sources if s in self._context)


def _linked(context: StepContext, labels: tuple[str, ...]) -> Link:
    """A link as a running watch step attaches it: its outbound context from ``context`` and ``labels``, inbound
    context into ``context``."""
    started, link = clock.now(), Link()
    link.attach(
        lambda: outbound("heat", _escalating_check(), _reports(*labels), context.facts, started),
        lambda inbound: context.add(inbound.note, inbound.known),
    )
    return link


def _escalator(role, *, reader=None, **settings) -> ReportEscalator:
    return ReportEscalator(UNIT_GUIDANCE, role, reader, EscalationSettings(every_s=0.01, min_gap_s=0, **settings))


async def test_an_escalator_reads_the_steps_outbound_context_and_answers_only_inbound() -> None:
    models = FakeModels(
        {
            "reviewer": [
                {"missing": "the unit", "action": "ask", "words": "Celsius or Fahrenheit?"},
                {"action": "none"},
            ],
            "reader": [{"unit": "fahrenheit"}],
        }
    )
    context = StepContext()
    asked: list[str] = []

    async def ask(question: str) -> str | None:
        asked.append(question)
        return "Fahrenheit"

    escalator = _escalator(models.role("reviewer"), reader=models.role("reader"))
    side = asyncio.ensure_future(
        escalator.run(_linked(context, ("negative", "unknown", "unknown", "unknown")), _IO(ask=ask))
    )
    for _ in range(200):
        if len(models.calls("reviewer")) >= 2:
            break
        await asyncio.sleep(0.01)
    side.cancel()
    assert asked == ["Celsius or Fahrenheit?"]
    assert context.facts.known == Display(unit="fahrenheit")
    assert "They answered" in context.facts.text()
    calls = models.calls("reviewer")
    assert all(call.thinking for call in calls)  # reasoning, off the fast loop
    assert all(call.frame is None for call in calls)  # it reads the reports, never frames
    assert "unknown" in calls[0].prompt and '{"value": 75}' in calls[0].prompt
    assert UNIT_GUIDANCE in calls[0].prompt


async def test_an_escalator_tells_the_person_what_to_do_without_waiting_for_an_answer() -> None:
    models = FakeModels({"reviewer": [{"missing": "the kettle", "action": "tell", "words": "Show me the kettle."}]})
    said: list[str] = []

    async def say(text: str) -> None:
        said.append(text)

    async def ask(question: str) -> str | None:
        raise AssertionError("an instruction is not a question")

    context = StepContext()
    escalator = _escalator(models.role("reviewer"), max_lines=1)
    labels = ("unknown", "unknown", "unknown", "unknown")
    await asyncio.wait_for(escalator.run(_linked(context, labels), _IO(ask=ask, say=say)), 2)
    assert said == ["Show me the kettle."]
    assert context.facts.text() == ""


async def test_an_escalator_stays_quiet_while_the_check_still_decides() -> None:
    models = FakeModels({"reviewer": []})
    escalator = _escalator(models.role("reviewer"))
    side = asyncio.ensure_future(
        escalator.run(_linked(StepContext(), ("unknown", "negative", "negative", "unknown")), _IO())
    )
    await asyncio.sleep(0.1)
    side.cancel()
    assert models.calls("reviewer") == []  # a stray unknown is noise: no model call at all


async def test_each_escalator_reads_only_the_context_it_is_configured_with() -> None:
    labels = ("unknown", "unknown", "unknown", "unknown")
    io_context = {"run": "Tea guidance (run 1): running.", "dialogue": "## dialogue\nuser: it's in Fahrenheit"}
    prompts = {}
    for sources in ([], ["dialogue"], ["run", "dialogue"]):
        models = FakeModels({"reviewer": [{"action": "tell", "words": "Show me the kettle."}]})
        escalator = _escalator(models.role("reviewer"), max_lines=1, context=sources)
        io = _IO(say=lambda text: asyncio.sleep(0), context=io_context)
        await asyncio.wait_for(escalator.run(_linked(StepContext(), labels), io), 2)
        prompts[tuple(sources)] = models.calls("reviewer")[0].prompt
        assert io.sources == ([list(sources)] if sources else [])
    assert "Fahrenheit" not in prompts[()] and "run 1" not in prompts[()]
    assert "Fahrenheit" in prompts[("dialogue",)] and "run 1" not in prompts[("dialogue",)]
    both = prompts[("run", "dialogue")]
    assert "The step 'heat' has been running for" in both
    assert both.index("run 1") < both.index("Fahrenheit")  # in the order configured


async def test_without_guidance_there_is_no_escalation() -> None:
    models = FakeModels({"reviewer": []})
    unknowns = _reports("unknown", "unknown", "unknown", "unknown")
    step = outbound("fill", WATER_CHECK, unknowns, Facts(), clock.now())
    assert await escalation_for(step, models.role("reviewer"), guidance="") is None
    assert models.calls("reviewer") == []


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


async def test_a_step_asks_and_the_guide_sorts_each_reply_while_the_question_is_pending() -> None:
    models = FakeModels({"decide": ["none", "answer"]})
    app = _app()
    answers: list[str | None] = []

    async def procedure(run: Run) -> None:
        async def unit() -> None:
            answers.append(await run.ask("Celsius or Fahrenheit?", timeout_s=5))

        await run.step("heat", unit)

    runner = _runner(app, procedure, models)
    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        await runner.start(p)
        await _until(s, lambda: runner.run(p).asking is not None)
        assert await runner.control.holds_focus(p)  # the reply reaches the guide first
        assert "Waiting for the person's answer to: Celsius or Fahrenheit?" in runner.summary(p)
        side = await runner.control(Turn("How hot should it be?", 0, "t1"), p)
        assert side.kind == "pass" and runner.run(p).asking  # still pending
        await runner.control(Turn("Fahrenheit, I think", 0, "t2"), p)
        await _until(s, lambda: not runner.running(p))
        run = runner.run(p)
    assert answers == ["Fahrenheit, I think"]
    assert run.status == "completed" and run.asking is None
    assert "pending_question" in models.calls("decide")[0].context
    assert "answer" in models.calls("decide")[0].choices


async def test_skipping_while_a_question_is_pending_cancels_the_step_and_the_question() -> None:
    models = FakeModels({"decide": ["skip"]})
    app = _app()

    async def procedure(run: Run) -> None:
        await run.step("heat", lambda: run.ask("Celsius or Fahrenheit?", timeout_s=5))

    runner = _runner(app, procedure, models)
    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        await runner.start(p)
        await _until(s, lambda: runner.run(p).asking is not None)
        await runner.control(Turn("skip this", 0, "t"), p)
        await _until(s, lambda: not runner.running(p))
        run = runner.run(p)
    assert [(st.id, st.status) for st in run.steps] == [("heat", "skipped")]
    assert run.asking is None


async def test_an_unanswered_question_times_out_to_none() -> None:
    app = _app()
    got: list[str | None] = []

    async def procedure(run: Run) -> None:
        async def unit() -> None:
            got.append(await run.ask("Celsius or Fahrenheit?", timeout_s=0.05))

        await run.step("heat", unit)

    models = FakeModels()
    runner = _runner(app, procedure, models)
    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        await runner.start(p)
        await _until(s, lambda: not runner.running(p))
    assert got == [None]


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
        context = json.loads(models.calls("decide")[0].context)
        assert context == {"request": "stop it", "guide": "Tea guidance: running"}  # nothing else
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
