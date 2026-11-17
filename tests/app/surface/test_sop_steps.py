# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr_ai_kit.sop's step types, their wording file, and the step evaluation (scene, scripted io, scoring)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from PIL import Image
from xr_ai_app import App
from xr_ai_app._base.app import AppConfig
from xr_ai_app.devkit import run_virtual
from xr_ai_app.testing import FakeModels, solid_frame
from xr_ai_kit.sop import (
    Check,
    Criteria,
    EscalationSettings,
    Escalator,
    Roles,
    Run,
    SopRunner,
    StepDefinitions,
    WaitStep,
    WatchSettings,
    WatchStep,
    evaluation,
)
from xr_ai_kit.sop.evaluation import Line, StepTrace
from xr_ai_kit.sop.read import read_scoring
from xr_ai_kit.sop.wait import wait_scoring
from xr_ai_kit.sop.watch import watch_scoring
from xr_ai_kit.sop.watch.watch_scoring import Window

WATER = Criteria("Is there water?", "Water is visible.", "The kettle is empty.", "The kettle's inside cannot be seen.")
FAST = WatchSettings(interval_s=0.01, required=2, window=4)


def _helper(models: FakeModels, guidance: str, **settings) -> Escalator:
    return Escalator(
        guidance, models.role("reason"), settings=EscalationSettings(**{"every_s": 0.01, "max_lines": 1, **settings})
    )


def _roles(models: FakeModels) -> Roles:
    return Roles(models.role("observe"), models.role("decide"), models.role("reason"))


async def _until(s, condition, timeout_s: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        assert loop.time() < deadline, f"not in time; errors: {s.trace.events(kind='error')}"
        await s.show_frame(solid_frame("white"))
        await asyncio.sleep(0.01)
        await s.settle()


async def _run_alone(models: FakeModels, step, until=None, then=None):
    """Run ``step`` as the only step of a procedure; return the run."""
    app = App(AppConfig(config_dir=Path(".")))
    results: list[object] = []

    async def procedure(run: Run) -> None:
        results.append(await run.step(step))

    runner = app.add(SopRunner("t", procedure, title="T", decide=models.role("decide")))
    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        await runner.start(p)
        if until is not None:
            await _until(s, lambda: until(runner.run(p)))
            await then(runner, p)
        await _until(s, lambda: not runner.running(p))
        return runner.run(p), results


# ---------------------------------------------------------------------------------- the watch step


async def test_a_watch_step_completes_on_the_fast_check_and_a_failing_escalation_cannot_fail_it() -> None:
    models = FakeModels(
        {
            "observe": ["A kettle."] * 60,
            "decide": ["unknown"] * 12 + ["positive"] * 40,
            "reason": [ValueError("the reasoning model is down")] * 5,
        }
    )
    step = WatchStep("fill", Check("Water?", WATER), _roles(models), settings=FAST).escalating(
        _helper(models, "The kettle must be in view.")
    )
    run, results = await _run_alone(models, step)
    assert run.status == "completed" and run.steps[0].status == "done"
    assert results[0].status == "completed"
    assert models.calls("reason")  # it was stuck, the escalation ran and failed


async def test_skipping_a_watch_step_withdraws_its_pending_question() -> None:
    ask = {"missing": "whether there is water", "action": "ask", "words": "Is there water in the kettle?"}
    models = FakeModels({"observe": ["A kettle."] * 300, "decide": ["unknown"] * 300, "reason": [ask]})
    step = WatchStep("fill", Check("Water?", WATER), _roles(models), settings=FAST).escalating(
        _helper(models, "The person knows.")
    )

    async def skip(runner, p) -> None:
        await runner.skip(p)

    run, results = await _run_alone(models, step, until=lambda r: r is not None and r.asking is not None, then=skip)
    assert run.steps[0].status == "skipped" and run.asking is None and results == [None]


async def test_a_watch_step_without_an_escalator_never_reasons() -> None:
    models = FakeModels({"observe": ["A kettle."] * 60, "decide": ["unknown"] * 8 + ["positive"] * 40, "reason": []})
    run, _ = await _run_alone(models, WatchStep("fill", Check("Water?", WATER), _roles(models), settings=FAST))
    assert run.status == "completed" and models.calls("reason") == []


def test_watch_step_scoring() -> None:
    expect = watch_scoring.Expect
    done = StepTrace("s", completed=True, completed_at_s=341.0)
    assert watch_scoring.score(done, expect(completes=True)) == []
    assert watch_scoring.score(done, expect(completes=Window(after=330, before=350))) == []
    assert watch_scoring.score(done, expect(completes=Window(before=300))) == [
        "completed at 341 s, expected within [None, 300.0]"
    ]
    asked = StepTrace("s", lines=[Line(48, "ask", "What unit is the display showing?", "F")])
    assert watch_scoring.score(asked, expect(asks=["unit"])) == []
    assert watch_scoring.score(asked, expect(says=["kettle"]))[0].startswith("said nothing")
    assert watch_scoring.score(asked, expect(quiet=True))[0].endswith("expected nothing")
    with pytest.raises(ValueError):
        expect.model_validate({"complete": True})  # a misspelled key is an error, not a silent pass


def test_read_and_wait_step_scoring() -> None:
    class Read:
        visual = type("V", (), {"tea_type": "green tea"})()
        result = type("R", (), {"temperature_c": 80})()

    trace = StepTrace("identify", completed=True, result=Read())
    expect = read_scoring.Expect(completes=True, reads={"tea_type": ["green"]}, derives={"temperature_c": []})
    assert read_scoring.score(trace, expect) == []
    assert read_scoring.score(trace, read_scoring.Expect(reads={"tea_type": ["black"]})) == [
        "read tea_type='green tea', expected one of ['black']"
    ]
    assert wait_scoring.score(StepTrace("w"), wait_scoring.Expect(completes=True)) == ["did not complete"]


def test_evaluation_finds_each_step_types_scoring_and_an_apps_own() -> None:
    models = FakeModels()
    watch = WatchStep("fill", Check("Water?", WATER), _roles(models))
    assert evaluation.scoring_for(watch) is watch_scoring
    assert evaluation.scoring_for(WaitStep("w", 1.0)) is wait_scoring

    class Mine:
        id, say = "mine", None

    with pytest.raises(TypeError, match="no scoring for Mine"):
        evaluation.scoring_for(Mine())  # type: ignore[arg-type]
    assert evaluation.scoring_for(Mine(), {Mine: read_scoring}) is read_scoring  # type: ignore[arg-type]


def test_running_code_never_imports_evaluation() -> None:
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    probe = "import sys, xr_ai_kit.sop; print(any(m.startswith('xr_ai_kit.sop.evaluation') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


# ---------------------------------------------------------------------------------- the step definitions


def test_step_definitions_build_steps_from_yaml_and_code(tmp_path: Path) -> None:
    (tmp_path / "steps.yaml").write_text(
        """
fill:
  does: fill the kettle
  say: Fill the kettle to {level}.
  question: Is there water?
  criteria: {instructions: "Water at {level}?", positive: "yes", negative: "no", unknown: unclear}
  escalation: The person knows.
steep:
  does: wait
  say: Steeping for {minutes:g} minutes.
lines:
  done: Done after {minutes:g} minutes.
""",
        encoding="utf-8",
    )
    words = StepDefinitions.load(tmp_path / "steps.yaml")
    models = FakeModels()
    fill = words.watch("fill", _roles(models)).format(level="the line")
    assert (fill.say, fill.check.criteria.instructions) == ("Fill the kettle to the line.", "Water at the line?")
    assert fill.escalator is None  # wired where the step is built
    helper = words.escalator("fill", models.role("reason"), EscalationSettings(context=["dialogue"]))
    assert (helper.guidance, helper.settings.context) == ("The person knows.", ["dialogue"])
    assert helper.but(max_lines=5).settings.max_lines == 5 and helper.settings.max_lines == 2
    with pytest.raises(ValueError, match="escalation words"):
        words.escalator("steep", models.role("reason"))
    steep = words.wait("steep").lasting(120).format(minutes=2)
    assert (steep.duration_s, steep.say) == (120, "Steeping for 2 minutes.")
    assert words.plan() == [("fill", "fill the kettle"), ("steep", "wait")]
    assert words.line("done", minutes=2) == "Done after 2 minutes."
    with pytest.raises(ValueError, match="needs question and criteria"):
        words.watch("steep", _roles(models))


# ---------------------------------------------------------------------------------- the evaluation


def _recording(tmp_path: Path, seconds: int) -> evaluation.Recording:
    Image.new("RGB", (8, 8), "gray").save(tmp_path / "f.jpg")
    return evaluation.Recording([(float(t), tmp_path / "f.jpg") for t in range(seconds)])


def test_a_scene_plays_parts_back_to_back_on_its_own_clock(tmp_path: Path) -> None:
    recording = _recording(tmp_path, 60)
    one = recording.shots([10, 13])
    assert [(s.t_s, s.source_s) for s in one] == [(10, 10), (11, 11), (12, 12)]
    spliced = recording.shots([[40, 42], [10, 12], [10, 12]])
    assert [s.t_s for s in spliced] == [40, 41, 42, 43, 44, 45]
    assert [s.source_s for s in spliced] == [40, 41, 10, 11, 10, 11]


def test_scene_frames_stream_like_a_camera_and_numbers_repeats_as_new_frames(tmp_path: Path) -> None:
    scene = evaluation.SceneFrames(_recording(tmp_path, 60), [[0, 3], [0, 3]])

    async def take() -> list[tuple[float, int]]:
        scene.begin()
        out = []
        async for frame in scene.every(1.0):
            out.append((round(scene.now()), frame.sequence))
            if len(out) == 6:
                return out
        return out

    seen = run_virtual(take())
    assert [t for t, _ in seen] == [0, 1, 2, 3, 4, 5]
    assert len({seq for _, seq in seen}) == 6  # a repeated part is new frames, not duplicates


def test_a_case_runs_the_real_step_on_a_scene_and_a_scripted_person(tmp_path: Path) -> None:
    """The person answers the step's question; the answer reaches the check, which then completes."""
    ask = {"missing": "the unit", "action": "ask", "words": "Which unit is the display in?"}
    models = FakeModels(
        {
            "observe": ["A kettle reading 176."] * 400,
            "decide": [lambda call: "positive" if "Fahrenheit" in call.context else "unknown"] * 400,
            "reason": [ask] * 3,
        }
    )
    step = WatchStep(
        "heat",
        Check("Read the display.", WATER),
        _roles(models),
        settings=WatchSettings(interval_s=2.0),
    ).escalating(_helper(models, "The person knows the unit.", every_s=20.0, max_lines=2, context=["dialogue"]))
    case = evaluation.Case(
        name="answered",
        scene=[0, 120],
        person={"answers": ["Fahrenheit"]},
        context={"dialogue": "user: is it hot yet?", "run": "not read"},
        expect={"asks": ["unit"], "completes": {"after": 20, "before": 40}},
    )
    result = run_virtual(evaluation.run_case(step, case, _recording(tmp_path, 120)))
    assert result.ok, result.problems
    (line,) = result.trace.lines
    assert (line.kind, line.answer) == ("ask", "Fahrenheit")
    prompt = models.calls("reason")[0].prompt
    assert "is it hot yet?" in prompt and "not read" not in prompt  # only the context it is configured with
