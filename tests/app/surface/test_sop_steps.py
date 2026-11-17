# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr_ai_kit.sop's step types, their wording file, and the step evaluation (scene, scripted io, scoring)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from PIL import Image
from pydantic import BaseModel, ConfigDict
from xr_ai_app import App
from xr_ai_app._base.app import AppConfig
from xr_ai_app.devkit import run_virtual
from xr_ai_app.testing import FakeModels, solid_frame
from xr_ai_kit.sop import (
    Check,
    Criteria,
    Escalated,
    EscalationSettings,
    Escalator,
    ReadStep,
    Run,
    SopRunner,
    StepDefinition,
    StepDefinitions,
    WaitStep,
    WatchSettings,
    WatchStep,
    evaluation,
)
from xr_ai_kit.sop.escalation import escalation_scoring
from xr_ai_kit.sop.evaluation import Line, StepTrace
from xr_ai_kit.sop.step_types.read import read_scoring
from xr_ai_kit.sop.step_types.wait import wait_scoring
from xr_ai_kit.sop.step_types.watch import watch_scoring
from xr_ai_kit.sop.step_types.watch.watch_scoring import Window

WATER = Criteria("Is there water?", "Water is visible.", "The kettle is empty.", "The kettle's inside cannot be seen.")
FAST = WatchSettings(interval_s=0.01, required=2, window=4)


def _helper(models: FakeModels, guidance: str, **settings) -> Escalator:
    return Escalator(
        guidance, models.role("reason"), settings=EscalationSettings(**{"every_s": 0.01, "max_lines": 1, **settings})
    )


def _eyes(models: FakeModels) -> dict[str, object]:
    """A watch step's models: ``observe`` describes a frame, ``decide`` labels it."""
    return {"observe": models.role("observe"), "decide": models.role("decide")}


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
    step = Escalated(
        WatchStep("fill", Check("Water?", WATER), **_eyes(models), settings=FAST),
        _helper(models, "The kettle must be in view."),
    )
    run, results = await _run_alone(models, step)
    assert run.status == "completed" and run.steps[0].status == "done"
    assert results[0].status == "completed"
    assert models.calls("reason")  # it was stuck, the escalation ran and failed


async def test_skipping_a_watch_step_withdraws_its_pending_question() -> None:
    ask = {"missing": "whether there is water", "action": "ask", "words": "Is there water in the kettle?"}
    models = FakeModels({"observe": ["A kettle."] * 300, "decide": ["unknown"] * 300, "reason": [ask]})
    step = Escalated(
        WatchStep("fill", Check("Water?", WATER), **_eyes(models), settings=FAST),
        _helper(models, "The person knows."),
    )

    async def skip(runner, p) -> None:
        await runner.skip(p)

    run, results = await _run_alone(models, step, until=lambda r: r is not None and r.asking is not None, then=skip)
    assert run.steps[0].status == "skipped" and run.asking is None and results == [None]


async def test_a_watch_step_without_an_escalator_never_reasons() -> None:
    models = FakeModels({"observe": ["A kettle."] * 60, "decide": ["unknown"] * 8 + ["positive"] * 40, "reason": []})
    run, _ = await _run_alone(models, WatchStep("fill", Check("Water?", WATER), **_eyes(models), settings=FAST))
    assert run.status == "completed" and models.calls("reason") == []


async def test_a_step_says_its_done_line_when_it_finishes_from_its_result() -> None:
    class Reading(BaseModel):
        value: float | None = None

    models = FakeModels({"observe": [{"value": 81}] * 60, "decide": ["positive"] * 60})
    heat = WatchStep(
        "heat",
        Check("Read the display.", WATER, observation_model=Reading),
        **_eyes(models),
        done="It reads {observation.value:g}, at least {minimum}.",
        settings=FAST,
    ).format(minimum=80)
    app = App(AppConfig(config_dir=Path(".")))

    async def procedure(run: Run) -> None:
        await run.step(heat)

    runner = app.add(SopRunner("t", procedure, title="T", decide=models.role("decide")))
    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        await runner.start(p)
        await _until(s, lambda: not runner.running(p))
        assert "It reads 81, at least 80." in s.spoken()


def test_an_escalator_needs_a_step_that_supports_escalation() -> None:
    with pytest.raises(TypeError, match="WaitStep does not support escalation"):
        Escalated(WaitStep("w", 1.0), _helper(FakeModels(), "The person knows."))


def test_watch_step_scoring() -> None:
    expect = watch_scoring.Expect
    done = StepTrace("s", completed=True, completed_at_s=341.0)
    assert watch_scoring.score(done, expect(completes=True)) == []
    assert watch_scoring.score(done, expect(completes=Window(after=330, before=350))) == []
    assert watch_scoring.score(done, expect(completes=Window(before=300))) == [
        "completed at 341 s, expected within [None, 300.0]"
    ]
    with pytest.raises(ValueError):
        expect.model_validate({"complete": True})  # a misspelled key is an error, not a silent pass
    with pytest.raises(ValueError):
        expect.model_validate({"asks": []})  # the escalator's key, not the watch step's


def test_escalation_scoring_and_a_case_of_an_escalated_step_split_by_owner() -> None:
    expect = escalation_scoring.Expect
    asked = StepTrace("s", completed=True, lines=[Line(48, "ask", "What unit is the display showing?", "F")])
    assert escalation_scoring.score(asked, expect(asks=["unit"])) == []
    assert escalation_scoring.score(asked, expect(says=["kettle"]))[0].startswith("said nothing")
    assert escalation_scoring.score(asked, expect(quiet=True))[0].endswith("expected nothing")
    done = StepTrace("s", completed=True, lines=[Line(9, "done", "Water is visible.")])
    assert escalation_scoring.score(done, expect(quiet=True)) == []  # a step's own done line is not the escalator's

    models = FakeModels()
    step = Escalated(WatchStep("fill", Check("Water?", WATER), **_eyes(models)), _helper(models, "The person knows."))
    scoring = evaluation.scoring_for(step)
    late = StepTrace("s", completed=True, completed_at_s=50.0, lines=asked.lines)
    both = scoring.Expect.model_validate({"completes": False, "asks": ["unit"]})
    assert scoring.score(late, both) == ["completed at 50 s"]  # each key scored by its owner
    with pytest.raises(ValueError):
        scoring.Expect.model_validate({"completes": True, "aks": ["unit"]})  # a key nobody owns


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
    watch = WatchStep("fill", Check("Water?", WATER), **_eyes(models))
    assert evaluation.scoring_for(watch) is watch_scoring
    assert evaluation.scoring_for(WaitStep("w", 1.0)) is wait_scoring

    class Mine:
        id, say = "mine", None

    with pytest.raises(TypeError, match="Mine names no scoring module"):
        evaluation.scoring_for(Mine())  # type: ignore[arg-type]
    Mine.scoring = read_scoring.__name__  # an app's own type names its module, like the kit's
    assert evaluation.scoring_for(Mine()) is read_scoring  # type: ignore[arg-type]
    assert evaluation.scoring_for(watch, {WatchStep: wait_scoring}) is wait_scoring  # an override


def _loads(module: str, prefix: str) -> bool:
    """Whether importing ``module`` in a fresh interpreter loads any module starting with ``prefix``."""
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    probe = f"import sys, {module}; print(any(m.startswith({prefix!r}) for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    return out.strip() == "True"


def test_running_code_never_imports_evaluation() -> None:
    assert not _loads("xr_ai_kit.sop", "xr_ai_kit.sop.evaluation")


def _imports(package: str) -> set[str]:
    """The ``xr_ai_kit.sop`` modules ``package``'s own source imports (relative imports resolved)."""
    import ast  # noqa: PLC0415
    import importlib  # noqa: PLC0415

    root = Path(importlib.import_module(package).__file__).parent
    found: set[str] = set()
    for path in root.rglob("*.py"):
        parts = [*package.split("."), *path.relative_to(root).parent.parts]  # the package the file is in
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:
                base = parts[: len(parts) - node.level + 1]
                found.add(".".join([*base, *([node.module] if node.module else [])]))
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
    return {m for m in found if m.startswith("xr_ai_kit.sop")}


def test_the_core_and_escalation_import_no_step_type() -> None:
    """steps/ knows no step type and no escalation; escalation knows no step type; step types opt into it."""
    core, escalation = _imports("xr_ai_kit.sop.steps"), _imports("xr_ai_kit.sop.escalation")
    assert core and escalation  # the scan sees their imports
    assert not any(m.startswith(("xr_ai_kit.sop.step_types", "xr_ai_kit.sop.escalation")) for m in core), core
    assert not any(m.startswith("xr_ai_kit.sop.step_types") for m in escalation), escalation
    assert "xr_ai_kit.sop.escalation.link" in _imports("xr_ai_kit.sop.step_types.watch")


# ---------------------------------------------------------------------------------- the step definitions


def test_step_definitions_build_steps_from_yaml_and_code(tmp_path: Path) -> None:
    (tmp_path / "steps.yaml").write_text(
        """
fill:
  does: fill the kettle
  say: Fill the kettle to {level}.
  done: Filled to {level}.
  watch:
    question: Is there water?
    criteria: {instructions: "Water at {level}?", positive: "yes", negative: "no", unknown: unclear}
  escalation: The person knows.
steep:
  does: wait
  say: Steeping for {minutes:g} minutes.
  done: Steeped for {minutes:g} minutes.
lines:
  not_identified: The default settings.
""",
        encoding="utf-8",
    )
    steps = StepDefinitions.load(tmp_path / "steps.yaml")
    models = FakeModels()
    fill = WatchStep.define(steps["fill"], **_eyes(models)).format(level="the line")
    assert (fill.say, fill.check.criteria.instructions) == ("Fill the kettle to the line.", "Water at the line?")
    assert steps.unread() == ["fill.escalation"]  # nothing has wired its escalator yet
    helper = Escalator.define(steps["fill"], models.role("reason"), EscalationSettings(context=["dialogue"]))
    assert (helper.guidance, helper.settings.context) == ("The person knows.", ["dialogue"])
    assert helper.but(max_lines=5).settings.max_lines == 5 and helper.settings.max_lines == 2
    steps.check_read()
    with pytest.raises(ValueError, match="'steep' has no 'escalation' section"):
        Escalator.define(steps["steep"], models.role("reason"))
    steep = WaitStep.define(steps["steep"]).lasting(120).format(minutes=2)
    assert (steep.duration_s, steep.say, steep.closing(None)) == (
        120,
        "Steeping for 2 minutes.",
        "Steeped for 2 minutes.",
    )
    assert steps.plan() == [("fill", "fill the kettle"), ("steep", "wait")]
    assert steps.line("not_identified") == "The default settings."
    with pytest.raises(ValueError, match="'steep' has no 'watch' section"):
        WatchStep.define(steps["steep"], **_eyes(models))
    with pytest.raises(KeyError, match="no definition of step 'boil'"):
        steps["boil"]


def test_each_step_uses_its_own_models_and_a_misspelled_key_fails() -> None:
    class Label(BaseModel):
        name: str | None = None

    read = StepDefinition(
        "identify", say="Show the label.", sections={"read": {"question": "Q.", "instructions": "I."}}
    )
    step = ReadStep.define(
        read, observe=FakeModels().role("ocr"), reason=FakeModels().role("big"), visual_model=Label, result_model=Label
    )
    assert (step.observe.name, step.reason.name, step.question) == ("ocr", "big", "Q.")
    criteria = {"instructions": "?", "positive": "y", "negative": "n", "unknown": "u"}
    typo = StepDefinition("fill", sections={"watch": {"question": "Water?", "criteria": {**criteria, "postive": "y"}}})
    with pytest.raises(ValueError, match="postive"):
        WatchStep.define(typo, **_eyes(FakeModels()))  # inside a section: when the step is built
    misnamed = StepDefinitions({"fill": StepDefinition("fill", sections={"escalaton": "x"})}, {})
    with pytest.raises(ValueError, match="fill.escalaton"):
        misnamed.check_read()  # a section nothing reads: once the app has built its steps


def test_an_apps_own_step_type_reads_its_own_section_of_the_same_file(tmp_path: Path) -> None:
    (tmp_path / "steps.yaml").write_text("scan:\n  say: Scan it.\n  scan: {pattern: '[0-9]{12}'}\n", encoding="utf-8")

    class ScanWords(BaseModel):
        model_config = ConfigDict(extra="forbid")
        pattern: str

    definition = StepDefinitions.load(tmp_path / "steps.yaml")["scan"]
    assert definition.section("scan", ScanWords).pattern == "[0-9]{12}" and definition.say == "Scan it."


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
    step = Escalated(
        WatchStep("heat", Check("Read the display.", WATER), **_eyes(models), settings=WatchSettings(interval_s=2.0)),
        _helper(models, "The person knows the unit.", every_s=20.0, max_lines=2, context=["dialogue"]),
    )
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


def test_a_case_records_the_steps_closing_line_apart_from_what_an_escalator_says(tmp_path: Path) -> None:
    step = WaitStep("steep", 5.0, done="Steeped for {minutes:g} minutes.").format(minutes=2)
    case = evaluation.Case(name="steeps", scene=[0, 30], expect={"completes": True})
    result = run_virtual(evaluation.run_case(step, case, _recording(tmp_path, 30)))
    assert result.ok, result.problems
    assert [(line.kind, line.text) for line in result.trace.lines] == [("done", "Steeped for 2 minutes.")]
