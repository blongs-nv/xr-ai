# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""xr_ai_kit.sop's step types, their wording file, and the step evaluation (scene, scripted person, scoring)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from PIL import Image
from pydantic import BaseModel
from xr_ai_app import App
from xr_ai_app._base.app import AppConfig
from xr_ai_app.devkit import run_virtual
from xr_ai_app.testing import FakeModels, solid_frame
from xr_ai_kit.sop import (
    MESSAGES,
    AddedContext,
    Check,
    Criteria,
    EscalationChannel,
    EscalationSettings,
    EscalationWords,
    EscalatorEnvironment,
    ReadStep,
    ReadWords,
    ReasoningEscalator,
    Run,
    SopRunner,
    Step,
    StepEnvironment,
    StepWithEscalator,
    StepWords,
    StructuredObservation,
    TransformSettings,
    TransformStep,
    TransformWords,
    WaitInputs,
    WaitStep,
    WatchModels,
    WatchSettings,
    WatchStep,
    WatchWords,
    check_placeholders,
    decide_escalation,
    evaluation,
    fill_placeholders,
    load_words,
)
from xr_ai_kit.sop.escalation import PROMPT, escalation_scoring
from xr_ai_kit.sop.evaluation import Line, StepTrace
from xr_ai_kit.sop.step_types.read import read_scoring
from xr_ai_kit.sop.step_types.wait import wait_scoring
from xr_ai_kit.sop.step_types.watch import Facts, Judgment, watch_scoring, watch_step_snapshot
from xr_ai_kit.sop.step_types.watch.watch_scoring import Window

WATER = Criteria("Is there water?", "Water is visible.", "The kettle is empty.", "The kettle's inside cannot be seen.")
WATER_SECTION = {
    "question": "Water?",
    "criteria": {
        "instructions": WATER.instructions,
        "positive": WATER.positive,
        "negative": WATER.negative,
        "unknown": WATER.unknown,
    },
}
FAST = WatchSettings(interval_s=0.01, required=2, window=4)


def _water(**lines: str) -> WatchWords:
    """A watch step's words, as load_words makes them from YAML."""
    return WatchWords.model_validate({**WATER_SECTION, **lines})


def _escalator(models: FakeModels, guidance: str, **settings) -> ReasoningEscalator:
    return ReasoningEscalator(
        EscalationWords(guidance=guidance),
        models.role("reason"),
        EscalationSettings(**{"every_s": 0.01, "max_lines": 1, **settings}),
    )


def _camera(models: FakeModels) -> WatchModels:
    """A watch step's models: ``perceive`` describes a frame, ``decide`` labels it."""
    return WatchModels(perceive=models.role("observe"), decide=models.role("decide"))


async def _until(s, condition, timeout_s: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        assert loop.time() < deadline, f"not in time; errors: {s.trace.events(kind='error')}"
        await s.show_frame(solid_frame("white"))
        await asyncio.sleep(0.01)
        await s.settle()


async def _run_alone(models: FakeModels, step, inputs=None, until=None, then=None):
    """Run ``step`` with ``inputs`` as the only step of a procedure; return the run."""
    app = App(AppConfig(config_dir=Path(".")))
    results: list[object] = []

    async def procedure(run: Run) -> None:
        results.append(await run.step(step, inputs))

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
    step = StepWithEscalator(
        WatchStep("fill", _water(), _camera(models), settings=FAST),
        _escalator(models, "The kettle must be in view."),
    )
    run, results = await _run_alone(models, step)
    assert run.status == "completed" and run.steps[0].status == "done"
    assert results[0].status == "completed"
    assert models.calls("reason")  # it was stuck, the escalation ran and failed


async def test_skipping_a_watch_step_withdraws_its_pending_question() -> None:
    ask = {"missing": "whether there is water", "action": "ask", "words": "Is there water in the kettle?"}
    models = FakeModels({"observe": ["A kettle."] * 300, "decide": ["unknown"] * 300, "reason": [ask]})
    step = StepWithEscalator(
        WatchStep("fill", _water(), _camera(models), settings=FAST),
        _escalator(models, "The person knows."),
    )

    async def skip(runner, p) -> None:
        await runner.skip(p)

    run, results = await _run_alone(models, step, until=lambda r: r is not None and r.asking is not None, then=skip)
    assert run.steps[0].status == "skipped" and run.asking is None and results == [None]


async def test_a_watch_step_without_an_escalator_never_reasons() -> None:
    models = FakeModels({"observe": ["A kettle."] * 60, "decide": ["unknown"] * 8 + ["positive"] * 40, "reason": []})
    run, _ = await _run_alone(models, WatchStep("fill", _water(), _camera(models), settings=FAST))
    assert run.status == "completed" and models.calls("reason") == []


async def test_a_step_says_its_done_line_when_it_finishes_from_its_result() -> None:
    class Reading(BaseModel):
        value: float | None = None

    class Target(BaseModel):
        minimum: float

    models = FakeModels({"observe": [{"value": 81}] * 60, "decide": ["positive"] * 60})
    heat = WatchStep(
        "heat",
        _water(done_line="It reads {observation.value:g}, at least {minimum:g}."),
        _camera(models),
        inputs_schema=Target,
        observation=StructuredObservation(Reading),
        settings=FAST,
    )
    app = App(AppConfig(config_dir=Path(".")))

    async def procedure(run: Run) -> None:
        await run.step(heat, Target(minimum=80))

    runner = app.add(SopRunner("t", procedure, title="T", decide=models.role("decide")))
    async with app.test_session(models=models) as s:
        await s.join()
        p = s.participant()
        await runner.start(p)
        await _until(s, lambda: not runner.running(p))
        assert "It reads 81, at least 80." in s.spoken()


def test_an_escalator_needs_a_step_that_supports_escalation() -> None:
    with pytest.raises(TypeError, match="WaitStep does not support escalation"):
        StepWithEscalator(WaitStep("w", StepWords(), duration_s=1.0), _escalator(FakeModels(), "The person knows."))


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
    asked = StepTrace(
        "s", completed=True, lines=[Line(48, "ask", "What unit is the display showing?", "F", by="escalator")]
    )
    assert escalation_scoring.score(asked, expect(asks=["unit"])) == []
    assert escalation_scoring.score(asked, expect(says=["kettle"]))[0].startswith("said nothing")
    assert escalation_scoring.score(asked, expect(quiet=True))[0].endswith("expected nothing")
    done = StepTrace("s", completed=True, lines=[Line(9, "done", "Water is visible.")])
    assert escalation_scoring.score(done, expect(quiet=True)) == []  # a step's own done line is not the escalator's

    models = FakeModels()
    step = StepWithEscalator(WatchStep("fill", _water(), _camera(models)), _escalator(models, "The person knows."))
    scoring = evaluation.scoring_for(step)
    late = StepTrace("s", completed=True, completed_at_s=50.0, lines=asked.lines)
    both = scoring.Expect.model_validate({"completes": False, "asks": ["unit"]})
    assert scoring.score(late, both) == ["completed at 50 s"]  # each key scored by its owner
    with pytest.raises(ValueError):
        scoring.Expect.model_validate({"completes": True, "aks": ["unit"]})  # a key nobody owns


def test_read_and_wait_step_scoring() -> None:
    read = type("Read", (), {"tea_type": "green tea", "temperature_c": 80})()
    trace = StepTrace("identify", completed=True, result=read)
    expect = read_scoring.Expect(completes=True, reads={"tea_type": ["green"], "temperature_c": []})
    assert read_scoring.score(trace, expect) == []
    assert read_scoring.score(trace, read_scoring.Expect(reads={"tea_type": ["black"]})) == [
        "read tea_type='green tea', expected one of ['black']"
    ]
    assert wait_scoring.score(StepTrace("w"), wait_scoring.Expect(completes=True)) == ["did not complete"]


def test_evaluation_finds_each_step_types_scoring_and_an_apps_own() -> None:
    models = FakeModels()
    watch = WatchStep("fill", _water(), _camera(models))
    assert evaluation.scoring_for(watch) is watch_scoring
    assert evaluation.scoring_for(WaitStep("w", StepWords(), duration_s=1.0)) is wait_scoring

    class Mine:
        id, start_line = "mine", None

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


def test_the_core_and_escalation_import_no_step_type_and_the_runner_no_escalation() -> None:
    """steps/ knows no step type and no escalation; escalation knows no step type; the runner knows no escalation;
    step types opt into it."""
    core, escalation = _imports("xr_ai_kit.sop.steps"), _imports("xr_ai_kit.sop.escalation")
    runner = _imports("xr_ai_kit.sop.runner")
    assert core and escalation and runner  # the scan sees their imports
    assert not any(m.startswith(("xr_ai_kit.sop.step_types", "xr_ai_kit.sop.escalation")) for m in core), core
    assert not any(m.startswith("xr_ai_kit.sop.step_types") for m in escalation), escalation
    assert not any(m.startswith("xr_ai_kit.sop.escalation") for m in runner), runner
    assert "xr_ai_kit.sop.escalation.escalation_channel" in _imports("xr_ai_kit.sop.step_types.watch")


async def test_another_escalator_can_send_the_frame_behind_the_evidence_to_a_model_of_its_own() -> None:
    """An escalator is any run(channel, environment): this one looks at the evidence itself, with a model the step
    never uses."""

    class FrameEscalator:
        def __init__(self, model) -> None:
            self.model = model

        async def run(self, channel: EscalationChannel, environment: EscalatorEnvironment) -> None:
            while True:
                await asyncio.sleep(0.01)
                step = channel.snapshot()
                if step and step.evidence and step.evidence[-1].frame is not None:
                    seen = await self.model.ask(step.evidence[-1].frame, "Is there water in the kettle?")
                    channel.add_context(AddedContext(f"A closer look at the frame: {seen}"))
                    return

    models = FakeModels(
        {
            "observe": ["A kettle."] * 60,
            "decide": [lambda call: "positive" if "closer look" in call.context else "unknown"] * 60,
            "frontier": ["The kettle is full."],
        }
    )
    step = StepWithEscalator(
        WatchStep("fill", _water(), _camera(models), settings=FAST),
        FrameEscalator(models.role("frontier")),
    )
    run, _ = await _run_alone(models, step)
    assert run.status == "completed"
    (call,) = models.calls("frontier")
    assert call.frame is not None  # the frame the step judged, not a description of it


def test_a_step_with_an_escalator_checks_its_parts_and_its_environment_up_front() -> None:
    models = FakeModels()
    watch = WatchStep("fill", _water(), _camera(models))
    with pytest.raises(TypeError, match="is not an escalator"):
        StepWithEscalator(watch, object())  # type: ignore[arg-type]
    paired = StepWithEscalator(watch, _escalator(models, "The person knows."))

    class Mute:
        frames = None

        async def report_progress(self, step_id, state) -> None:
            return None

    with pytest.raises(TypeError, match="Mute is not one"):
        run_virtual(paired.run(Mute()))  # type: ignore[arg-type]


async def test_each_steps_escalation_may_carry_its_own_prompt(tmp_path: Path) -> None:
    (tmp_path / "fill.escalation.yaml").write_text("guidance: The person knows.\n", encoding="utf-8")
    (tmp_path / "seal.escalation.yaml").write_text(
        "guidance: The person can press the lid.\n"
        "instructions: Ask the person to press the lid and say whether it clicks.\n",
        encoding="utf-8",
    )
    reason = FakeModels().role("reason")
    fill, seal = (
        ReasoningEscalator(load_words(tmp_path / f"{name}.escalation.yaml", EscalationWords), reason)
        for name in ("fill", "seal")
    )
    assert (fill.guidance, fill.prompt) == ("The person knows.", PROMPT)
    assert seal.instructions == "Ask the person to press the lid and say whether it clicks."
    models = FakeModels({"reason": [{"action": "ask", "words": "Does it click?"}]})
    unknowns = [Judgment("unknown", "a lid", i, i) for i in range(4)]
    said = await decide_escalation(
        watch_step_snapshot("seal", Check("Sealed?", WATER), unknowns, Facts(), 0.0),
        models.role("reason"),
        guidance=seal.guidance,
        instructions=seal.instructions,
    )
    assert said is not None and models.calls("reason")[0].prompt.endswith(seal.instructions)
    (tmp_path / "typo.escalation.yaml").write_text("guidance: g\ninstruction: typo\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"(?s)typo.escalation.yaml.*instruction"):
        load_words(tmp_path / "typo.escalation.yaml", EscalationWords)  # a misspelled key fails when it is loaded


# ---------------------------------------------------------------------------------- the words


def test_words_load_from_yaml_into_each_step_types_schema(tmp_path: Path) -> None:
    (tmp_path / "fill.yaml").write_text(
        """
does: fill the kettle
start_line: Fill the kettle to {level}.
done_line: Filled to {level}.
question: Is there water?
criteria: {instructions: "Water at {level}?", positive: "yes", negative: "no", unknown: unclear}
""",
        encoding="utf-8",
    )
    (tmp_path / "procedure.yaml").write_text(
        """
steep:
  step:
    does: wait
    start_line: Steeping for {minutes:g} minutes.
    done_line: Steeped for {minutes:g} minutes.
""",
        encoding="utf-8",
    )
    models = FakeModels()

    class Level(BaseModel):
        level: str

    class Steep(WaitInputs):
        @property
        def minutes(self) -> float:
            return self.duration_s / 60

    words = load_words(tmp_path / "fill.yaml", WatchWords)
    fill = WatchStep("fill", words, _camera(models), inputs_schema=Level)
    line = Level(level="the line")
    assert (fill.start_line(line), fill.check_for(line).criteria.instructions) == (
        "Fill the kettle to the line.",
        "Water at the line?",
    )
    steep = WaitStep("steep", load_words(tmp_path / "procedure.yaml", StepWords, key="steep.step"), inputs_schema=Steep)
    two_minutes = Steep(duration_s=120)
    assert (steep.start_line(two_minutes), steep.done_line(two_minutes, None)) == (  # type: ignore[arg-type]
        "Steeping for 2 minutes.",
        "Steeped for 2 minutes.",
    )
    assert [(step.id, step.words.does) for step in (fill, steep)] == [("fill", "fill the kettle"), ("steep", "wait")]
    with pytest.raises(ValueError, match=r"its criteria uses \{level\}"):
        WatchStep("fill", words, _camera(models))  # a placeholder its inputs do not have fails when the step is built
    with pytest.raises(TypeError, match="runs with Level; it got NoneType"):
        run_virtual(fill.run(None))  # type: ignore[arg-type]  # inputs are checked before anything runs
    with pytest.raises(ValueError, match="question"):
        load_words(tmp_path / "procedure.yaml", WatchWords, key="steep.step")  # a watch step needs its question
    with pytest.raises(KeyError, match="'boil.step'"):
        load_words(tmp_path / "procedure.yaml", StepWords, key="boil.step")


def test_each_step_uses_its_own_models_and_a_misspelled_key_fails() -> None:
    class Label(BaseModel):
        name: str | None = None

    step = ReadStep(
        "identify", ReadWords(start_line="Show the label.", question="Q."), FakeModels().role("ocr"), reply_schema=Label
    )
    assert (step.perceive.name, step.question) == ("ocr", "Q.")
    criteria = {"instructions": "?", "positive": "y", "negative": "n", "unknown": "u"}
    with pytest.raises(ValueError, match="postive"):
        WatchWords.model_validate({"question": "Water?", "criteria": {**criteria, "postive": "y"}})
    with pytest.raises(ValueError, match="escalaton"):
        WatchWords.model_validate({"question": "Water?", "criteria": criteria, "escalaton": "x"})


def test_an_apps_own_step_type_defines_its_own_words(tmp_path: Path) -> None:
    (tmp_path / "scan.yaml").write_text("start_line: Scan it.\npattern: '[0-9]{12}'\n", encoding="utf-8")

    class ScanWords(StepWords):
        pattern: str

    words = load_words(tmp_path / "scan.yaml", ScanWords)
    assert (words.pattern, words.start_line) == ("[0-9]{12}", "Scan it.")


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
    step = StepWithEscalator(
        WatchStep("heat", _water(), _camera(models), settings=WatchSettings(interval_s=2.0)),
        _escalator(models, "The person knows the unit.", every_s=20.0, max_lines=2, reads_conversation=True),
    )
    case = evaluation.Case(
        name="answered",
        scene=[0, 120],
        person={"answers": ["Fahrenheit"]},
        conversation="user: is it hot yet?",
        run_summary="not read",
        expect={"asks": ["unit"], "completes": {"after": 20, "before": 40}},
    )
    result = run_virtual(evaluation.run_case(step, case, _recording(tmp_path, 120)))
    assert result.ok, result.problems
    (line,) = result.trace.lines
    assert (line.by, line.kind, line.answer) == ("escalator", "ask", "Fahrenheit")
    prompt = models.calls("reason")[0].prompt
    assert "is it hot yet?" in prompt and "not read" not in prompt  # the conversation, not the run's summary


def test_a_case_records_the_steps_done_line_apart_from_what_an_escalator_says(tmp_path: Path) -> None:
    step = WaitStep("steep", StepWords(done_line="Steeped for 2 minutes."), duration_s=5.0)
    case = evaluation.Case(name="steeps", scene=[0, 30], expect={"completes": True})
    result = run_virtual(evaluation.run_case(step, case, _recording(tmp_path, 30)))
    assert result.ok, result.problems
    assert [(line.kind, line.text) for line in result.trace.lines] == [("done", "Steeped for 2 minutes.")]


# ---------------------------------------------------------------------------------- an app's own step type


class AskWords(StepWords):
    """An ask step's words: what every step says, and its question."""

    question: str


class AskStep:
    """An app's own step type, written with the public API only: ask the person for a value; no camera, no model."""

    scoring = wait_scoring.__name__  # scored on whether it completes, like a wait step

    inputs_schema = None

    def __init__(self, step_id: str, words: AskWords) -> None:
        self.id = step_id
        self.words = words
        self.question = words.question
        check_placeholders(self.id, "done_line", words.done_line, None, also=("answer",))

    def start_line(self, inputs: None = None) -> str | None:
        return self.words.start_line

    def done_line(self, inputs: None, answer: str) -> str | None:
        return fill_placeholders(self.words.done_line, answer=answer)

    async def run(self, environment: StepEnvironment, inputs: None = None) -> str:
        while (answer := await environment.ask(self.question)) is None:
            await environment.report_progress(self.id, "no answer yet")
        return answer


def test_an_apps_own_step_type_runs_and_is_evaluated_like_the_kits(tmp_path: Path) -> None:
    (tmp_path / "batch.yaml").write_text(
        """
start_line: Next, the batch number.
done_line: Batch {answer} noted.
question: What is the batch number on the label?
""",
        encoding="utf-8",
    )
    words = load_words(tmp_path / "batch.yaml", AskWords)
    step = AskStep("batch", words)
    assert isinstance(step, Step)

    async def answer(runner, p) -> None:
        runner.run(p).answer("A-17")

    models = FakeModels()
    run, results = asyncio.run(
        _run_alone(models, step, until=lambda r: r is not None and r.asking is not None, then=answer)
    )
    assert run.status == "completed" and results == ["A-17"]

    case = evaluation.Case(name="answered", scene=[0, 10], person={"answers": ["B-2"]}, expect={"completes": True})
    result = run_virtual(evaluation.run_case(AskStep("batch", words), case, _recording(tmp_path, 10)))
    assert result.ok, result.problems
    assert [(line.by, line.kind, line.text) for line in result.trace.lines] == [
        ("step", "ask", "What is the batch number on the label?"),
        ("step", "done", "Batch B-2 noted."),
    ]
    # the step's own question is not an escalator's: it neither satisfies asks nor breaks quiet
    expect = escalation_scoring.Expect
    assert escalation_scoring.score(result.trace, expect(quiet=True)) == []
    assert escalation_scoring.score(result.trace, expect(asks=["batch"]))[0].startswith("asked nothing")


async def test_the_runs_lines_queue_in_order_and_none_replaces_another() -> None:
    """A step's done_line and the next step's start_line follow each other closely; both must be spoken."""
    said: list[tuple[str, dict]] = []

    class Person:
        async def say(self, text: str, **options) -> None:
            said.append((text, options))

    run = Run(Person(), 1, name="tea")  # type: ignore[arg-type]
    await run.say("Water is visible in the vessel.")
    await run.say("Heat the water to 80 degrees Celsius.")
    assert [options.get("source") for _, options in said] == [None, None]  # no source: queued, never replaced


# ---------------------------------------------------------------------------------- the transform step


class Brewing(BaseModel):
    tea_type: str | None = None
    temperature_c: float | None = None
    minutes: float | None = None


SETTINGS = TransformWords(
    done_line="{output.temperature_c:g} degrees for {output.minutes:g} minutes.",
    instructions="Fill in conventional brewing settings for this tea. Ask the person only when the kind of tea is "
    "unknown.",
)


def _settings_step(models: FakeModels, **settings) -> TransformStep:
    return TransformStep(
        "settings",
        SETTINGS,
        models.role("reason"),
        input_schema=Brewing,
        output_schema=Brewing,
        keep_input_values=True,
        required_fields=("temperature_c", "minutes"),
        settings=TransformSettings(**settings),
    )


def _transform(tmp_path: Path, step: TransformStep, inputs: dict, answers: list[str] = ()) -> evaluation.CaseResult:
    case = evaluation.Case(name="t", scene=[0, 120], person={"answers": list(answers)})
    return run_virtual(evaluation.run_case(step, case, _recording(tmp_path, 120), inputs=inputs))


def test_a_transform_fills_in_what_is_missing_and_keeps_what_is_known(tmp_path: Path) -> None:
    models = FakeModels({"reason": [{"output": {"tea_type": "black", "temperature_c": 80, "minutes": 3}}]})
    result = _transform(tmp_path, _settings_step(models), {"tea_type": "green", "minutes": 2})
    assert result.trace.result == Brewing(tea_type="green", temperature_c=80, minutes=2)  # the model's guesses lost
    assert [(line.kind, line.text) for line in result.trace.lines] == [("done", "80 degrees for 2 minutes.")]
    prompt = models.calls("reason")[0].prompt
    assert "already known and are kept as they are: tea_type, minutes" in prompt and '"tea_type":"green"' in prompt
    (state,) = [state for _, state in result.trace.states if state.status == "completed"]
    assert state.filled == ("temperature_c",)


def test_a_transform_may_ask_the_person_and_reads_their_answer(tmp_path: Path) -> None:
    question = {"question": "What kind of tea is it?"}
    models = FakeModels({"reason": [question, {"output": {"tea_type": "oolong", "temperature_c": 90, "minutes": 4}}]})
    result = _transform(tmp_path, _settings_step(models), {}, answers=["Oolong"])
    assert result.trace.result.tea_type == "oolong"
    ask = result.trace.lines[0]
    assert (ask.by, ask.kind, ask.text, ask.answer) == ("step", "ask", "What kind of tea is it?", "Oolong")
    assert 'They answered: "Oolong"' in models.calls("reason")[1].prompt


def test_a_transform_answers_itself_when_no_answer_comes(tmp_path: Path) -> None:
    question = {"question": "What kind of tea is it?"}
    models = FakeModels({"reason": [question, {"output": {"temperature_c": 85, "minutes": 3}}]})
    result = _transform(tmp_path, _settings_step(models, max_questions=1), {})
    assert result.trace.result == Brewing(temperature_c=85, minutes=3)
    second = models.calls("reason")[1].prompt
    assert "No answer came." in second and "set question" not in second  # its one question is spent


def test_a_transform_asks_again_for_a_missing_required_field_then_fails(tmp_path: Path) -> None:
    models = FakeModels({"reason": [{"output": {"temperature_c": 80}}] * 3})
    result = _transform(tmp_path, _settings_step(models, max_questions=0), {})
    assert result.trace.error is not None and "no complete Brewing after 3 replies" in result.trace.error
    assert "left these empty: minutes" in models.calls("reason")[1].prompt


def test_a_transform_names_only_fields_it_has() -> None:
    with pytest.raises(ValueError, match="required_fields"):
        TransformStep(
            "s", SETTINGS, FakeModels().role("r"), input_schema=Brewing, output_schema=Brewing, required_fields=("x",)
        )
    wrong = TransformWords(done_line="{result.minutes}", instructions=SETTINGS.instructions)
    with pytest.raises(ValueError, match=r"\{result\}"):
        TransformStep("s", wrong, FakeModels().role("r"), input_schema=Brewing, output_schema=Brewing)


async def test_a_transform_cannot_be_skipped_even_while_it_asks() -> None:
    question = {"question": "What kind of tea is it?"}
    models = FakeModels({"reason": [question, {"output": {"tea_type": "green", "temperature_c": 80, "minutes": 2}}]})
    said: list[str] = []

    async def try_to_skip_then_answer(runner, p) -> None:
        said.append(await runner.skip(p))
        runner.run(p).answer("Green")

    run, results = await _run_alone(
        models,
        _settings_step(models),
        Brewing(),
        until=lambda r: r is not None and r.asking is not None,
        then=try_to_skip_then_answer,
    )
    assert said == [MESSAGES["cannot_skip"]]
    assert run.steps[0].status == "done" and results == [Brewing(tea_type="green", temperature_c=80, minutes=2)]
