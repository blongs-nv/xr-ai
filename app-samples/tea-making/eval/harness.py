# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the tea guide's test packets: one per step, against the app's code.

A packet (packets/<step>.yaml) says what can happen while one step of the
guide is active, and what the guide should do about it:

- frames: groups of frames from a labeled recording, each with what the
  step's check should conclude on it (done, not done, can't tell). The
  check is judged on every frame of a group, exactly as it runs live, and
  the live rule (2 of the last 4) is replayed over the group: a not-done
  group the rule completes on is a false completion.
- utterances: things the user may say during the step, each with where
  the turn should go (route), which control it asks for (control: start,
  stop, skip, restart, or none), and, for answers, what the answer must or
  must not say. common.yaml adds utterances asked during every step.

The run is put in the packet's state (SopRunner.assume) and each utterance
goes through the app's real Router decision (Router.choose), the guide's
control decision (SopRunner.classify; a non-control passes the turn on, as
live) and the Conversation's answer; nothing is performed and nothing is
spoken. Every model call uses the app's roles (yaml/models.json).

    uv --config-file ../../uv.toml run python eval/harness.py \\
        --recording eval/recordings/green-tea-04.yaml \\
        --frames-index /path/to/frames/index.csv --out runs/packets/

Add a case: one entry in a packet's `frames` or `utterances`. Add a step:
one packet file. See eval/README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

SAMPLE = Path(__file__).resolve().parents[1]
EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(SAMPLE))

from procedure import Steps, checks  # noqa: E402
from tea import TeaLabel  # noqa: E402
from xr_ai_app import (  # noqa: E402
    Frame,
    Participant,
    Role,
    StructuredOutputError,
    TextSink,
    Turn,
)
from xr_ai_kit import Conversation  # noqa: E402
from xr_ai_kit.sop import (  # noqa: E402
    Check,
    Facts,
    Judgment,
    Run,
    StepRecord,
    judge,
    question_for,
)

CONTROLS = ("start", "stop", "skip", "restart", "none")


# ------------------------------------------------------------------ packets


@dataclass
class Packet:
    name: str
    step: str | None
    state: dict[str, Any]
    frames: list[dict[str, Any]]
    utterances: list[dict[str, Any]]
    words: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, common: dict[str, Any]) -> Packet:
        raw = yaml.safe_load(path.read_text("utf-8")) or {}
        unknown = set(raw) - {"step", "state", "frames", "utterances"}
        if unknown:
            raise ValueError(f"{path.name}: unknown keys {sorted(unknown)}")
        # common.yaml's utterances are asked during every step.
        shared = common.get("utterances", []) if raw.get("step") else []
        return cls(
            name=path.stem,
            step=raw.get("step"),
            state=raw.get("state") or {},
            frames=raw.get("frames") or [],
            utterances=list(raw.get("utterances") or []) + list(shared),
            words=common.get("step_words", {}),
        )

    def fill(self, text: str) -> str:
        """{current}, {next}: the packet's steps, as words the user hears."""
        order = list(self.words)
        current = self.step
        nxt = None
        if current in order and order.index(current) + 1 < len(order):
            nxt = order[order.index(current) + 1]
        return text.format(current=current or "", next=nxt or "")

    def words_for(self, needle: str) -> list[str]:
        """A step id stands for any of its words; anything else for itself."""
        return self.words.get(needle, [needle])


# ------------------------------------------------------------------ frames


def load_index(path: Path) -> list[tuple[float, Path]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    out = []
    for row in rows:
        t = row.get("source_time_s") or row.get("time_s")
        image = row.get("frame_image") or row.get("path")
        if t is None or image is None:
            raise ValueError(f"{path}: needs source_time_s and frame_image")
        out.append((float(t), (path.parent / image).resolve()))
    return sorted(out)


def on_grid(
    index: list[tuple[float, Path]], window: list[float], every_s: float
) -> list[tuple[float, Path]]:
    start, end = float(window[0]), float(window[1])
    picked: list[tuple[float, Path]] = []
    t = start
    while t < end:
        nearest = min(index, key=lambda item: abs(item[0] - t))
        if start <= nearest[0] < end and (not picked or picked[-1] != nearest):
            picked.append(nearest)
        t += every_s
    return picked


def frame_of(t: float, path: Path) -> Frame:
    data = path.read_bytes()
    width, height = Image.open(io.BytesIO(data)).size
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return Frame(data, int(t * 1_000_000), width, height, mime)


def step_check(step: str, models: Models, recording: dict[str, Any]) -> Check:
    """The step's Check, built exactly as the procedure builds it."""
    target_c = float(
        recording.get("target_c", models.settings.default_temperature_c)
    )
    built = checks(
        models.steps, temperature_c=target_c, unit=models.settings.display_unit
    )
    return built[step]


def group_facts(check: Check, group: dict[str, Any]) -> Facts | None:
    """The facts a group starts with: what the person already said (typed)."""
    given = group.get("facts")
    if not given:
        return None
    if check.needs is None:
        raise ValueError(
            f"group {group['name']}: facts for a check without needs"
        )
    return Facts(check.needs(**given), ("(Given in the test packet.)",))


async def check(
    step: str,
    frame: Frame,
    models: Models,
    recording: dict[str, Any],
    facts: Facts | None,
) -> tuple[Judgment | None, dict[str, Any]]:
    """The step's live check on one frame: positive, negative or unknown."""
    steps, settings = models.steps, models.settings
    if step == "identify":
        try:
            tea = await models.perceive.ask(
                frame,
                steps.identify.question,
                response_format=TeaLabel,
                timeout_s=settings.timeout_s,
            )
        except StructuredOutputError:
            return None, {"label": "unknown", "reason": "unreadable"}
        ready = tea.tea_type is not None  # the procedure's ready()
        return None, {
            "label": "positive" if ready else "negative",
            "observed": tea.model_dump_json(exclude_none=True),
        }
    j = await judge(
        frame,
        step_check(step, models, recording),
        observe=models.perceive,
        decide=models.decide,
        facts=facts,
        timeout_s=settings.timeout_s,
    )
    return j, {"label": j.label, "observed": j.description, "reason": j.reason}


async def would_ask(
    packet: Packet,
    done: list[tuple[float, Frame, Judgment | None]],
    models: Models,
    recording: dict[str, Any],
    facts: Facts | None,
) -> tuple[float | None, str | None]:
    """The first question the side escalation would ask over this group.

    Replays gather's pacing on the recording's clock: every review_every_s
    seconds, the reasoning model gets the latest frame, what is going on
    (the step's state and the fast check's latest report, as live) and the
    facts, and decides whether to ask. Nobody is asked anything.
    (None, None) when it never asks.
    """
    if not done:
        return None, None
    check = step_check(packet.step, models, recording)
    pace = models.settings.review_every_s
    started = done[0][0]
    next_review = started + pace
    judged: list[Judgment] = []
    for t, frame, j in done:
        if j is not None:
            judged.append(j)
        if t < next_review:
            continue
        next_review = t + pace
        question = await question_for(
            check,
            models.review,
            frame,
            facts=facts,
            situation=situation_at(
                packet, models.settings.window, judged, t - started
            ),
            timeout_s=max(models.settings.timeout_s, 60.0),
        )
        if question is not None:
            return t, question
    return None, None


def situation_at(
    packet: Packet, window: int, judged: list[Judgment], elapsed_s: float
) -> str:
    """What is going on, as the live app tells the side escalation: the
    guide's steps and the running check's latest report."""
    steps = ", ".join(
        f"{k}: {v}" for k, v in (packet.state.get("steps") or {}).items()
    )
    recent = judged[-window:]
    positives = sum(1 for j in recent if j.label == "positive")
    latest = (
        f" Latest: {judged[-1].label}: {judged[-1].description[:300]}"
        if judged
        else " No frame judged yet."
    )
    return (
        f"Steps: {steps}\n"
        f"Current step: {packet.step}, for {elapsed_s:.0f} s. Its camera "
        f"check: {positives} positive in the last {len(recent)} judged "
        f"frames.{latest}"
    )


def fires(labels: list[str], required: int, window: int) -> int | None:
    """The index at which N of the last M decidable labels are positive."""
    recent: deque[str] = deque(maxlen=window)
    for i, label in enumerate(labels):
        if label != "unknown":
            recent.append(label)
        if sum(1 for x in recent if x == "positive") >= required:
            return i
    return None


def group_verdict(expect: str, fired: int | None, labels: list[str]) -> bool:
    """done: the live rule completes; not_done: it must not; either: both OK."""
    if expect == "either":
        return True
    if expect == "done":
        return fired is not None
    return fired is None


def ask_verdict(group: dict[str, Any], question: str | None) -> list[str]:
    """What is wrong with the asking in this group (asks, asks_about)."""
    problems = []
    if "asks" in group and bool(question) != bool(group["asks"]):
        problems.append(
            f"asked: {question!r}" if question else "did not ask when stuck"
        )
    about = group.get("asks_about")
    if (
        question
        and about
        and not any(w.lower() in question.lower() for w in about)
    ):
        problems.append(f"question is not about any of {about}")
    return problems


async def run_frames(
    packet: Packet,
    index: list[tuple[float, Path]],
    models: Models,
    recording: dict[str, Any],
    gate: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    if not packet.step or not packet.frames:
        return []
    settings = models.settings

    async def one(
        t: float, path: Path, facts: Facts | None
    ) -> tuple[Judgment | None, dict[str, Any]]:
        async with gate:
            began = time.monotonic()
            try:
                j, out = await check(
                    packet.step, frame_of(t, path), models, recording, facts
                )
            except Exception as exc:  # noqa: BLE001 - scored as unknown
                j, out = None, {"label": "unknown", "reason": f"error: {exc!r}"}
            out.update(t=round(t, 2), s=round(time.monotonic() - began, 2))
            return j, out

    groups = []
    for group in packet.frames:
        grid = on_grid(index, group["at"], settings.interval_s)
        facts = None
        if packet.step != "identify":
            facts = group_facts(
                step_check(packet.step, models, recording), group
            )
        done = await asyncio.gather(*(one(t, p, facts) for t, p in grid))
        frames = [out for _, out in done]
        labels = [f["label"] for f in frames]
        fired = fires(labels, settings.required, settings.window)
        asked_at, question = None, None
        if "asks" in group or "asks_about" in group:
            seen = [
                (t, frame_of(t, p), j)
                for (t, p), (j, _) in zip(grid, done, strict=True)
            ]
            asked_at, question = await would_ask(
                packet, seen, models, recording, facts
            )
        problems = ask_verdict(group, question)
        groups.append(
            {
                "name": group["name"],
                "expect": group["expect"],
                "why": group.get("why", ""),
                "facts": group.get("facts"),
                "frames": frames,
                "counts": {k: labels.count(k) for k in set(labels)},
                "fired_at": None if fired is None else frames[fired]["t"],
                "asked_at": asked_at,
                "question": question,
                "problems": problems,
                "ok": group_verdict(group["expect"], fired, labels)
                and not problems,
            }
        )
    return groups


# ------------------------------------------------------------------ dialogue


@dataclass
class Models:
    """The app's pieces the harness drives (from app.py), or scripted ones."""

    perceive: Role
    decide: Role
    answer: Role
    review: Role  # the side escalation's reasoning model
    routing: Any  # the app's Router
    tea: Any  # the app's SopRunner
    steps: Steps
    settings: Any
    instructions: str


def run_in_state(packet: Packet, p: Participant) -> Run | None:
    """The run the packet describes (None: the guide was never started)."""
    state = packet.state
    if not state.get("steps") and state.get("status") is None:
        return None
    records = [
        StepRecord(
            step, status, state.get("detail", "") if status == "running" else ""
        )
        for step, status in (state.get("steps") or {}).items()
    ]
    return Run(
        p,
        1,
        status=state.get("status", "running"),
        steps=records,
        values=dict(state.get("noted") or {}),
        asking=state.get("asking"),
    )


async def run_utterance(
    packet: Packet, case: dict[str, Any], models: Models
) -> dict[str, Any]:
    p = Participant(f"eval-{uuid.uuid4().hex[:6]}")
    run = run_in_state(packet, p)
    if run is not None:
        await models.tea.assume(p, run)
    else:
        await models.tea.context.set(p, models.tea.summary(p))
    text = packet.fill(case["say"])
    turn = Turn(text, 0, uuid.uuid4().hex)
    choice = await models.routing.choose(turn, p)
    route, control = choice.name, "none"
    if choice.name == models.tea.control.name:
        control = await models.tea.classify(text, p)
        if control == "none":  # the guide passes the turn on, as live
            reason = f"{choice.name}: not a control of the guide"
            route = (
                await models.routing.choose(
                    turn, p, exclude=[choice.name], reasons=[reason]
                )
            ).name
    answer = ""
    if route == "conversation":
        sink = TextSink()
        conversation = Conversation(
            role=models.answer,
            output=sink,
            instructions=models.instructions,
            context=[models.tea.context],
        )
        await conversation.answer(turn, p)
        answer = " ".join(sink.said.get(p.id, []))
    return {
        "say": text,
        "route": route,
        "control": control,
        "answer": answer,
        **score_utterance(
            packet,
            case.get("expect") or {},
            route=route,
            control=control,
            answer=answer,
            guide=models.tea.control.name,
        ),
    }


def score_utterance(
    packet: Packet,
    expect: dict[str, Any],
    *,
    route: str,
    control: str,
    answer: str,
    guide: str,
) -> dict[str, Any]:
    """What went wrong for one utterance (an empty list: it passed)."""
    problems = []
    wanted = expect.get("control", "none")
    controls = [wanted] if isinstance(wanted, str) else list(wanted)
    unintended = control != "none" and control not in controls
    missed = "none" not in controls and control not in controls
    if unintended:
        problems.append(f"unintended {control}")
    elif missed:
        problems.append(f"missed {'/'.join(controls)} (got {control})")
    routes = expect.get("route")
    if routes is not None:
        allowed = [routes] if isinstance(routes, str) else list(routes)
        if control != "none":
            allowed = [*allowed, guide]
        if route not in allowed and not unintended and not missed:
            problems.append(f"route {route}, expected {'/'.join(allowed)}")
    checks = expect.get("answer") or {}
    if checks and route == "conversation":
        said = answer.lower()
        any_of = [
            w
            for n in checks.get("mentions", [])
            for w in packet.words_for(packet.fill(n))
        ]
        if any_of and not any(w.lower() in said for w in any_of):
            problems.append(f"answer mentions none of {any_of}")
        for n in checks.get("not", []):
            for w in packet.words_for(packet.fill(n)):
                if w.lower() in said:
                    problems.append(f"answer says {w!r}")
    return {
        "ok": not problems,
        "problems": problems,
        "state_change": control != "none",
        "unintended": unintended,
        "missed": missed,
    }


# ------------------------------------------------------------------ report


def report(results: list[dict[str, Any]], models_label: str) -> str:
    lines = ["# Tea guide test packets", "", f"Models: {models_label}.", ""]
    groups = [g for r in results for g in r["frames"]]
    utt = [u for r in results for u in r["utterances"]]
    false_completions = [
        g
        for g in groups
        if g["expect"] == "not_done" and g["fired_at"] is not None
    ]
    lines += [
        "| | Total | Passed | Failed |",
        "|---|---|---|---|",
        f"| Frame groups | {len(groups)} | {sum(g['ok'] for g in groups)}"
        f" | {sum(not g['ok'] for g in groups)} |",
        f"| Utterances | {len(utt)} | {sum(u['ok'] for u in utt)}"
        f" | {sum(not u['ok'] for u in utt)} |",
        "",
        f"False completions (a step completed on frames that do not show it"
        f" done): {len(false_completions)}. Unintended state changes (a"
        f" control from something that is not one):"
        f" {sum(u['unintended'] for u in utt)}. Missed controls:"
        f" {sum(u['missed'] for u in utt)}.",
        "",
    ]
    for r in results:
        lines += [f"## {r['packet']}", ""]
        if r["frames"]:
            lines += [
                "| Frames | Expect | Judged (pos/neg/unk) | Live rule | OK |",
                "|---|---|---|---|---|",
            ]
            for g in r["frames"]:
                c = g["counts"]
                judged = (
                    f"{c.get('positive', 0)}/{c.get('negative', 0)}"
                    f"/{c.get('unknown', 0)}"
                )
                rule = (
                    "-"
                    if g["fired_at"] is None
                    else f"done at {g['fired_at']:g} s"
                )
                mark = "yes" if g["ok"] else "**no**"
                row = [g["name"], g["expect"], judged, rule, mark]
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        if r["utterances"]:
            lines += [
                "| Said | Route | Control | OK | Problems |",
                "|---|---|---|---|---|",
            ]
            for u in r["utterances"]:
                mark = "yes" if u["ok"] else "**no**"
                lines.append(
                    f"| {u['say']} | {u['route']} | {u['control']} | {mark}"
                    f" | {'; '.join(u['problems'])} |"
                )
            lines.append("")
    return "\n".join(lines)


async def run_all(
    packets: list[Packet],
    models: Models,
    index: list[tuple[float, Path]] | None,
    recording: dict[str, Any],
    concurrency: int,
) -> list[dict[str, Any]]:
    gate = asyncio.Semaphore(concurrency)

    async def utter(packet: Packet, case: dict[str, Any]) -> dict[str, Any]:
        async with gate:
            try:
                return await run_utterance(packet, case, models)
            except Exception as exc:  # noqa: BLE001 - reported as a failure
                return {
                    "say": case.get("say", ""),
                    "route": "error",
                    "control": "none",
                    "answer": "",
                    "ok": False,
                    "problems": [f"error: {exc!r}"],
                    "state_change": False,
                    "unintended": False,
                    "missed": False,
                }

    results = []
    for packet in packets:
        frames = (
            await run_frames(packet, index, models, recording, gate)
            if index
            else []
        )
        utterances = list(
            await asyncio.gather(*(utter(packet, c) for c in packet.utterances))
        )
        results.append(
            {"packet": packet.name, "frames": frames, "utterances": utterances}
        )
    return results


def load_models(fakes: Path | None) -> tuple[Models, str]:
    import app as sample  # the roles, router, runner and wording the app uses

    steps = Steps.load(SAMPLE / "steps.yaml")
    perceive, decide, answer = sample.perceive, sample.decide, sample.answer
    review = sample.review
    label = (
        "yaml/models.json (perceive, decide, answer_with_reasoning, "
        "perceive_with_reasoning)"
    )
    routing, tea = sample.routing, sample.tea
    if fakes is not None:
        from xr_ai_app.devkit import fakes_from_file
        from xr_ai_kit import Router
        from xr_ai_kit.sop import SopRunner

        scripted = fakes_from_file(fakes)
        perceive, decide = scripted.role("perceive"), scripted.role("decide")
        answer = scripted.role("answer_with_reasoning")
        review = scripted.role("perceive_with_reasoning")

        async def never(run: Any) -> None:  # classified, never run
            raise AssertionError("the harness runs no procedure")

        tea = SopRunner("tea", never, title=sample.tea.title, decide=decide)
        routing = Router(
            decide=decide,
            destinations=[tea.control, sample.view, sample.chat],
            default=sample.chat,
        )
        label = f"scripted ({fakes.name})"
    models = Models(
        perceive=perceive,
        decide=decide,
        answer=answer,
        review=review,
        routing=routing,
        tea=tea,
        steps=steps,
        settings=sample.SETTINGS.tea,
        instructions=sample.prompt("conversation"),
    )
    return models, label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--packets", type=Path, default=EVAL / "packets")
    parser.add_argument(
        "--recording",
        type=Path,
        default=EVAL / "recordings" / "green-tea-04.yaml",
    )
    parser.add_argument(
        "--frames-index",
        type=Path,
        help="frames of the recording (else utterances only)",
    )
    parser.add_argument("--only", nargs="*", help="packet names to run")
    parser.add_argument("--out", type=Path, default=Path("runs/packets"))
    parser.add_argument(
        "--fakes", type=Path, help="scripted replies (plumbing only)"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args(argv)

    common_path = args.packets / "common.yaml"
    common = (
        yaml.safe_load(common_path.read_text("utf-8"))
        if common_path.exists()
        else {}
    )
    packets = [
        Packet.load(path, common)
        for path in sorted(args.packets.glob("*.yaml"))
        if path.name != "common.yaml"
        and (
            not args.only
            or path.stem in args.only
            or path.stem.split(".")[0] in args.only
        )
    ]
    recording = (
        yaml.safe_load(args.recording.read_text("utf-8"))
        if args.recording.exists()
        else {}
    )
    index = load_index(args.frames_index) if args.frames_index else None
    models, label = load_models(args.fakes)
    results = asyncio.run(
        run_all(packets, models, index, recording, args.concurrency)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    text = report(results, label)
    (args.out / "report.md").write_text(text)
    print(text)
    failed = sum(not g["ok"] for r in results for g in r["frames"]) + sum(
        not u["ok"] for r in results for u in r["utterances"]
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
