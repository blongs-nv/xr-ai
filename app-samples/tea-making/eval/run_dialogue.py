# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the user may say during each step, and what the guide should do.

The step cases (run_steps.py) evaluate each step on its own; these cases
evaluate the guide around the steps: a packet (dialogue/<step>.yaml) puts
the run in a state (some steps done, one running, maybe a question
pending) and lists utterances, each with where the turn should go
(route), which control it asks for (control: start, stop, skip, restart,
answer, or none), and, for answers, what the answer must or must not say.
common.yaml adds utterances asked during every step.

The run is put in the packet's state (SopRunner.assume) and each utterance
goes through the app's real Router decision (Router.choose), the guide's
control decision (SopRunner.classify; a non-control passes the turn on, as
live) and the Conversation's answer; nothing is performed and nothing is
spoken. Every model call uses the app's roles (yaml/models.json).

    uv --config-file ../../uv.toml run python eval/run_dialogue.py \\
        --out runs/dialogue/

Add a case: one entry in a packet's `utterances`. See eval/README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SAMPLE = Path(__file__).resolve().parents[1]
EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(SAMPLE))

from xr_ai_app import Participant, Role, TextSink, Turn  # noqa: E402
from xr_ai_kit import Conversation  # noqa: E402
from xr_ai_kit.sop import Run, StepRecord  # noqa: E402

# ------------------------------------------------------------------ packets


@dataclass
class Packet:
    name: str
    step: str | None
    state: dict[str, Any]
    utterances: list[dict[str, Any]]
    words: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, common: dict[str, Any]) -> Packet:
        raw = yaml.safe_load(path.read_text("utf-8")) or {}
        unknown = set(raw) - {"step", "state", "utterances"}
        if unknown:
            raise ValueError(f"{path.name}: unknown keys {sorted(unknown)}")
        # common.yaml's utterances are asked during every step.
        shared = common.get("utterances", []) if raw.get("step") else []
        return cls(
            name=path.stem,
            step=raw.get("step"),
            state=raw.get("state") or {},
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


# ------------------------------------------------------------------ dialogue


@dataclass
class Models:
    """The app's pieces the cases drive (from app.py), or scripted ones."""

    decide: Role
    answer: Role
    routing: Any  # the app's Router
    tea: Any  # the app's SopRunner
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
    utt = [u for r in results for u in r["utterances"]]
    lines = [
        "# Tea guide dialogue",
        "",
        f"Models: {models_label}.",
        "",
        f"**{sum(u['ok'] for u in utt)} of {len(utt)} utterances passed.**"
        f" Unintended state changes (a control from something that is not"
        f" one): {sum(u['unintended'] for u in utt)}. Missed controls:"
        f" {sum(u['missed'] for u in utt)}.",
        "",
    ]
    for r in results:
        lines += [
            f"## {r['packet']}",
            "",
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
        utterances = list(
            await asyncio.gather(*(utter(packet, c) for c in packet.utterances))
        )
        results.append({"packet": packet.name, "utterances": utterances})
    return results


def load_models(fakes: Path | None) -> tuple[Models, str]:
    import app as sample  # the router, runner and wording the app uses

    decide, answer = sample.decide, sample.answer_with_reasoning
    label = "yaml/models.json (decide, answer_with_reasoning)"
    routing, tea = sample.routing, sample.tea
    if fakes is not None:
        from xr_ai_app.devkit import fakes_from_file
        from xr_ai_kit import Router
        from xr_ai_kit.sop import SopRunner

        scripted = fakes_from_file(fakes)
        decide = scripted.role("decide")
        answer = scripted.role("answer_with_reasoning")

        async def never(run: Any) -> None:  # classified, never run
            raise AssertionError("the dialogue cases run no procedure")

        tea = SopRunner("tea", never, title=sample.tea.title, decide=decide)
        routing = Router(
            decide=decide,
            destinations=[tea.control, sample.view, sample.chat],
            default=sample.chat,
        )
        label = f"scripted ({fakes.name})"
    models = Models(
        decide=decide,
        answer=answer,
        routing=routing,
        tea=tea,
        instructions=sample.prompt("conversation"),
    )
    return models, label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--packets", type=Path, default=EVAL / "dialogue")
    parser.add_argument("--only", nargs="*", help="packet names to run")
    parser.add_argument("--out", type=Path, default=Path("runs/dialogue"))
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
    models, label = load_models(args.fakes)
    results = asyncio.run(run_all(packets, models, args.concurrency))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )
    text = report(results, label)
    (args.out / "report.md").write_text(text)
    print(text)
    failed = sum(not u["ok"] for r in results for u in r["utterances"])
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
