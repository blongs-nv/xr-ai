# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score each check of the tea procedure on frames of a labeled recording.

For every step with labels, frames are taken on a grid (every `interval_s`
of app.yaml's tea settings, as the check looks live) inside the step's
`watch` window, and each frame is judged exactly as the running procedure
judges it: the same steps.yaml wording, the same roles (yaml/models.json),
the same `xr_ai_kit.sop.judge` (identify: the same label schema). Then:

- per frame, by label: positive frames judged positive (recall), negative
  frames judged positive (false positives), unknowns; trap frames (look
  like done, are not) judged positive;
- per step, the N-of-M rule replayed over the frames in time order: when
  the live check would have completed, against the labeled start of done
  (`early` is a false completion, `never` a missed one).

Run from the sample's folder, with the recording's frames and an index CSV
(columns `source_time_s` and `frame_image`, paths relative to the CSV):

    uv --config-file ../../uv.toml run python eval/step_eval.py \\
        --labels eval/labels/green-tea-04.yaml \\
        --frames-index /path/to/run/timeline.csv --out runs/eval/

Real models are yaml/models.json's (perceive and decide must be running);
`--fakes FILE` uses scripted replies instead (a smoke test).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

SAMPLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAMPLE))

from procedure import Steps, TeaSettings, target  # noqa: E402
from tea import TeaLabel, Thermometer, assessable  # noqa: E402
from xr_ai_app import Frame, Role, StructuredOutputError  # noqa: E402
from xr_ai_kit.sop import judge  # noqa: E402

Interval = tuple[float, float]


# ------------------------------------------------------------------ frames


def load_index(path: Path) -> list[tuple[float, Path]]:
    """(time, image path) for each frame, sorted by time."""
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
    index: list[tuple[float, Path]], window: Interval, every_s: float
) -> list[tuple[float, Path]]:
    """The frame nearest to each grid time in `window`, each at most once."""
    start, end = window
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


# ------------------------------------------------------------------ labels


def _intervals(spec: dict[str, Any], key: str) -> list[Interval]:
    return [(float(a), float(b)) for a, b in spec.get(key) or []]


def expected(t: float, spec: dict[str, Any], margin_s: float) -> str | None:
    """positive / negative / trap for a frame at `t`, or None (not scored)."""
    for a, b in _intervals(spec, "traps"):
        if a <= t < b:
            return "trap"
    for a, b in _intervals(spec, "exclude"):
        if a <= t < b:
            return None
    for kind in ("positive", "negative"):
        for a, b in _intervals(spec, kind):
            if a + margin_s <= t < b - margin_s:
                return kind
    return None


# ------------------------------------------------------------------ judging


async def judge_identify(
    frame: Frame, perceive: Role, steps: Steps, timeout_s: float
) -> dict[str, Any]:
    try:
        tea = await perceive.ask(
            frame,
            steps.identify.question,
            response_format=TeaLabel,
            timeout_s=timeout_s,
        )
    except StructuredOutputError:
        return {"label": "unknown", "reason": "unreadable"}
    ready = tea.tea_type is not None  # the procedure's ready()
    return {
        "label": "positive" if ready else "negative",
        "description": tea.model_dump_json(exclude_none=True),
        "tea_type": tea.tea_type,
        "tea_name": tea.tea_name,
    }


async def judge_check(
    step_id: str,
    frame: Frame,
    *,
    perceive: Role,
    decide: Role,
    steps: Steps,
    unit: str | None,
    target_c: float,
    timeout_s: float,
) -> dict[str, Any]:
    check = getattr(steps, step_id)
    extra: dict[str, Any] = {}
    if step_id == "heat_water":
        criteria = check.to_criteria(
            unit=unit or "not known", minimum=target(target_c, unit)
        )
        extra = {
            "observation_model": Thermometer,
            "can_assess": lambda reading: assessable(reading, unit),
        }
    else:
        criteria = check.to_criteria()
    j = await judge(
        frame,
        observe=perceive,
        decide=decide,
        question=check.question,
        criteria=criteria,
        timeout_s=timeout_s,
        **extra,
    )
    return {
        "label": j.label,
        "description": j.description,
        "reason": j.reason,
        "confidence": j.confidence,
    }


# ------------------------------------------------------------------ scoring


def score(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and rates for one step's scored frames."""
    counts: dict[str, dict[str, int]] = {}
    for r in records:
        if r["expected"] is None:
            continue
        row = counts.setdefault(r["expected"], {})
        row[r["label"]] = row.get(r["label"], 0) + 1
    pos = counts.get("positive", {})
    neg = counts.get("negative", {})
    trap = counts.get("trap", {})
    n_pos, n_neg = sum(pos.values()), sum(neg.values())
    tp, fp = pos.get("positive", 0), neg.get("positive", 0)
    return {
        "counts": counts,
        "positive_frames": n_pos,
        "negative_frames": n_neg,
        "recall": tp / n_pos if n_pos else None,
        "false_positive_rate": fp / n_neg if n_neg else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "unknown_rate": (pos.get("unknown", 0) + neg.get("unknown", 0))
        / (n_pos + n_neg)
        if n_pos + n_neg
        else None,
        "traps": sum(trap.values()),
        "traps_judged_positive": trap.get("positive", 0),
    }


def replay(
    records: list[dict[str, Any]],
    spec: dict[str, Any],
    *,
    required: int,
    window: int,
    margin_s: float,
) -> dict[str, Any]:
    """When the live N-of-M rule would have completed over these frames."""
    labels: deque[str] = deque(maxlen=window)
    fired_at = None
    for r in sorted(records, key=lambda r: r["t"]):
        if r["label"] != "unknown":
            labels.append(r["label"])
        if sum(1 for label in labels if label == "positive") >= required:
            fired_at = r["t"]
            break
    done_from = min((a for a, _ in _intervals(spec, "positive")), default=None)
    if fired_at is None:
        verdict = "never"
    elif done_from is not None and fired_at < done_from - margin_s:
        verdict = "early"
    else:
        verdict = "ok"
    return {
        "fired_at_s": fired_at,
        "done_from_s": done_from,
        "verdict": verdict,
        "delay_s": None
        if fired_at is None or done_from is None
        else round(fired_at - done_from, 1),
    }


# ------------------------------------------------------------------ run


async def evaluate(
    labels: dict[str, Any],
    index: list[tuple[float, Path]],
    *,
    perceive: Role,
    decide: Role,
    steps: Steps,
    settings: TeaSettings,
    concurrency: int,
) -> dict[str, Any]:
    margin = float(labels.get("margin_s", 1.0))
    unit = labels.get("display_unit") or settings.display_unit
    target_c = float(labels.get("target_c", settings.default_temperature_c))
    gate = asyncio.Semaphore(concurrency)
    results: dict[str, Any] = {}
    for step_id, spec in labels["steps"].items():
        if spec.get("no_data"):
            results[step_id] = {"no_data": True, "note": spec.get("note", "")}
            continue
        grid = on_grid(index, tuple(spec["watch"]), settings.interval_s)

        async def one(t: float, path: Path, step_id: str = step_id):
            async with gate:
                frame = frame_of(t, path)
                began = time.monotonic()
                try:
                    if step_id == "identify":
                        out = await judge_identify(
                            frame, perceive, steps, settings.timeout_s
                        )
                    else:
                        out = await judge_check(
                            step_id,
                            frame,
                            perceive=perceive,
                            decide=decide,
                            steps=steps,
                            unit=unit,
                            target_c=target_c,
                            timeout_s=settings.timeout_s,
                        )
                except Exception as exc:  # noqa: BLE001 - scored as unknown
                    out = {"label": "unknown", "reason": f"error: {exc!r}"}
                out.update(
                    step=step_id,
                    t=round(t, 2),
                    frame=path.name,
                    expected=expected(t, spec, margin),
                    seconds=round(time.monotonic() - began, 2),
                )
                return out

        records = await asyncio.gather(*(one(t, p) for t, p in grid))
        result = {"frames": list(records), "score": score(list(records))}
        if step_id == "identify":
            want = spec.get("expect") or {}
            hits = [r for r in records if r["expected"] == "positive"]
            for field in ("tea_type", "tea_name"):
                needle = want.get(f"{field}_contains")
                if needle and hits:
                    ok = sum(
                        1
                        for r in hits
                        if needle.lower() in (r.get(field) or "").lower()
                    )
                    result["score"][f"{field}_correct"] = ok / len(hits)
        else:
            result["replay"] = replay(
                list(records),
                spec,
                required=settings.required,
                window=settings.window,
                margin_s=margin,
            )
        results[step_id] = result
    return results


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def markdown(name: str, results: dict[str, Any], models: str) -> str:
    lines = [
        f"# Step accuracy: {name}",
        "",
        f"Models: {models}.",
        "",
        "| Step | Pos frames | Neg frames | Recall | False pos. | Precision"
        " | Unknown | Traps judged pos. | Live rule |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for step_id, result in results.items():
        if result.get("no_data"):
            lines.append(f"| {step_id} | no data | | | | | | | |")
            continue
        s = result["score"]
        r = result.get("replay")
        rule = "-"
        if r:
            rule = r["verdict"]
            if r["fired_at_s"] is not None:
                rule += f" at {r['fired_at_s']:g} s"
            if r["done_from_s"] is not None:
                rule += f" (done from {r['done_from_s']:g} s)"
        traps = f"{s['traps_judged_positive']} of {s['traps']}"
        lines.append(
            f"| {step_id} | {s['positive_frames']} | {s['negative_frames']}"
            f" | {_pct(s['recall'])} | {_pct(s['false_positive_rate'])}"
            f" | {_pct(s['precision'])} | {_pct(s['unknown_rate'])}"
            f" | {traps if s['traps'] else '-'} | {rule} |"
        )
        extra = {k: v for k, v in s.items() if k.endswith("_correct")}
        for key, value in extra.items():
            lines.append(f"|  | {key}: {_pct(value)} | | | | | | | |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--frames-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/eval"))
    parser.add_argument("--fakes", type=Path, help="scripted replies")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args(argv)

    import app as sample  # the roles, settings and wording the app uses

    steps = Steps.load(SAMPLE / "steps.yaml")
    settings = sample.SETTINGS.tea
    if args.fakes:
        from xr_ai_app.devkit import fakes_from_file

        fakes = fakes_from_file(args.fakes)
        perceive, decide = fakes.role("perceive"), fakes.role("decide")
        models = f"scripted ({args.fakes.name})"
    else:
        perceive, decide = sample.perceive, sample.decide
        models = "yaml/models.json (perceive, decide)"
    labels = yaml.safe_load(args.labels.read_text(encoding="utf-8"))
    index = load_index(args.frames_index)
    results = asyncio.run(
        evaluate(
            labels,
            index,
            perceive=perceive,
            decide=decide,
            steps=steps,
            settings=settings,
            concurrency=args.concurrency,
        )
    )
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "frames.jsonl").open("w", encoding="utf-8") as f:
        for step in results.values():
            for record in step.get("frames", []):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "frames"}
        for k, v in results.items()
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    name = labels.get("recording", args.labels.stem)
    report = markdown(name, results, models)
    (args.out / "summary.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
