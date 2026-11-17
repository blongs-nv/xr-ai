# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``python -m xr_ai_app.devkit``: run a scenario against an app, or serve it on a local endpoint.

    python -m xr_ai_app.devkit run  path/to/app.py scenario.yaml --out runs/x/            # scripted replies beside it
    python -m xr_ai_app.devkit run  path/to/app.py scenarios/ --out runs/                 # every scenario in a folder
    python -m xr_ai_app.devkit run  path/to/app.py scenarios/ --real-models --media-dir ~/recordings --out runs/
    python -m xr_ai_app.devkit serve path/to/app.py --fakes fakes.yaml --port 8765
    python -m xr_ai_app.devkit exercise path/to/app.py fn:units.py:bed_clear --video clip.mp4 --from 12 --to 31.5 \
        --fakes fakes.yaml --expect Verification --between 14 30 --out runs/bed/

Roles are the app's real profile (``--real-models``: real model calls, only when asked for) or scripted: ``--fakes``,
else the scenario's own ``<name>.fakes.yaml`` (or ``fakes.yaml``) beside it, else an empty script, so any model call
fails loudly. ``--media-dir``: a scenario's media is looked up there first (your own recordings, same file names).

The exit code of ``run`` is 0 when every check (of every scenario) passed, 1 otherwise; of ``exercise``, 0 when the
unit met ``--expect``/``--between`` (or, without them, did not raise, and its result did not say it failed).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .device import LocalDevice, fakes_from_file, load_app
from .exercise import exercise, resolve_target
from .runner import run_scenario
from .scenario import Scenario
from .server import serve
from .virtual_time import run_virtual


def _models(args: argparse.Namespace):  # noqa: ANN202
    if args.real_models:
        return "real"
    return fakes_from_file(args.fakes) if args.fakes else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xr-ai-dev", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "serve", "exercise"):
        p = sub.add_parser(name)
        p.add_argument("app", help="the app file (its module-level `app`)")
        if name == "exercise":
            p.add_argument("target", help="fn:<file.py or module>:<attr>, or a kit's: destination:<name>")
            p.add_argument("--video", help="a video file to play into the camera")
            p.add_argument("--from", dest="from_s", type=float, default=0.0, help="window start, seconds")
            p.add_argument("--to", dest="to_s", type=float, default=None, help="window end, seconds")
            p.add_argument("--frames-dir", help="a folder of images to play instead of a video")
            p.add_argument("--fps", type=float, default=2.0)
            p.add_argument("--say", action="append", default=[], metavar="SECONDS:TEXT", help="a timed typed turn")
            p.add_argument("--text", help="the turn text for a destination target")
            p.add_argument(
                "--option",
                action="append",
                default=[],
                metavar="KEY=JSON|@FILE",
                help="a target option: facts={...} or results={...} for a step, "
                "annotations=@windows.yaml for a procedure",
            )
            p.add_argument("--setup", help="file.py:fn or module:fn, called as fn(p, probe) before the unit")
            p.add_argument("--linger", type=float, default=3.0, help="seconds the unit may run past the window")
            p.add_argument("--expect", help="the outcome expected: the result's type name, returned, unfinished...")
            p.add_argument(
                "--between",
                nargs=2,
                type=float,
                metavar=("EARLIEST", "LATEST"),
                help="the clip time the unit must end in",
            )
            p.add_argument("--out", default=None, help="write exercise.json and trace.jsonl here")
            p.add_argument(
                "--clock",
                choices=("virtual", "real"),
                default=None,
                help="virtual: skip idle waits (look intervals, timers, frame spacing), wait for real "
                "model calls; real: wall clock. Default: virtual with scripted roles, real with "
                "--real-models",
            )
        if name == "run":
            p.add_argument("scenario", help="a scenario file, or a folder: every scenario in it")
            p.add_argument("--media-dir", help="look for the scenarios' media here first (your own recordings)")
            p.add_argument("--out", default=None, help="report directory")
            p.add_argument(
                "--clock",
                choices=("fast", "virtual", "real"),
                default="fast",
                help="fast: skip idle time between steps; virtual: also skip every idle wait inside the "
                "app (look intervals, timers), still waiting for real model calls; real: wall clock",
            )
            p.add_argument("--relay", action="store_true", help="also write NeMo Relay ATOF/ATIF under --out")
        else:
            p.add_argument("--port", type=int, default=8765)
        group = p.add_mutually_exclusive_group()
        group.add_argument("--fakes", help="YAML of scripted role replies")
        group.add_argument("--real-models", action="store_true", help="use the app's model profile (real calls)")
        p.add_argument("--speed", type=float, default=1.0, help="media playback speed")
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run(args)
    app = load_app(args.app)
    models = _models(args)

    if args.command == "exercise":
        return _exercise(app, models, args)

    async def serve_forever() -> None:
        async with LocalDevice(app, models=models, speed=args.speed) as device:
            await serve(device, port=args.port)

    try:
        asyncio.run(serve_forever())
    except KeyboardInterrupt:
        pass
    return 0


def _exercise(app, models, args: argparse.Namespace) -> int:  # noqa: ANN001
    from pathlib import Path  # noqa: PLC0415

    base = Path(args.app).resolve().parent
    options = {k: _option(v, base) for k, _, v in (o.partition("=") for o in args.option)}
    if args.text is not None:
        options["text"] = args.text
    unit = resolve_target(app, args.target, options, base=base)
    setup = resolve_target(app, f"fn:{args.setup}", base=base) if args.setup else None
    say = [(float(at), text) for at, _, text in (s.partition(":") for s in args.say)]
    clock = args.clock or ("real" if args.real_models else "virtual")
    result = (run_virtual if clock == "virtual" else asyncio.run)(
        exercise(
            app,
            unit,
            target=args.target,
            video=args.video,
            from_s=args.from_s,
            to_s=args.to_s,
            frames_dir=args.frames_dir,
            fps=args.fps,
            say=say,
            setup=setup,
            models=models,
            speed=args.speed,
            linger_s=args.linger,
        )
    )
    if args.out:
        result.write(args.out)
    sys.stdout.write(result.to_json() + "\n")
    ok = (result.status != "raised" and result.passed) if args.expect is None else result.outcome == args.expect
    if args.between is not None:
        earliest, latest = args.between
        ok = ok and result.ended_at_s is not None and earliest <= result.ended_at_s <= latest
    return 0 if ok else 1


def _run(args: argparse.Namespace) -> int:
    from pathlib import Path  # noqa: PLC0415

    path = Path(args.scenario)
    files = sorted(f for f in path.iterdir() if _is_scenario(f)) if path.is_dir() else [path]
    if not files:
        sys.stderr.write(f"no scenarios in {path}\n")
        return 1
    runner = run_virtual if args.clock == "virtual" else asyncio.run
    reports = []
    for file in files:
        app = load_app(args.app)  # a fresh app for each scenario: an app starts once
        if args.real_models:
            models = "real"
        elif args.fakes:
            models = fakes_from_file(args.fakes)
        else:
            beside = _fakes_beside(file)
            models = fakes_from_file(beside) if beside is not None else None
        out = None if args.out is None else (Path(args.out) / file.stem if path.is_dir() else Path(args.out))
        report = runner(
            run_scenario(
                app,
                Scenario.load(file),
                models=models,
                speed=args.speed,
                clock="fast" if args.clock == "virtual" else args.clock,
                out=out,
                relay=args.relay,
                media_dir=args.media_dir,
            )
        )
        reports.append((file, report))
        if not path.is_dir():
            sys.stdout.write(report.transcript())
    if path.is_dir():
        for file, report in reports:
            sys.stdout.write(f"{'PASS' if report.passed else 'FAIL'}  {report.scenario}  ({file.name})\n")
            for failure in report.failures:
                sys.stdout.write(f"      step {failure.index} {failure.action}: {failure.detail}\n")
    return 0 if all(report.passed for _, report in reports) else 1


def _is_scenario(file) -> bool:  # noqa: ANN001
    import yaml  # noqa: PLC0415

    if file.suffix not in (".yaml", ".yml") or file.name.endswith((".fakes.yaml", ".fakes.yml")):
        return False
    data = yaml.safe_load(file.read_text(encoding="utf-8"))
    return isinstance(data, dict) and "steps" in data


def _fakes_beside(file):  # noqa: ANN001, ANN202
    for candidate in (file.with_name(f"{file.stem}.fakes.yaml"), file.with_name("fakes.yaml")):
        if candidate.exists():
            return candidate
    return None


def _option(value: str, base):  # noqa: ANN001, ANN202
    """A target option: JSON, or ``@file`` (YAML or JSON, relative to the app's directory)."""
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import yaml  # noqa: PLC0415

    if value.startswith("@"):
        path = Path(value[1:])
        return yaml.safe_load((path if path.is_absolute() else base / path).read_text(encoding="utf-8"))
    return json.loads(value)


if __name__ == "__main__":
    raise SystemExit(main())
