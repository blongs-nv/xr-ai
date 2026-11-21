# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``python -m xr_ai_app.devkit``: run a scenario against an app, or serve it on a local endpoint.

    python -m xr_ai_app.devkit run  path/to/app.py scenario.yaml --out runs/x/            # scripted replies beside it
    python -m xr_ai_app.devkit run  path/to/app.py scenarios/ --out runs/                 # every scenario in a folder
    python -m xr_ai_app.devkit run  path/to/app.py scenarios/ --real-models --media-dir ~/recordings --out runs/
    python -m xr_ai_app.devkit prepare path/to/app.py speech.yaml --out media/
    python -m xr_ai_app.devkit run path/to/app.py speech.yaml --real-models --real-audio --clock real --media-dir media/
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
            p.add_argument("--real-audio", action="store_true", help="Replay WAVs through real hub/STT/TTS")
            p.add_argument("--image", action="append", default=[], help="Image NAME=PATH override")
            p.add_argument("--metadata", help="Operator JSON file describing hardware and serving configuration")
            p.add_argument("--profile", help="Model profile name or path, local or hosted")
            p.add_argument("--drain", type=float, default=10.0, help="Real-audio response drain, seconds")
            p.add_argument("--ready-timeout", type=float, default=600.0, help="Startup deadline, seconds")
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
    prepare = sub.add_parser("prepare", help="Synthesize and pin fixed speech/media before measurement")
    prepare.add_argument("app", help="the app file (its module-level `app`)")
    prepare.add_argument("scenario", help="Real-audio scenario file")
    prepare.add_argument("--profile", help="Model profile name or path, local or hosted")
    prepare.add_argument("--out", required=True, help="Directory for pinned fixtures")
    prepare.add_argument("--tts-entry", help="Explicit TTS model entry for fixture synthesis")
    prepare.add_argument("--image", action="append", default=[], help="Image NAME=PATH input")
    args = parser.parse_args(argv)
    if args.command == "prepare" or args.real_models:
        try:
            from xr_ai_launcher import load_credentials  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            if exc.name != "xr_ai_launcher":
                raise
            sys.stderr.write("Credential store unavailable; use environment credentials or install xr-ai-app[audio].\n")
        else:
            load_credentials()
    if args.command == "prepare":
        from ._fixtures import _prepare_fixtures  # noqa: PLC0415

        scenario = Scenario.load(args.scenario)
        app = load_app(args.app, profile=args.profile)
        result = asyncio.run(
            _prepare_fixtures(app, scenario, args.out, images=_image_bindings(args.image), tts_entry=args.tts_entry)
        )
        sys.stdout.write(str(result) + "\n")
        return 0

    if args.command == "run":
        if args.real_audio and (not args.real_models or args.clock != "real" or args.speed != 1):
            parser.error("--real-audio requires --real-models --clock real --speed 1")
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
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    metadata = json.loads(Path(args.metadata).read_text()) if args.metadata else None
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    images = _image_bindings(args.image)
    path = Path(args.scenario)
    files = sorted(f for f in path.iterdir() if _is_scenario(f)) if path.is_dir() else [path]
    if not files:
        sys.stderr.write(f"no scenarios in {path}\n")
        return 1
    runner = run_virtual if args.clock == "virtual" else asyncio.run
    reports = []
    for file in files:
        scenario = Scenario.load(file)
        # An app starts once, so each scenario gets a fresh instance.
        app = load_app(args.app, profile=args.profile)
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
                scenario,
                models=models,
                speed=args.speed,
                clock="fast" if args.clock == "virtual" else args.clock,
                out=out,
                relay=args.relay,
                media_dir=args.media_dir,
                real_audio=args.real_audio,
                drain_s=args.drain,
                ready_timeout_s=args.ready_timeout,
                metadata=metadata,
                images=images,
            )
        )
        reports.append((file, report))
        if not path.is_dir():
            sys.stdout.write(report.transcript())
        if report.performance is not None:
            fields = (
                "execution_status",
                "timing_comparable",
                "handler_coverage",
                "reply_coverage",
                "error_events",
                "stt",
                "max_sender_lateness_s",
            )
            sys.stdout.write(
                json.dumps({"scenario": report.scenario, **{key: report.performance[key] for key in fields}}) + "\n"
            )
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


def _image_bindings(values):
    from pathlib import Path  # noqa: PLC0415

    images = {}
    for value in values:
        name, separator, path = value.partition("=")
        name = name if Path(name).suffix else name + ".png"
        if not separator or name == ".png" or not path or name in images:
            raise ValueError("images must be unique NAME=PATH bindings")
        images[name] = Path(path).expanduser().resolve()
    return images


if __name__ == "__main__":
    raise SystemExit(main())
