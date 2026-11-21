# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real speech execution for the devkit's scenario and report contract."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import hashlib
import importlib.util
import io
import json
import logging
import math
import multiprocessing
import tempfile
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

from loguru import logger

from .._base._tracing import MemorySink
from .runner import Report, StepResult, _Media, _relay_context, _write


def _contains(text, word):
    import re  # noqa: PLC0415

    pattern = re.escape(str(word))
    if str(word)[0].isalnum():
        pattern = r"(?<!\w)" + pattern
    if str(word)[-1].isalnum():
        pattern += r"(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _positive(value, name, *, zero=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _wav_samples(raw, label):
    import numpy as np  # noqa: PLC0415

    with wave.open(io.BytesIO(raw), "rb") as wav:
        rate, channels = wav.getframerate(), wav.getnchannels()
        if (
            not 1 <= rate <= 192000
            or channels not in (1, 2)
            or wav.getsampwidth() != 2
            or wav.getcomptype() != "NONE"
        ):
            raise ValueError(f"{label}: audio must be uncompressed mono/stereo PCM16 WAV")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype("<f4") / 32768
        samples = samples.reshape(-1, channels).mean(axis=1)
        if rate != 16000:
            from scipy.signal import resample_poly  # noqa: PLC0415

            divisor = math.gcd(rate, 16000)
            samples = resample_poly(samples, 16000 // divisor, rate // divisor).astype("<f4")
    if not len(samples):
        raise ValueError(f"{label}: audio must be nonempty")
    return samples


def _prepare_audio(scenario, media_dir, drain_s, *, images=None):
    from PIL import Image  # noqa: PLC0415

    _positive(drain_s, "drain_s", zero=True)
    media = _Media(scenario.base_dir, None if media_dir is None else Path(media_dir))
    events, checks, hashes, labels = [], [], {}, {}
    at, audio_end, last_at = 0.0, 0.0, -1.0
    joined, left = False, False
    last_input_at = 0.0
    for index, step in enumerate(scenario.steps):
        if step.participant and step.participant != scenario.participant:
            raise ValueError("real audio currently supports one participant per scenario")
        if step.at is not None:
            _positive(step.at, f"step {index} at", zero=True)
            at = at + step.at if step.relative else step.at
        if at < last_at:
            raise ValueError("real audio steps must be ordered on a fixed timeline")
        last_at = at
        value = step.value
        event = {"index": index, "action": step.action, "at": at}
        if step.action in {"expect", "wait_for"}:
            spec = value if isinstance(value, dict) else {"said": value}
            if set(spec) - {"said", "said_any", "within_s", "unprompted"} or ("said" in spec) == ("said_any" in spec):
                raise ValueError("audio checks require said or said_any, and optional within_s/unprompted")
            if "unprompted" in spec and type(spec["unprompted"]) is not bool:
                raise ValueError("audio check unprompted must be boolean")
            words = spec.get("said_any", [spec.get("said")])
            if not isinstance(words, list) or not words or any(not isinstance(word, str) or not word for word in words):
                raise ValueError("audio checks need nonempty text or a nonempty said_any list")
            timeout = _positive(spec.get("within_s", 10), "within_s", zero=True)
            checks.append((index, at, spec, last_input_at, words, timeout))
            continue
        if step.action == "sleep":
            at += _positive(value, "sleep", zero=True)
            continue
        if left:
            raise ValueError("real audio inputs cannot follow leave")
        if step.action == "join":
            if joined or at != 0 or str(value or scenario.participant) != scenario.participant:
                raise ValueError("real audio requires exactly one join at time zero for the scenario participant")
            joined = True
        elif not joined:
            raise ValueError("real audio scenarios must start with join")
        elif step.action == "audio":
            if (
                not isinstance(value, dict)
                or set(value) - {"path", "text", "handler", "reply", "sha256"}
                or "path" not in value
            ):
                raise ValueError("audio requires path and optional text, handler, reply, sha256")
            if any(type(value[key]) is not bool for key in ("handler", "reply") if key in value):
                raise ValueError("audio.handler and audio.reply must be booleans")
            if value.get("reply", value.get("handler", True)) and not value.get("handler", True):
                raise ValueError("audio.reply requires handler: true")
            path = media / str(value["path"])
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if "sha256" in value and value["sha256"] != digest:
                raise ValueError(f"audio hash mismatch: {path}")
            hashes[str(value["path"])] = digest
            if "text" in value:
                labels[str(value["path"])] = value["text"]
            samples = _wav_samples(raw, path)
            audio_at = math.ceil(at * 50 - 1e-9) / 50
            if not len(samples) or audio_at < audio_end:
                raise ValueError("audio clips must be nonempty and must not overlap")
            event.update(
                samples=samples.tobytes(),
                duration_s=len(samples) / 16000,
                text=str(value.get("text", path.stem)),
                handler=value.get("handler", True),
                reply=value.get("reply", value.get("handler", True)),
            )
            audio_end = audio_at + event["duration_s"]
        elif step.action == "image":
            stream = value.get("stream", True) if isinstance(value, dict) else True
            if isinstance(value, dict):
                if set(value) - {"path", "stream"} or type(stream) is not bool:
                    raise ValueError("real audio image requires path and optional boolean stream")
                value = value["path"]
            path = (images or {}).get(str(value), media / str(value))
            raw = Path(path).read_bytes()
            hashes[str(value)] = hashlib.sha256(raw).hexdigest()
            with Image.open(io.BytesIO(raw)) as source:
                image = source.convert("RGB")
                image.thumbnail((1280, 1280))
                pixels = image.tobytes()
                event.update(pixels=pixels, width=image.width, height=image.height, stream=stream)
                if not stream:
                    encoded = io.BytesIO()
                    image.save(encoded, format="PNG")
                    event["encoded"] = encoded.getvalue()
        elif step.action == "leave":
            if at < audio_end:
                raise ValueError("leave must follow the end of all audio")
            left = True
        else:
            raise ValueError(
                f"{step.action}: real audio supports join, audio, image, sleep, expect, wait_for and leave"
            )
        events.append(event)
        last_input_at = at
    image_names = {
        str(step.value["path"] if isinstance(step.value, dict) else step.value)
        for step in scenario.steps
        if step.action == "image"
    }
    if images and set(images) - image_names:
        raise ValueError(f"unknown image bindings: {sorted(set(images) - image_names)}")
    if not any(event["action"] == "audio" for event in events):
        raise ValueError("real audio scenario needs at least one audio clip")
    duration = scenario.duration_s if scenario.duration_s is not None else max(at, audio_end)
    _positive(duration, "duration_s")
    if duration < max(at, audio_end):
        raise ValueError("duration_s must contain all inputs and checks")
    if any(event["action"] == "leave" and event["at"] < duration for event in events):
        raise ValueError("leave must be at the end of the real audio timeline")
    if media_dir is not None and (manifest_path := Path(media_dir) / "fixtures.json").is_file():
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
            raise ValueError("fixture manifest requires a files mapping")
        for name, digest in hashes.items():
            if name not in (images or {}) and manifest["files"].get(name) != digest:
                raise ValueError(f"pinned fixture changed or missing: {name}")
        for name, text in labels.items():
            if name in manifest.get("utterances", {}) and manifest["utterances"][name] != text:
                raise ValueError(f"pinned audio has different text: {name}")
    return {
        "events": events,
        "checks": checks,
        "duration_s": duration,
        "drain_s": drain_s,
        "participant": scenario.participant,
        "fixture_sha256": hashes,
    }


_RUN_LOCK = threading.Lock()


async def _run_audio(app, scenario, **kwargs):
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("only one real-audio replay may run per process")
    try:
        return await _run_target(app, scenario, **kwargs)
    finally:
        _RUN_LOCK.release()


async def _run_target(app, scenario, *, plan, out, ready_timeout_s, metadata=None, relay=False):
    from xr_ai_hub.isolation import isolated_endpoints  # noqa: PLC0415
    from xr_ai_tools.tool_calling import observe_tool_errors  # noqa: PLC0415
    from xr_ai_voice.diagnostics import observe  # noqa: PLC0415

    from ._audio_report import _performance  # noqa: PLC0415
    from ._audio_source import _main, _receive  # noqa: PLC0415
    from ._targets import _AppTarget, _source_identity  # noqa: PLC0415

    if importlib.util.find_spec("device_io_hub") is None:
        raise ImportError("real audio requires xr-ai-app[audio]")
    _positive(ready_timeout_s, "ready_timeout_s")
    target = _AppTarget(app)
    scenario_snapshot = dataclasses.asdict(scenario)
    scenario_snapshot.pop("base_dir")
    identity = {
        "scenario_sha256": hashlib.sha256(json.dumps(scenario_snapshot, sort_keys=True).encode()).hexdigest(),
        "duration_s": plan["duration_s"],
        "drain_s": plan["drain_s"],
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **target.identity,
        **_source_identity(),
        "fixture_sha256": plan["fixture_sha256"],
        "operator_metadata": metadata or {},
    }
    trace, measured, speech = MemorySink(), [], []
    results = []
    started = None
    ended = None
    completed = False
    cancelled = None
    source_rows = []

    outputs_seen = []
    output_ready = asyncio.Event()
    loop = asyncio.get_running_loop()

    def observed(event):
        trace(event)
        now = time.monotonic()
        if event.kind != "output":
            measured.append((now, event))
        if event.kind != "response" or event.data.get("message_kind") == "withdraw":
            return
        data = event.data
        pid = data.get("participant_id")
        if pid is None:
            return
        text = data.get("text", "")
        if text.strip():
            outputs_seen.append((now, text, data.get("message_kind")))
            loop.call_soon_threadsafe(output_ready.set)
            measured.append(
                (
                    now,
                    SimpleNamespace(
                        kind="output", participant=pid, data={"text": text, "urgent": data.get("urgent", False)}
                    ),
                )
            )

    def diagnostic(event):
        if event["kind"] == "response":
            from .._base._tracing import Event, current_ids, new_id  # noqa: PLC0415

            observed(Event(kind="response", name="voice", id=new_id("e"), t=time.time(),
                           participant=event["participant"], turn=current_ids().turn, data=event))
        else:
            speech.append({"monotonic": time.monotonic(), **event})

    def log_error(message):
        record = message.record
        exception = record["exception"].type if record["exception"] else None
        if record["level"].no >= 40 or exception:
            speech.append(
                {
                    "monotonic": time.monotonic(),
                    "kind": "error",
                    "exception_type": exception.__name__ if exception else None,
                }
            )

    class ErrorHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR or record.exc_info:
                diagnostic(
                    {
                        "kind": "error",
                        "logger": record.name,
                        "function": record.funcName,
                        "exception_type": record.exc_info[0].__name__ if record.exc_info else None,
                    }
                )

    async with contextlib.AsyncExitStack() as cleanup:
        root = Path(cleanup.enter_context(tempfile.TemporaryDirectory(prefix="xr-audio-")))
        directory = str(root)
        push, pub = f"ipc://{root}/in", f"ipc://{root}/pub"
        if out is not None:
            Path(out).mkdir(parents=True, exist_ok=False)
            (Path(out) / ".gitignore").write_text("*\n")
            (Path(out) / "scenario.json").write_text(json.dumps(scenario_snapshot, indent=2) + "\n")
        if relay:
            await cleanup.enter_async_context(_relay_context(app, scenario, out))
        context = multiprocessing.get_context("spawn")
        connection, child = context.Pipe()
        cleanup.callback(connection.close)
        cleanup.callback(child.close)
        process = context.Process(target=_main, args=(plan, directory, push, pub, root.name, child))
        target.trace.add(observed)
        cleanup.callback(target.trace.remove, observed)
        sink = None
        standard_logger = logging.getLogger()
        standard_sink = ErrorHandler(level=logging.WARNING)
        cleanup.callback(standard_logger.removeHandler, standard_sink)

        def logging_ready():
            nonlocal sink
            remove_sink()
            sink = logger.add(log_error, level="WARNING")
            standard_logger.removeHandler(standard_sink)
            standard_logger.addHandler(standard_sink)

        def remove_sink():
            try:
                if sink is not None:
                    logger.remove(sink)
            except ValueError:
                pass

        cleanup.callback(remove_sink)
        logging_ready()
        cleanup.enter_context(isolated_endpoints(sub_addr=pub, push_addr=push))
        cleanup.enter_context(target.configure(root, Path(out) if out is not None else root))
        cleanup.enter_context(observe_tool_errors(diagnostic))

        async def signal():
            return await _receive(connection)

        with observe(diagnostic):
            try:
                process.start()
                child.close()
                async with asyncio.timeout(ready_timeout_s):
                    if await signal() != "ready":
                        raise RuntimeError("audio source failed readiness")
                    await target.start()
                    while not target.ready():
                        if not process.is_alive():
                            raise RuntimeError("audio source exited before readiness")
                        await asyncio.sleep(0.02)
                started = time.monotonic() + 0.5
                connection.send(
                    {
                        "command": "start",
                        "monotonic": started,
                        "epoch_us": time.time_ns() // 1000 + round((started - time.monotonic()) * 1e6),
                    }
                )
                if await asyncio.wait_for(signal(), 5) != "started":
                    raise RuntimeError("audio source did not start")
                async with asyncio.timeout(plan["duration_s"] + plan["drain_s"] + 10):
                    for index, at, value, after, needles, timeout in plan["checks"]:
                        await asyncio.sleep(max(0, started + at - time.monotonic()))
                        spec = value
                        deadline = min(started + at + timeout, started + plan["duration_s"] + plan["drain_s"])

                        def matches():
                            return any(
                                started + after <= t <= deadline
                                and (not spec.get("unprompted") or kind == "update")
                                and any(_contains(text, word) for word in needles)
                                for t, text, kind in outputs_seen
                            )

                        while True:
                            output_ready.clear()
                            if matches() or time.monotonic() >= deadline:
                                break
                            try:
                                await asyncio.wait_for(output_ready.wait(), deadline - time.monotonic())
                            except TimeoutError:
                                break
                        ok = matches()
                        results.append(
                            StepResult(
                                index, "expect", value, at, ok, "" if ok else "no matching output before deadline"
                            )
                        )
                    if await signal() != "done":
                        raise RuntimeError("audio source did not finish")
                completed = True
            except asyncio.CancelledError as exc:
                cancelled = exc
                results.append(StepResult(-1, "execution", None, 0, False, "CancelledError"))
            except Exception as exc:  # noqa: BLE001 - an execution failure is a failed report
                results.append(StepResult(-1, "execution", None, 0, False, type(exc).__name__ + ": " + str(exc)))
            finally:
                ended = time.monotonic()
                try:
                    await target.stop()
                finally:
                    try:
                        connection.send({"command": "stop"})
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                    connection.close()
                    child.close()
                    if process.pid is not None:
                        deadline = time.monotonic() + 5
                        while process.is_alive() and time.monotonic() < deadline:
                            await asyncio.sleep(0.02)
                        if process.is_alive():
                            process.kill()
                            completed = False
                        process.join()
                        if process.exitcode != 0:
                            completed = False
                            # The connector's SHM belongs exclusively to this run.
                            from multiprocessing.shared_memory import SharedMemory  # noqa: PLC0415

                            try:
                                memory = SharedMemory(name=root.name.replace("-", "_"), create=False)
                            except FileNotFoundError:
                                pass
                            else:
                                memory.close()
                                memory.unlink()
                        process.close()
                    path = root / "source.jsonl"
                    if path.exists():
                        source_rows = [json.loads(line) for line in path.read_text().splitlines()]
    inputs = [
        {"t": row["t"], "kind": row["action"], "participant": scenario.participant, "label": row.get("text", "")}
        for row in source_rows
        if row["kind"] == "input"
    ]
    for row in source_rows:
        if row["kind"] == "input":
            results.append(StepResult(row["index"], row["action"], row.get("text", ""), row["t"]))
    outputs = [
        {
            "t": t - started,
            "participant": event.participant,
            "text": event.data.get("text", ""),
            "urgent": event.data.get("urgent", False),
        }
        for t, event in measured
        if started is not None and t >= started and event.kind == "output" and event.data.get("text")
    ]
    performance = _performance(plan, source_rows, measured, speech, started, ended, completed)
    performance["configuration"] = identity
    performance["gpu_memory_peak_bytes"] = "unmeasured"
    report = Report(
        scenario.name,
        completed and all(result.ok for result in results) and performance["timing_comparable"],
        sorted(results, key=lambda row: row.index),
        inputs,
        outputs,
        performance,
    )
    if out is not None:
        _write(report, SimpleNamespace(trace=trace), Path(out))
        (Path(out) / "speech-events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in speech))
        (Path(out) / "audio-events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    if cancelled is not None:
        raise cancelled
    return report
