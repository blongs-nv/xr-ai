# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private hub and wall-clock sender, isolated from the app event loop."""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

import numpy as np


async def _receive(connection):
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def readable():
        if not future.done():
            try:
                future.set_result(connection.recv())
            except (EOFError, OSError):
                future.set_exception(RuntimeError("audio control peer disconnected"))

    loop.add_reader(connection.fileno(), readable)
    try:
        return await future
    finally:
        loop.remove_reader(connection.fileno())


async def _serve(plan, directory, push, pub, name, connection):
    from device_io_hub.ipc import ConnectorEndpoint, HubEndpoint  # noqa: PLC0415
    from xr_ai_hub import AudioChunk, PixelFormat  # noqa: PLC0415

    started = None
    path = Path(directory) / "source.jsonl"
    with path.open("w", buffering=1) as file:

        def emit(kind, **data):
            now = time.monotonic()
            file.write(
                json.dumps({"kind": kind, "monotonic": now, "t": None if started is None else now - started, **data})
                + "\n"
            )

        hub = HubEndpoint(pull_addr=push, pub_addr=pub)
        connector = ConnectorEndpoint(
            push_addr=push,
            sub_addr=pub,
            connector_id=name,
            shm_name=name.replace("-", "_"),
            num_slots=4,
            max_frame_bytes=6_000_000,
        )
        tasks = []
        current = None
        streaming = True

        async def capture(message):
            if message.topic != "camera.capture.request" or message.participant_id != plan["participant"]:
                return
            request = json.loads(message.data)
            if request.get("version") != 1 or not isinstance(request.get("request_id"), str):
                raise ValueError("invalid camera capture request")
            if current is None or streaming:
                await connector.push_image_capture(
                    participant_id=message.participant_id, request_id=request["request_id"],
                    pts_us=time.time_ns() // 1000,
                    mime_type="application/vnd.xr-ai.capture-rejection+json",
                    data=json.dumps({"version": 1, "status": "rejected"}).encode(),
                )
                if current is not None and streaming:
                    pixels, width, height, _encoded = current
                    await connector.push_frame(
                        pixels, width, height, PixelFormat.RGB24, time.time_ns() // 1000,
                        message.participant_id, "camera",
                    )
                emit("image_capture_rejected", participant=message.participant_id)
                return
            await connector.push_image_capture(
                participant_id=message.participant_id, request_id=request["request_id"],
                pts_us=time.time_ns() // 1000, mime_type="image/png", data=current[3],
            )
            emit("image_capture", participant=message.participant_id)

        connector.on_return_data(capture)
        try:
            tasks.append(asyncio.create_task(hub.run()))

            async def returned(chunk):
                samples = np.frombuffer(chunk.data, dtype="<f4")
                emit(
                    "audio_out",
                    rms=float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0,
                    samples=chunk.samples,
                    sample_rate=chunk.sample_rate,
                )

            connector.on_return_audio(returned)
            await connector.register()
            tasks.append(asyncio.create_task(connector.run()))
            connection.send("ready")
            command = await _receive(connection)
            if command == {"command": "stop"}:
                return
            if command["command"] != "start":
                raise ValueError("audio source expected start")
            started, epoch = command["monotonic"], command["epoch_us"]
            pid = plan["participant"]
            await asyncio.sleep(max(0, started - time.monotonic()))
            await connector.notify_participant_joined(pid, pts_us=epoch)
            owner = asyncio.create_task(_receive(connection))
            tasks.append(owner)
            connection.send("started")
            closing_leave = next((event for event in plan["events"] if event["action"] == "leave"), None)
            events = iter(event for event in plan["events"] if event["action"] != "leave")
            event = next(events, None)
            speech, offset = None, 0
            maximum_late = 0.0
            for tick in range(math.ceil((plan["duration_s"] + plan["drain_s"]) * 50) + 1):
                await asyncio.sleep(max(0, started + tick * 0.02 - time.monotonic()))
                if owner.done():
                    if await owner == {"command": "stop"}:
                        return
                    raise RuntimeError("audio replay stopped before completion")
                at = tick * 0.02
                if at < plan["duration_s"]:
                    maximum_late = max(maximum_late, time.monotonic() - started - at)
                new_image = False
                while event is not None and event["at"] <= at + 1e-9:
                    action = event["action"]
                    if action == "image":
                        current = event["pixels"], event["width"], event["height"], event.get("encoded")
                        streaming = event.get("stream", True)
                        new_image = True
                    elif action == "audio":
                        speech, offset = np.frombuffer(event["samples"], dtype="<f4"), 0
                        speech_index = event["index"]
                    data = {key: value for key, value in event.items() if key not in {"samples", "pixels", "encoded"}}
                    emit(
                        "input",
                        **data,
                        audio_start_us=epoch + tick * 20000,
                        audio_end_us=epoch + tick * 20000 + round(event.get("duration_s", 0) * 1e6),
                    )
                    event = next(events, None)
                if current is not None and streaming and (new_image or tick % 50 == 0):
                    pixels, width, height, _encoded = current
                    await connector.push_frame(
                        pixels, width, height, PixelFormat.RGB24, epoch + tick * 20000, pid, "camera"
                    )
                chunk = np.zeros(320, dtype="<f4")
                if speech is not None:
                    part = speech[offset : offset + 320]
                    chunk[: len(part)] = part
                    offset += len(part)
                    if offset >= len(speech):
                        emit(
                            "speech_end",
                            audio_end_us=epoch + tick * 20000 + len(part) * 1000000 // 16000,
                            input_index=speech_index,
                        )
                        speech = None
                await connector.push_audio(
                    AudioChunk(
                        pts_us=epoch + tick * 20000,
                        sample_rate=16000,
                        channels=1,
                        samples=320,
                        data=chunk.tobytes(),
                        participant_id=pid,
                        track_id="microphone",
                    )
                )
            if closing_leave is not None:
                await connector.notify_participant_left(pid, pts_us=epoch + tick * 20000)
                emit("input", **closing_leave)
            emit("measurement_finished", max_sender_lateness_s=maximum_late)
            connection.send("done")
            if await owner != {"command": "stop"}:
                raise ValueError("audio source expected stop")
        except BaseException as exc:
            emit("source_failed", exception_type=type(exc).__name__)
            raise
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            connector.stop()
            connector.close()
            hub.stop()
            hub.close()


def _main(*args):
    connection = args[-1]
    try:
        asyncio.run(_serve(*args))
    finally:
        connection.close()
