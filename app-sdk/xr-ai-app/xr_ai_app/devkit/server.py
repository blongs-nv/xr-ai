# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A local HTTP endpoint for a :class:`~xr_ai_app.devkit.LocalDevice`: drive a running app with ``curl``.

Bound to 127.0.0.1 only. JSON in and out (``POST /frame`` takes the image bytes as its body).

=========================  =================================================================================
``GET  /health``           ``{"ok": true}``
``POST /join``             ``{"participant": "p1"}``
``POST /leave``            ``{"participant": "p1"}``
``POST /say``              ``{"text": "...", "participant": "p1"}``: a typed turn
``POST /frame``            body: an encoded image; ``?participant=p1``
``POST /image``            ``{"path": "...", "participant": "p1"}``
``POST /frames``           ``{"dir": "...", "fps": 1, "participant": "p1"}`` (returns when played)
``POST /video``            ``{"path": "...", "fps": 2, "from_s": 0, "to_s": 10, "participant": "p1"}``
``POST /settle``           wait until the app is idle
``GET  /output``           ``?after=<n>&wait_s=<s>&participant=p1``: outputs from index ``n`` (waits up to
                           ``wait_s`` for at least one); ``{"outputs": [...], "next": <index>}``
``GET  /trace``            ``?turn=<id>&kind=<kind>``: trace events
=========================  =================================================================================
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any
from urllib.parse import parse_qs, urlsplit

from loguru import logger

from .device import LocalDevice, image_frame

__all__ = ["serve"]

_MAX_BODY = 32 * 1024 * 1024


async def _handle(device: LocalDevice, method: str, path: str, query: dict[str, str], body: bytes) -> Any:
    data: dict[str, Any] = {}
    if body and path != "/frame":
        data = json.loads(body)
    who = str(data.get("participant") or query.get("participant") or "p1")
    if (method, path) == ("GET", "/health"):
        return {"ok": True}
    if (method, path) == ("POST", "/join"):
        await device.join(who)
        return {"ok": True}
    if (method, path) == ("POST", "/leave"):
        await device.leave(who)
        return {"ok": True}
    if (method, path) == ("POST", "/say"):
        await device.say(str(data["text"]), who)
        return {"ok": True, "output_index": len(device.hub.outputs)}
    if (method, path) == ("POST", "/frame"):
        await device.show(image_frame(body), who, label="http")
        return {"ok": True}
    if (method, path) == ("POST", "/image"):
        await device.show_image(data["path"], who)
        return {"ok": True}
    if (method, path) == ("POST", "/frames"):
        count = await device.play_frames(data["dir"], fps=float(data.get("fps", 1)), participant=who)
        return {"ok": True, "frames": count}
    if (method, path) == ("POST", "/video"):
        count = await device.play_video(
            data["path"],
            fps=float(data.get("fps", 2)),
            from_s=float(data.get("from_s", 0)),
            to_s=None if data.get("to_s") is None else float(data["to_s"]),
            participant=who,
        )
        return {"ok": True, "frames": count}
    if (method, path) == ("POST", "/settle"):
        await device.settle()
        return {"ok": True}
    if (method, path) == ("GET", "/output"):
        after = int(query.get("after", 0))
        wait_s = float(query.get("wait_s", 0))
        participant = query.get("participant")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_s
        while not device.outputs(after=after, participant=participant) and loop.time() < deadline:
            await asyncio.sleep(0.05)
        outputs = device.outputs(after=after, participant=participant)
        return {"outputs": [asdict(o) for o in outputs], "next": len(device.hub.outputs)}
    if (method, path) == ("GET", "/trace"):
        events = device.trace.events(kind=query.get("kind"), turn=query.get("turn"))
        return {"events": [json.loads(e.to_json()) for e in events]}
    raise LookupError(f"no route {method} {path}")


async def _serve_connection(device: LocalDevice, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        request_line = (await reader.readline()).decode("latin-1").strip()
        if not request_line:
            return
        method, target, _ = request_line.split(" ", 2)
        headers: dict[str, str] = {}
        while (line := (await reader.readline()).decode("latin-1").strip()) != "":
            name, _, value = line.partition(":")
            headers[name.strip().lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        if length > _MAX_BODY:
            raise ValueError("body too large")
        body = await reader.readexactly(length) if length else b""
        url = urlsplit(target)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            payload, status = await _handle(device, method.upper(), url.path, query, body), 200
        except LookupError as exc:
            payload, status = {"error": str(exc)}, 404
        except Exception as exc:  # noqa: BLE001 - report to the client
            payload, status = {"error": f"{type(exc).__name__}: {exc}"}, 400
        data = json.dumps(payload, default=str).encode()
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found"}[status]
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n"
            "Connection: close\r\n\r\n".encode()
            + data
        )
        await writer.drain()
    except Exception:  # noqa: BLE001 - one bad request never stops the server
        logger.exception("devkit: request failed")
    finally:
        writer.close()


async def serve(device: LocalDevice, *, port: int = 8765, ready: asyncio.Event | None = None) -> None:
    """Serve ``device`` on 127.0.0.1:``port`` until cancelled (the device must already be entered)."""
    server = await asyncio.start_server(lambda r, w: _serve_connection(device, r, w), "127.0.0.1", port)
    logger.info("xr-ai-dev: serving on http://127.0.0.1:{}", port)
    if ready is not None:
        ready.set()
    async with server:
        await server.serve_forever()
