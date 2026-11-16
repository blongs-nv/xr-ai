# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lab instruments: read marker-labelled meters, watch them, note changes.

SHAPE: the kit's tool loop. One `Conversation` on the omni LLM picks a tool
per turn (upstream's prompt and tool set, no router):

  current_view            the camera frame asked of Cosmos, answered as is
  lab_instruments__read   find the QR/ArUco markers (OpenCV, no model), mark
                          them, ask Cosmos once for every display
                                                       (instruments.py)
  lab_instruments__start  /stop /status: a background task that reads them
                          every few seconds and speaks when one is found,
                          changes or is lost   (instrument_monitor.py)
  visual_monitor__start   /stop /status: a background task that captions the
                          view and notes changes, silently
                                                       (visual_monitor.py)
  recent_visual_history   the visual monitor's latest looks, for the model

Tools return values; the conversation is the only speaker. Eight tools are
`return_direct` (upstream's flags): their result is the answer, said as it
is (the view as Cosmos streams it) without asking the LLM again. The
monitors' notes go to a `ContextLog` the conversation reads, and every
record to JSONL files (records.py). Roles are the entry names of
yaml/models.json (upstream's): llm, vlm, tts. The SDK waits for those
servers before it reports ready.
"""

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from instrument_monitor import InstrumentMonitor
from instruments import InstrumentReader, load_device_map
from pydantic import BaseModel, Field
from records import AnswerVoice, ForegroundRecord, Records, lab_tool
from visual_monitor import VisualMonitor
from xr_ai_kit import Conversation
from xr_ai_next import App, ContextLog, Hub, Participant, PerConnection, Turn

HERE = Path(__file__).resolve().parent


class Settings(BaseModel):
    device_map_yaml: str = "yaml/device_map.yaml"
    capture_marker_scans: bool = False
    monitor_interval_s: float = 5.0
    instrument_monitor_interval_s: float = 5.0
    instrument_state_interval_s: float = 10.0
    instrument_lost_after_s: float = 30.0
    monitor_history_size: int = 20
    history_budget_bytes: int = 4_000
    frame_max_age_s: float = 5.0
    frame_timeout_s: float = 5.0


def prompt(name: str) -> str:
    path = HERE / "prompts" / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()


app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
SETTINGS = app.settings(Settings)
FRAMES = {
    "max_age_s": SETTINGS.frame_max_age_s,
    "timeout_s": SETTINGS.frame_timeout_s,
}
app.add(Hub())  # the device connection: who is here, and their camera
llm = app.role("llm")
vlm = app.role("vlm")
voice = app.add(AnswerVoice(tts=app.role("tts")))
records = app.add(Records())
# What the monitors noticed, newest last; the conversation reads it.
history = ContextLog(
    "monitoring history",
    scope="participant",
    budget_bytes=SETTINGS.history_budget_bytes,
)

reader = InstrumentReader(
    vlm,
    load_device_map(HERE / SETTINGS.device_map_yaml),
    prompt=prompt("instrument_prompt"),
    keep_scans=SETTINGS.capture_marker_scans,
    **FRAMES,
)
instruments = app.add(
    InstrumentMonitor(
        reader,
        output=voice,
        records=records,
        history=history,
        every_s=SETTINGS.instrument_monitor_interval_s,
        state_every_s=SETTINGS.instrument_state_interval_s,
        lost_after_s=SETTINGS.instrument_lost_after_s,
    )
)
watcher = app.add(
    VisualMonitor(
        vlm,
        prompt=prompt("monitor_prompt"),
        records=records,
        history=history,
        every_s=SETTINGS.monitor_interval_s,
        keep=SETTINGS.monitor_history_size,
        **FRAMES,
    )
)

# -- the tools: upstream's names, descriptions and return_direct flags -------

CURRENT_VIEW_PROMPT = prompt("current_view_prompt")
NO_DESCRIPTION = (
    "Unable to inspect the current frame because the vision model returned "
    "no description."
)


@lab_tool(
    "current_view",
    "Inspect the glasses camera for the assistant's present visual field "
    "and any current or deictic visual request, including reading "
    "unspecified visible text. Call even when no object or referent is "
    "named, and report actual visible contents from the result. Never use "
    "for instrument readings.",
    return_direct=True,
)
async def current_view(p: Participant, turn: Turn) -> str | AsyncIterator[str]:
    frame = await p.frames.current(**FRAMES)
    if frame is None:
        return "No camera frame available — please try again."
    # Cosmos gets the user's own words; its answer streams to the user.
    return _or_no_description(
        vlm.stream(frame, turn.text, system=CURRENT_VIEW_PROMPT)
    )


async def _or_no_description(parts: AsyncIterator[str]) -> AsyncIterator[str]:
    """The stream as it comes; upstream's text if it brought nothing."""
    said = False
    async with contextlib.aclosing(parts):  # an answer cut off stops Cosmos
        async for part in parts:
            said = said or bool(part.strip())
            yield part
    if not said:
        yield NO_DESCRIPTION


@lab_tool(
    "lab_instruments__read",
    "Read current marker-labelled instruments, "
    "meters, gauges, readings, or numeric displays.",
    return_direct=True,
)
async def read_instruments(p: Participant) -> str:
    result = await reader.read(p)
    if result.readings:
        await history.publish(p, f"Read on request: {result.spoken()}")
    return result.spoken()


@lab_tool(
    "lab_instruments__start",
    "Start continuous marker-labelled instrument reading monitoring.",
    return_direct=True,
)
async def start_instruments(p: Participant) -> str:
    return instruments.start(p)


@lab_tool(
    "lab_instruments__stop",
    "The only route to stop continuous "
    "marker-labelled instrument reading monitoring.",
    return_direct=True,
)
async def stop_instruments(p: Participant) -> str:
    return await instruments.stop(p)


@lab_tool(
    "lab_instruments__status",
    "Report whether continuous "
    "marker-labelled instrument reading monitoring is running.",
    return_direct=True,
)
async def instruments_status(p: Participant) -> str:
    return instruments.status(p)


@lab_tool(
    "visual_monitor__start",
    "Start an ordinary background visual watch for the requested focus.",
    return_direct=True,
)
async def start_watch(
    p: Participant,
    instruction: Annotated[
        str,
        Field(
            max_length=240,
            description="Optional concise focus for the background monitor.",
        ),
    ] = "",
) -> str:
    return watcher.start(p, instruction)


@lab_tool(
    "visual_monitor__stop",
    "Stop, quit, cancel, or end the ordinary "
    "background visual watch. Never use for instruments.",
    return_direct=True,
)
async def stop_watch(p: Participant) -> str:
    return await watcher.stop(p)


@lab_tool(
    "visual_monitor__status",
    "Report whether the ordinary background visual watch is running.",
    return_direct=True,
)
async def watch_status(p: Participant) -> str:
    return watcher.status(p)


@lab_tool(
    "recent_visual_history", "Recent background visual observations or changes."
)
async def recent_visual_history(
    p: Participant,
    limit: Annotated[
        int,
        Field(
            ge=1,
            le=20,
            description="Maximum number of newest observations to return.",
        ),
    ] = 5,
) -> dict:
    # Not direct: the model words the answer from these.
    return {"observations": watcher.recent(p, limit)}


conversation = app.add(
    Conversation(
        role=llm,
        output=voice,
        instructions=prompt("foreground_prompt"),
        context=[history],
        tools=[
            current_view,
            recent_visual_history,
            start_watch,
            stop_watch,
            watch_status,
            read_instruments,
            start_instruments,
            stop_instruments,
            instruments_status,
        ],
        max_rounds=4,
        max_tokens=512,
        fallback="I couldn't complete that request. Please try again.",
    )
)


async def answer(turn: Turn, p: Participant) -> None:
    """One turn, then its foreground.jsonl record."""
    await conversation.answer(turn, p)
    record = ForegroundRecord(
        timestamp_us=turn.timestamp_us,
        query=turn.text,
        response=turn.state.get("answer") or "I don't have an answer.",
        tools=turn.state.get("tools", []),
    )
    await records.write(p, "foreground", record)


app.on_turn(answer)


def main() -> None:
    """`lab_instruments_app`: the launcher passes --config and --ready-file."""
    app.run()


if __name__ == "__main__":
    main()
