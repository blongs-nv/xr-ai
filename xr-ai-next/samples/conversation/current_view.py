# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current view: questions about what the camera shows now, asked of Cosmos.

`p.frames.current()` takes the newest frame when it is fresh; when the client
streams no video it asks the client for one picture (the web client's
on-demand camera mode). The answer streams to the voice output.
"""

from datetime import UTC, datetime

from xr_ai_kit import RouterDestination
from xr_ai_next import Component, Participant, Role, Sink, Turn

ROUTING = (
    "Identify or describe visible objects, read visible text, and report "
    "observable colors, counts, or spatial relations from the current camera "
    "frame."
)
NO_CAMERA = "I couldn't get a current camera frame. Please try again."
NO_EVIDENCE = "I couldn't inspect the current view right now. Please try again."


class CurrentView(Component):
    def __init__(
        self,
        vlm: Role,
        output: Sink,
        *,
        system: str,
        max_age_s: float,
        timeout_s: float,
    ) -> None:
        super().__init__()
        self._vlm = vlm
        self._output = output
        self._system = system
        self._max_age_s = max_age_s
        self._timeout_s = timeout_s
        self.observation = RouterDestination("observation", ROUTING, self._ask)

    async def _ask(self, turn: Turn, p: Participant) -> None:
        frame = await p.frames.current(
            max_age_s=self._max_age_s, timeout_s=self._timeout_s
        )
        if frame is None:
            await p.reply(NO_CAMERA, to=self._output)
            return
        question = f"Frame source time: {_minute(frame.timestamp_us)}.\n"
        question += turn.text
        stream = self._vlm.stream(frame, question, system=self._system)
        if not (await p.reply(stream, to=self._output)).strip():
            await p.reply(NO_EVIDENCE, to=self._output)


def _minute(timestamp_us: int) -> str:
    """The frame's time to the minute, as upstream labels it."""
    moment = datetime.fromtimestamp(timestamp_us / 1_000_000, UTC)
    return moment.strftime("%Y-%m-%d %H:%M UTC")
