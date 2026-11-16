# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Voice``: speech out (a :class:`~xr_ai_app.output.Sink`) and the dialogue log.

Speech itself (speech to text, the voice gate, text to speech, barge-in) is the app's existing voice pipeline,
configured by ``app.yaml``'s ``voice:`` section and the profile's ``speech_in`` and ``speech_out`` roles; this
component is its face in ``xr_ai_app``. It speaks through the participant's default output and keeps
``voice.dialogue``, a :class:`~xr_ai_app.ContextLog` of what the user said and what was spoken, which is the only
writer of that log.
"""

from __future__ import annotations

from collections.abc import AsyncIterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._base.component import Component
from .context import ContextLog
from .output import mark_first, note_output

if TYPE_CHECKING:
    from ._base.participant import Participant, Turn
    from ._base.profiles import Role


class Voice(Component):
    """Speech out, and the dialogue log (``voice.dialogue``).

    ``summarize`` compacts the dialogue when it outgrows ``dialogue_budget_bytes``. ``stt`` and ``tts`` are the
    speech roles (``app.role(...)``; default: the profile's ``speech_in`` and ``speech_out``, else its ``stt`` and
    ``tts`` entries), ``gate`` the voice-gate file (relative to ``app.yaml``; default: ``voice.voice_gate_yaml``,
    else ``voice_gate.yaml`` beside it), ``text_input`` whether typed messages from the device count as turns.
    """

    def __init__(
        self,
        *,
        summarize: Role | None = None,
        dialogue_budget_bytes: int = 8_000,
        stt: Role | None = None,
        tts: Role | None = None,
        gate: str | Path | None = None,
        text_input: bool = True,
        name: str = "voice",
    ) -> None:
        super().__init__()
        self._name = name
        self._speech: dict[str, Any] = {"stt": stt, "tts": tts, "gate": gate, "text_input": text_input}
        self.dialogue = ContextLog(
            "dialogue", scope="participant", budget_bytes=dialogue_budget_bytes, summarize=summarize
        )

    @property
    def name(self) -> str:
        return self._name

    async def _observe_turn(self, turn: Turn, p: Participant) -> None:
        await self.dialogue.publish(p, f"user: {turn.text}")

    async def say(self, p: Participant, text: str, *, urgent: bool = False, source: str | None = None) -> None:
        note_output(self._name, "say", text)
        await p.say(text, urgent=urgent, source=source)
        await self.dialogue.publish(p, f"assistant: {text}")

    async def reply(self, p: Participant, text: str | AsyncIterable[str]) -> str:
        if isinstance(text, str):
            if text:
                note_output(self._name, "reply", text)
            spoken = await p.reply(text)
        else:
            spoken = await p.reply(mark_first(text, lambda: note_output(self._name, "reply", "")))
            if spoken:
                note_output(self._name, "reply_end", spoken)
        if spoken.strip():
            await self.dialogue.publish(p, f"assistant: {spoken}")
        return spoken

    async def withdraw(self, p: Participant, source: str) -> bool:
        return await p.withdraw(source)


__all__ = ["Voice"]
