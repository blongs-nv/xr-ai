# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot visual alert: "let me know when the sign says OPEN".

Clef first checks that the request explicitly starts or cancels an alert
(admission). The LLM turns the request, with the dialogue and the notes, into
one visible condition. A background task (`p.spawn`) then checks each new
camera frame with one Clef image decision and, when the condition holds,
says "Alert" (urgent) and Cosmos's description of that frame, once. A new
alert replaces the running one; the task ends when the participant leaves.
"""

import asyncio
import json
from pathlib import Path

from xr_ai_app import Component, Frame, Participant, Role, Sink, Turn, trace
from xr_ai_app.context import ContextChannel
from xr_ai_kit import Admission, RouterDestination

PROMPTS = Path(__file__).resolve().parent / "prompts"


def _prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8").strip()


ROUTING = (
    "Set a one-shot background visual alert ONLY when explicitly asked to "
    "notify the user when a specific visible condition occurs, including a "
    "condition they gave earlier or noted; cancel an existing alert on an "
    "explicit request. Visual questions, observations, recall, and fragments "
    "are not alert controls."
)
CONTROL = {
    "start": (
        "An explicit request to be alerted later when a specific visually "
        "observable condition is seen, stated directly or clearly referring "
        "to one the participant gave earlier."
    ),
    "cancel": "An explicit request to cancel or stop a visual alert.",
    "decline": (
        "Anything else, including current visual questions, observations, "
        "ambiguous fragments, capability questions, or an alert with neither "
        "a stated nor a referenced condition."
    ),
}
CONDITION = {
    "yes": (
        "The image clearly and directly establishes the requested visible "
        "condition."
    ),
    "no": (
        "The condition is absent, uncertain, obscured, or cannot be "
        "established from this single image."
    ),
}
ALERT = "Alert"
CANCELLED = "The visual alert is cancelled."
UNCLEAR = (
    "I couldn't tell what to watch for. Please say what should trigger the "
    "alert."
)
STOPPED = "I couldn't check the visual alert. It has stopped."
COMPILE_TIMEOUT_S = 30.0  # as upstream
COMPILE_TIMED_OUT = "I couldn't set up the new alert in time. Please try again."
KEY = "visual-alert"  # the task's key, and the source of what it says
_DESC = f"{KEY}:description"


class VisualAlert(Component):
    def __init__(
        self,
        *,
        decide: Role,
        llm: Role,
        vlm: Role,
        output: Sink,
        history: ContextChannel,
        saved: ContextChannel,
        check_every_s: float,
    ) -> None:
        super().__init__()
        self._decide = decide
        self._llm = llm
        self._vlm = vlm
        self._output = output
        self._history = history
        self._saved = saved
        self._every_s = check_every_s
        # Clef checks the request in the admission check, before the alert
        # runs; the description says when an alert is active.
        self.alerts = RouterDestination(
            "alerts", self.routing, self._alert, admit=self._admit
        )

    def routing(self, p: Participant) -> str:
        active = " An alert is active now." if self.active(p) else ""
        return ROUTING + active

    def active(self, p: Participant) -> bool:
        task = p.state.get(KEY)
        return task is not None and not task.done()

    async def _alert(self, turn: Turn, p: Participant) -> None:
        if turn.state.alert_action == "cancel":  # set by _admit() below
            await self._retire(p)
            await p.reply(CANCELLED, to=self._output)
            return
        try:
            async with asyncio.timeout(COMPILE_TIMEOUT_S):
                condition = await self._compile(turn, p)
        except TimeoutError:
            await p.reply(COMPILE_TIMED_OUT, to=self._output)
            return
        if condition is None:  # a running alert is left alone
            await p.reply(UNCLEAR, to=self._output)
            return
        await self._retire(p)
        await p.reply(f"I'll alert you once when {condition}.", to=self._output)
        # Detached: not part of this turn, so it outlives it and speaks on
        # the assistant's own initiative.
        watch = self._watch(p, condition)
        p.state[KEY] = p.spawn(watch, key=KEY, supersede=True, detached=True)

    async def _admit(self, turn: Turn, p: Participant) -> Admission:
        state = {"request": turn.text, "alert_active": self.active(p)}
        action = await self._decide.decide(
            _prompt("alert_control"), choices=CONTROL, context=json.dumps(state)
        )
        if action == "decline":
            return Admission("decline", reason="not an alert request")
        turn.state.alert_action = str(action)
        return Admission("accept")

    async def _compile(self, turn: Turn, p: Participant) -> str | None:
        """The request as one visible condition, or None when unclear."""
        sections = [f"Alert request:\n{turn.text}"]
        if history := await self._history.view(p):
            sections.append(f"Recent exchanges, oldest first:\n{history}")
        if saved := await self._saved.view(p):
            sections.append(f"Saved context:\n{saved}")
        # With reasoning (the profile's default), as upstream: without it
        # the omni model often answers "unclear" once history is present.
        reply = await self._llm.ask(
            None, "\n\n".join(sections), system=_prompt("alert_compile")
        )
        return _condition(str(reply))

    async def _watch(self, p: Participant, condition: str) -> None:
        try:
            async for frame in p.frames.every(self._every_s):
                if await self._holds(frame, condition):
                    break
            else:
                return  # the participant left
            await p.say(ALERT, to=self._output, urgent=True, source=KEY)
            text = await self._vlm.ask(
                frame, condition, system=_prompt("alert_description")
            )
            if str(text).strip():
                await p.say(str(text).strip(), to=self._output, source=_DESC)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - say so, and stop
            trace.emit("error", KEY, where="watch", error=type(exc).__name__)
            await p.say(STOPPED, to=self._output, urgent=True, source=KEY)

    async def _holds(self, frame: Frame, condition: str) -> bool:
        state = {"decision_path": "fast-clef", "condition": condition}
        verdict = await self._decide.decide(
            _prompt("alert_condition"),
            choices=CONDITION,
            context=json.dumps(state),
            frames=[frame],
        )
        return verdict == "yes"

    async def _retire(self, p: Participant) -> None:
        """Stop the running alert and drop what it has not said yet."""
        await p.cancel(KEY)
        await p.withdraw(KEY, to=self._output)
        await p.withdraw(_DESC, to=self._output)


def _condition(text: str) -> str | None:
    """The condition from the compile prompt's JSON reply, or None."""
    body = text.strip().removeprefix("```json").removeprefix("```")
    try:
        data = json.loads(body.removesuffix("```").strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or data.get("status") != "ready":
        return None
    condition = data.get("condition")
    if not isinstance(condition, str) or not condition.strip():
        return None
    # The prompt asks for a clause that completes "alert me when ..."; the
    # omni model often repeats those words.
    condition = condition.strip().rstrip(".")
    if condition.lower().startswith(LEAD):
        condition = condition[len(LEAD) :].strip()
    return condition or None


LEAD = "alert me when "
