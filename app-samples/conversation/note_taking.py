# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Note-taking: a destination that holds the focus until the user finishes.

Clef classifies each request the router sends here (admission): a question
asked during the session is handed back to the router, which answers it
elsewhere; the session stays open. Every note goes to the `notes` context log,
which the conversation reads, and to a full list for reading back.
"""

import json
from pathlib import Path

from xr_ai_app import Component, ContextLog, Participant, Role, Sink, Turn
from xr_ai_kit import Admission, RouterDestination

PROMPTS = Path(__file__).resolve().parent / "prompts"
CONTROL = (PROMPTS / "note_control.txt").read_text(encoding="utf-8").strip()
# Static, so the routing request stays the same size however many notes.
ROUTING = (
    "A participant-scoped note-taking destination. It starts only when "
    "explicitly requested, captures user-authored notes, and stays focused "
    "until the participant finishes, including across a question answered by "
    "another destination. It does not provide general knowledge or inspect "
    "the current camera."
)
FOCUSED = {
    "capture": (
        "A statement supplied as content during focused notes, or explicitly "
        "dictated words to save. The participant is not asking for an answer "
        "or asking the assistant to perform the stated task."
    ),
    "readback": "The participant wants to review notes already captured.",
    "finish": (
        "The participant clearly wants to end note-taking and consume this "
        "request."
    ),
    "handoff": (
        "The participant asks the assistant for information, an explanation, "
        "advice, or an action, rather than asking to record those words. A "
        "question seeking an answer is handoff even while focused."
    ),
}
# Upstream also offers "finish" here and declines it; five choices.
UNFOCUSED = {
    "start": "The participant explicitly wants to begin a note-taking session.",
    "capture": FOCUSED["capture"],
    "readback": FOCUSED["readback"],
    "handoff": FOCUSED["handoff"],
    "other": (
        "The current request is not a note action or an explicit request to "
        "save content."
    ),
}
# Without focus these start (or read) notes; anything else is declined.
OPENERS = ("start", "capture", "readback")
NO_NOTES = "There are no notes in this session yet."


class NoteTaking(Component):
    def __init__(self, decide: Role, output: Sink, log: ContextLog) -> None:
        super().__init__()
        self._decide = decide
        self._output = output
        self._log = log
        # What the router sees: Clef classifies in the admission check; the
        # router asks holds_focus before every turn.
        self.notes = RouterDestination(
            "notes",
            ROUTING,
            self._take,
            admit=self._admit,
            focused=self.holds_focus,
        )

    async def _take(self, turn: Turn, p: Participant) -> None:
        action = turn.state.note_action  # set by _admit() below
        if action in ("capture", "start"):
            p.state.notes_focused = True
        if action == "capture":
            note = turn.text.strip()
            p.state.setdefault("notes", []).append(note)
            await self._log.publish(p, note)
            text = "Noted."
        elif action == "start":
            text = "I’m ready to take notes."
        elif action == "finish":
            p.state.notes_focused = False
            text = "Note-taking is finished."
        else:  # readback: every note, word for word
            notes = p.state.get("notes")
            text = "Your notes: " + "; ".join(notes) if notes else NO_NOTES
        await p.reply(text, to=self._output)

    async def _admit(self, turn: Turn, p: Participant) -> Admission:
        focused = self.holds_focus(p)
        state = {
            "request": turn.text,
            "destination": {"name": "notes", "focused": focused},
        }
        action = await self._decide.decide(
            CONTROL,
            choices=FOCUSED if focused else UNFOCUSED,
            context=json.dumps(state),
        )
        if action == "handoff" or (not focused and action not in OPENERS):
            return Admission("decline", reason="not a note action")
        turn.state.note_action = str(action)
        return Admission("accept")

    # The router asks this before every turn: while it holds, every turn
    # comes here first, with no routing model call.
    def holds_focus(self, p: Participant) -> bool:
        return bool(p.state.get("notes_focused"))
