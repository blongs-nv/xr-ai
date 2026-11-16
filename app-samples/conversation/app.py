# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversation: talk, ask about the view, take notes, set a visual alert.

SHAPE: the kit's showcase. A Router asks Clef (`decide`) where each turn goes:

  conversation  the kit's Conversation on the omni LLM; it reads the dialogue
                and the notes (the default when nothing else fits)
  observation   the current camera frame (or one picture taken on demand)
                and the question, asked of Cosmos     (current_view.py)
  alerts        a one-shot visual alert: a background task that checks each
                new frame and speaks once             (visual_alert.py)
  notes         a note-taking session that holds the focus until finished;
                questions asked meanwhile go back to the router
                                                      (note_taking.py)

A destination's admission check (one Clef call) can decline a turn; the router
then picks again without it. Roles are the entry names of yaml/models.json
(upstream conversation-example's): llm, router, vlm, tts. The SDK waits for
those servers before it reports ready, and shows "processing" per turn.
"""

from pathlib import Path

from current_view import CurrentView
from note_taking import NoteTaking
from pydantic import BaseModel
from visual_alert import VisualAlert
from xr_ai_app import App, ContextLog, Hub, PerConnection, Voice
from xr_ai_kit import Ack, Conversation, OneShotDestination, Router

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"


class Settings(BaseModel):
    frame_max_age_s: float = 5.0
    frame_timeout_s: float = 5.0
    dialogue_budget_bytes: int = 6_000
    notes_budget_bytes: int = 6_000
    routing_history_budget_bytes: int = 8_000
    ack_delay_s: float | None = None  # None: no acknowledgment
    alert_check_interval_s: float = 1.0


def prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text(encoding="utf-8").strip()


app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
SETTINGS = app.settings(Settings)
app.add(Hub())  # the device connection: who is here, and their camera
llm = app.role("llm")
router = app.role("router")
vlm = app.role("vlm")
voice = app.add(
    Voice(
        tts=app.role("tts"),
        summarize=llm,  # folds old dialogue into a summary
        dialogue_budget_bytes=SETTINGS.dialogue_budget_bytes,
    )
)
# Participant-authored notes, as the conversation reads them.
notes = ContextLog(
    "notes",
    scope="participant",
    budget_bytes=SETTINGS.notes_budget_bytes,
    summarize=llm,
)

conversation = app.add(
    Conversation(
        role=llm,
        output=voice,
        instructions=prompt("conversation"),
        context=[voice.dialogue, notes],
    )
)
chat = OneShotDestination(
    "conversation",
    "Provide text assistance across subjects and recall relevant "
    "conversation history and participant-authored notes. If "
    "answering depends on what the camera shows now, use direct "
    "observation even when prior turns contain the same or a similar "
    "question and answer. This destination does not inspect the "
    "camera.",
    conversation.answer,
)
view = app.add(
    CurrentView(
        vlm,
        voice,
        system=prompt("system"),
        max_age_s=SETTINGS.frame_max_age_s,
        timeout_s=SETTINGS.frame_timeout_s,
    )
)
alert = app.add(
    VisualAlert(
        decide=router,
        llm=llm,
        vlm=vlm,
        output=voice,
        history=voice.dialogue,
        saved=notes,
        check_every_s=SETTINGS.alert_check_interval_s,
    )
)
note_taking = app.add(NoteTaking(router, voice, notes))

ack = None
if SETTINGS.ack_delay_s is not None:
    ack = Ack(role=llm, delay_s=SETTINGS.ack_delay_s)
routing = app.add(
    Router(
        decide=router,
        destinations=[
            chat,
            view.observation,
            alert.alerts,
            note_taking.notes,
        ],
        default=chat,
        history=voice.dialogue,
        history_budget_bytes=SETTINGS.routing_history_budget_bytes,
        output=voice,
        ack=ack,
    )
)
app.on_turn(routing.handle)  # every turn goes to the router first


def main() -> None:
    """`conversation_app`: the launcher passes --config and --ready-file."""
    app.run()


if __name__ == "__main__":
    main()
