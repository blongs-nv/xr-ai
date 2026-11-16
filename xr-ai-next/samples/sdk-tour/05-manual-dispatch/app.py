# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual dispatch: routing as code you write, not a component.

SHAPE: NO Router component. The turn handler is ordinary code: it asks the
decision role which option fits (`decide.decide(...)`), looks the answer up
in a dict, then calls it. That makes these easy:
  - a two-level menu (first the area, then the action inside it), so no
    single decision has more than a handful of options;
  - code rules before any model call (a keyword, a state, a permission);
  - doing two things for one turn;
  - stepping through it in a debugger.
What the Router component would have done for you, you now do yourself: the
"one moment" acknowledgment, retrying after a pass, and the fallback. Each is
a few lines below. The decision role's call is in the trace, as every role
call is.

  turn -> code rules -> decide(area) -> decide(action) -> call it
"""

from pathlib import Path

from xr_ai_kit import Conversation, OneShotDestination
from xr_ai_next import (
    App,
    Component,
    Hub,
    Participant,
    PerConnection,
    Turn,
    Voice,
)

HERE = Path(__file__).resolve().parent

app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
app.add(Hub())  # the device connection: who is here
decide = app.role("decide")
answer = app.role("answer_lightweight")
voice = app.add(Voice(summarize=answer))


class Lights(Component):
    """A pretend smart-home area with three actions, each a destination."""

    def __init__(self) -> None:
        super().__init__()
        self.on = OneShotDestination("on", "Turn lights on.", self._on)
        self.off = OneShotDestination("off", "Turn lights off.", self._off)
        self.dim = OneShotDestination(
            "dim", "Dim or brighten the lights.", self._dim
        )

    async def _on(self, turn: Turn, p: Participant) -> None:
        await p.reply("Lights on.", to=voice)

    async def _off(self, turn: Turn, p: Participant) -> None:
        await p.reply("Lights off.", to=voice)

    async def _dim(self, turn: Turn, p: Participant) -> None:
        await p.reply("Dimmed.", to=voice)


class Music(Component):
    def __init__(self) -> None:
        super().__init__()
        self.play = OneShotDestination(
            "play", "Play or resume music.", self._play
        )
        self.stop = OneShotDestination(
            "stop", "Stop or pause music.", self._stop
        )

    async def _play(self, turn: Turn, p: Participant) -> None:
        await p.reply("Playing.", to=voice)

    async def _stop(self, turn: Turn, p: Participant) -> None:
        await p.reply("Stopped.", to=voice)


lights = app.add(Lights())
music = app.add(Music())
conversation = app.add(
    Conversation(
        role=answer,
        output=voice,
        instructions="You are a home assistant. Be brief.",
        context=[voice.dialogue],
    )
)

# The areas, each a list of destinations. Area names and descriptions are
# what the first decision chooses between.
AREAS = {
    "lights": ("The lights in the room.", [lights.on, lights.off, lights.dim]),
    "music": ("Music playback.", [music.play, music.stop]),
    "chat": (
        "Anything else.",
        [OneShotDestination("conversation", "Chat.", conversation.answer)],
    ),
}
QUESTION = "Which option fits the user's latest message best?"


async def pick(options: dict[str, str], context: str) -> str:
    """One decision over `options`; no model call when there is only one."""
    if len(options) == 1:
        return next(iter(options))
    return str(await decide.decide(QUESTION, choices=options, context=context))


@app.on_turn
async def handle(turn: Turn, p: Participant) -> None:
    # 1. Code rules first: no model call for a bare "stop".
    if turn.text.strip().lower() in {"stop", "be quiet"}:
        await music.stop(turn, p)
        return

    # 2. Pick the area (a decision over three options), with the dialogue.
    latest = f"Latest message: {turn.text}"
    dialogue = await voice.dialogue.view(p, budget_bytes=4_000)
    area = await pick(
        {name: description for name, (description, _) in AREAS.items()},
        f"Recent dialogue:\n{dialogue}\n\n{latest}" if dialogue else latest,
    )
    # 3. Pick the action inside it. A destination describes itself.
    actions = {d.name: d for d in AREAS[area][1]}
    menu = {name: d.describe(p) for name, d in actions.items()}
    action = await pick(menu, latest)

    # 4. Run it. A destination is an async callable; a "pass" result means
    #    it was not for it, so fall back to the conversation ourselves.
    result = await actions[action](turn, p)
    if result is not None and result.passed:
        await conversation.answer(turn, p)


if __name__ == "__main__":
    app.run()
