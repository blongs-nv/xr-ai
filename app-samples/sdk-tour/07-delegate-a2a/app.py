# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delegate over A2A: a fast voice front end, a slow agent behind it.

SHAPE: FRONT END + BACK END. The front end stays simple and real time: it
talks, looks, and keeps short context. Anything long (research, planning,
booking, code) is DELEGATED to another agent that may run for minutes. Here
that agent speaks A2A (the Agent2Agent protocol, https://a2a-protocol.org),
so any A2A agent can sit behind it, in any language or framework;
research_agent/ is a stand-in one.

  "find me a quiet cafe near the station"
     -> conversation calls the `start_research` tool -> research.start(...)
     -> the conversation answers at once ("I'm on it")
  ...the agent works on an A2A task...
     its status messages -> research.status (context) + spoken, at most
                            every 20 s
     its result artifact -> research.results (context) + one spoken sentence
  "how's it going?"  -> the conversation reads research.status
  "never mind"       -> `cancel_research` -> research.cancel(p), which sends
                        A2A tasks/cancel to the agent

The delegate owns the job's lifecycle (start, updates, cancel, failure), so
the front end never blocks on it and the conversation always knows what is
running. The agent never speaks: it reports, and the front end decides what
to say.

On DeviceIOHub (`uv run delegate_a2a`) the agent starts with the stack;
for a scenario run, start research_agent/ first.
"""

from pathlib import Path

from pydantic import BaseModel
from xr_ai_app import App, Hub, Participant, PerConnection, Voice, tool
from xr_ai_kit import Conversation, Delegate
from xr_ai_kit.delegate import A2ABackend

HERE = Path(__file__).resolve().parent


class Settings(BaseModel):
    agent_url: str = "http://127.0.0.1:8830"  # the A2A agent
    progress_every_s: float = 20.0


app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
app.add(Hub())  # the device connection: who is here
SETTINGS = app.settings(Settings)
lightweight = app.role("answer_lightweight")
voice = app.add(Voice(summarize=lightweight))

research = app.add(
    Delegate(
        "research",
        backend=A2ABackend(SETTINGS.agent_url),
        output=voice,  # where progress and results are spoken
        progress_every_s=SETTINGS.progress_every_s,
        summarize=lightweight,  # speaks a result the back end did not word
        max_running=1,  # per participant; a new job replaces the old one
    )
)


@tool
async def start_research(p: Participant, request: str) -> str:
    """Start a long research task in the background. `request` is the
    user's request, complete and self-contained."""
    job = await research.start(p, request)
    return f"Started (job {job.id}). Tell the user you'll report back."


@tool
async def cancel_research(p: Participant) -> str:
    """Cancel the running research task."""
    cancelled = await research.cancel(p)
    return "Cancelled." if cancelled else "Nothing was running."


conversation = app.add(
    Conversation(
        role=app.role("answer_with_reasoning"),
        output=voice,
        instructions="You are a voice assistant. For anything that needs "
        "searching, comparing or planning, start research and say you will "
        "report back. Use the research sections to answer about it.",
        tools=[start_research, cancel_research],
        # research.status: what is running and its latest progress.
        # research.results: finished results, newest last.
        context=[voice.dialogue, research.status, research.results],
    )
)
app.on_turn(conversation.answer)

if __name__ == "__main__":
    app.run()
