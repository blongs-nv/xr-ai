# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delegate to LangGraph: the same front end, a LangGraph back end.

SHAPE: FRONT END + BACK END, with the back end a LangGraph graph running in
the same process. Identical to 07-delegate-a2a except for one line: the
delegate's `backend`. `GraphBackend` (from the optional `xr-ai-langchain`
package) runs the graph as a job:

  research.start(p, request)
    -> graph.astream({"request": request}, stream_mode=[..., "custom"])
    -> each custom event          -> progress, in the node's own words
                                     ("Looking into ...", spoken)
       (a node that finishes says nothing: node names stay internal)
    -> the final state's `answer` -> result
  research.cancel(p) -> the graph's task is cancelled

The graph's language model is our `answer_with_reasoning` role, wrapped as a
LangChain chat model (`chat_model`), so models.json still decides which model
runs. Any LangChain chat model works there too.
"""

from pathlib import Path

from graph import build_graph
from xr_ai_kit import Conversation, Delegate
from xr_ai_langchain import GraphBackend, chat_model
from xr_ai_next import App, Hub, Participant, PerConnection, Voice, tool

HERE = Path(__file__).resolve().parent

app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
app.add(Hub())  # the device connection: who is here
reasoning = app.role("answer_with_reasoning")
lightweight = app.role("answer_lightweight")
voice = app.add(Voice(summarize=lightweight))

research = app.add(
    Delegate(
        "research",
        backend=GraphBackend(
            build_graph(chat_model(reasoning)),
            input=lambda request: {"request": request, "notes": []},
            result=lambda state: state["answer"],
        ),
        output=voice,
        summarize=lightweight,
        progress_every_s=15.0,
    )
)


@tool
async def start_research(p: Participant, request: str) -> str:
    """Start a long research task in the background."""
    job = await research.start(p, request)
    return f"Started (job {job.id}). Tell the user you'll report back."


@tool
async def cancel_research(p: Participant) -> str:
    """Cancel the running research task."""
    return "Cancelled." if await research.cancel(p) else "Nothing running."


conversation = app.add(
    Conversation(
        role=reasoning,
        output=voice,
        instructions="You are a voice assistant. Start research for anything "
        "that needs searching or planning, and say you will report back.",
        tools=[start_research, cancel_research],
        context=[voice.dialogue, research.status, research.results],
    )
)
app.on_turn(conversation.answer)

if __name__ == "__main__":
    app.run()
