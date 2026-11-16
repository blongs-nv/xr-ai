# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A LangChain agent as a router destination.

SHAPE: an EXISTING LangChain/LangGraph agent, dropped into the front end as
one destination next to the built-in conversation. The router sends "math
and units" turns to it; it answers by voice, streaming.

The adapters (optional module `xr_ai_kit.langchain`) used here:
  chat_model(role)        our role as a LangChain chat model (tool calling
                          and streaming included), so models.json decides
                          the model
  langchain_tools(p, ...) our `@tool` functions as LangChain tools, bound to
                          this participant (so `look` sees THEIR camera)
  messages(p, ...)        our context channels as LangChain messages
  stream_text(stream)     the agent's message stream as text to speak
And the other way round:
  from_langchain(tool)    a LangChain tool usable by our Conversation

  turn -> Router -> MathAgent.solve
       -> agent.astream({"messages": [...context, user turn]})
       -> stream_text(...) -> p.reply(..., to=voice)
"""

from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool as lc_tool
from langgraph.prebuilt import create_react_agent
from xr_ai_app import (
    App,
    Component,
    Hub,
    Participant,
    PerConnection,
    Turn,
    Voice,
    tool,
)
from xr_ai_kit import (
    Conversation,
    OneShotDestination,
    Router,
)
from xr_ai_kit.langchain import (
    chat_model,
    from_langchain,
    langchain_tools,
    messages,
    stream_text,
)

HERE = Path(__file__).resolve().parent

app = App.from_config(HERE / "app.yaml", enrollment=PerConnection(admit=None))
hub = app.add(Hub())
reasoning = app.role("answer_with_reasoning")
perceive = app.role("perceive")
voice = app.add(Voice(summarize=app.role("answer_lightweight")))


@lc_tool
def convert(value: float, unit_from: str, unit_to: str) -> str:
    """Convert a value between units (c, f, km, mi)."""
    table = {
        ("c", "f"): lambda v: v * 9 / 5 + 32,
        ("f", "c"): lambda v: (v - 32) * 5 / 9,
        ("km", "mi"): lambda v: v * 0.621371,
        ("mi", "km"): lambda v: v / 0.621371,
    }
    fn = table.get((unit_from.lower(), unit_to.lower()))
    return f"{fn(value):.2f} {unit_to}" if fn else "unknown units"


@tool
async def look(p: Participant, question: str) -> str:
    """Look through the user's camera and answer a question about it."""
    # A fresh streamed frame, or with the camera on demand one picture.
    frame = await p.frames.current()
    if frame is None:
        return "The camera is not streaming."
    return await perceive.ask(frame, question, max_tokens=96)


class MathAgent(Component):
    def __init__(self) -> None:
        super().__init__()
        self.llm = chat_model(reasoning)
        # What the router sees: a name, a description, the handler.
        self.solve = OneShotDestination(
            "math",
            "Arithmetic, unit conversion, and measuring what is seen.",
            self._solve,
        )

    async def _solve(self, turn: Turn, p: Participant) -> None:
        # Tools bound to this participant: built per turn, cheap.
        agent = create_react_agent(
            self.llm,
            tools=[convert, *langchain_tools(p, look)],
            prompt="You solve math and unit questions. Answer in one short "
            "spoken sentence.",
        )
        history = await messages(p, voice.dialogue)
        stream = agent.astream(
            {"messages": [*history, HumanMessage(turn.text)]},
            stream_mode="messages",
        )
        await p.reply(stream_text(stream), to=voice)


math = app.add(MathAgent())
conversation = app.add(
    Conversation(
        role=reasoning,
        output=voice,
        instructions="You are a helpful assistant. Be brief.",
        # A LangChain tool in OUR conversation: the reverse adapter.
        tools=[from_langchain(convert)],
        context=[voice.dialogue],
    )
)
chat = OneShotDestination(
    "conversation", "Chat and general questions.", conversation.answer
)
routing = app.add(
    Router(
        decide=app.role("decide"),
        destinations=[chat, math.solve],
        default=chat,
        history=voice.dialogue,
        output=voice,
    )
)
app.on_turn(routing.handle)  # every turn goes to the router first

if __name__ == "__main__":
    app.run()
