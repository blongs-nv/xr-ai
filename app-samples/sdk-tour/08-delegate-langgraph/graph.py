# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An ordinary LangGraph graph: nothing XR-AI specific in this file.

plan -> research (one model call per planned question) -> answer. A node can
report its own progress with LangGraph's stream writer; the delegate speaks
it (rate limited) and puts it in the conversation's context.
"""

from operator import add
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    request: str
    questions: list[str]
    notes: Annotated[list[str], add]
    answer: str


def build_graph(llm: BaseChatModel):
    async def plan(state: State) -> dict:
        text = await llm.ainvoke(
            "List up to three short questions to research, one per line, "
            f"for: {state['request']}"
        )
        questions = [q for q in text.content.splitlines() if q.strip()][:3]
        return {"questions": questions}

    async def research(state: State) -> dict:
        write = get_stream_writer()
        notes = []
        for question in state["questions"]:
            # Spoken to the user ("Still working on it: ..."): plain words.
            write({"progress": f"Looking into: {question}"})
            reply = await llm.ainvoke(f"Answer briefly: {question}")
            notes.append(f"{question}: {reply.content}")
        return {"notes": notes}

    async def answer(state: State) -> dict:
        notes = "\n".join(state["notes"])
        reply = await llm.ainvoke(
            f"Request: {state['request']}\nNotes:\n{notes}\n"
            "Write the answer in a few sentences."
        )
        return {"answer": reply.content}

    graph = StateGraph(State)
    graph.add_node("plan", plan)
    graph.add_node("research", research)
    graph.add_node("answer", answer)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "answer")
    graph.add_edge("answer", END)
    return graph.compile()
