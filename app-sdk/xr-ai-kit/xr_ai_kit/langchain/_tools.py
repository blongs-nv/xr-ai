# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tools both ways: our ``@tool`` functions as LangChain tools, and LangChain tools in our ``Conversation``."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel
from xr_ai_app.tools import Tool, ToolSpec, describe, render

__all__ = ["from_langchain", "langchain_tools"]


def langchain_tools(p: Any, *tools: Any) -> list[StructuredTool]:
    """``tools`` (``@tool`` functions, or ``xr_ai_tools.Tool`` objects) as LangChain tools bound to ``p``.

    The model sees the same schema our ``Conversation`` offers (``p`` and ``turn`` hidden). Each call runs the tool as
    a ``Conversation`` does (``xr_ai_app.tools``): ``p`` injected, ``turn`` the turn being answered (``None`` outside
    a turn), a ``tool`` trace span, the result rendered as text (a streamed result joined). A failing tool raises, as
    LangChain tools do; LangGraph's ``ToolNode`` decides what the model sees. ``return_direct`` carries over
    (LangChain's agents then end with the tool's result). The tools are async only (``ainvoke``).
    """
    specs = [describe(t) for t in tools]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate tool names: {names}")
    return [_structured(spec, p) for spec in specs]


def _structured(spec: ToolSpec, p: Any) -> StructuredTool:
    async def run(**arguments: Any) -> str:
        return await spec.call(p, arguments)

    return StructuredTool.from_function(
        coroutine=run,
        name=spec.name,
        description=spec.description,
        args_schema=spec.arguments,
        return_direct=spec.return_direct,
    )


def from_langchain(tool: BaseTool) -> Tool[Any, Any]:
    """A LangChain tool as an ``xr_ai_tools.Tool``, for ``Conversation(tools=[...])``.

    The model sees the tool's ``tool_call_schema`` (its arguments minus any injected ones); a call runs
    ``await tool.ainvoke(arguments)`` (a sync tool runs in LangChain's executor) and the model reads its result as
    text (non-string results as JSON); its ``return_direct`` carries over (the result is the answer). A tool whose
    schema is a plain JSON-schema dict is refused with ``TypeError``.
    """
    if not isinstance(tool, BaseTool):
        raise TypeError(f"from_langchain expects a LangChain BaseTool, got {type(tool).__name__}")
    schema = tool.tool_call_schema
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError(f"tool {tool.name!r}: from_langchain needs a pydantic args schema, not {type(schema).__name__}")

    async def handler(request: BaseModel) -> str:
        value = await tool.ainvoke(request.model_dump(exclude_unset=True))
        return render(value)

    return Tool(tool.name, tool.description or tool.name, schema, None, handler, return_direct=tool.return_direct)
