# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agents and their exposed native tools."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

from xr_ai_tools import AsyncTool, Tool


class Agent:
    """An object that owns state and exposes ordinary native tools."""

    def __init__(
        self,
        tools: Iterable[Tool[Any, Any] | AsyncTool[Any, Any]] = (),
    ) -> None:
        owned = tuple(tools)
        if owned:
            warnings.warn(
                "Agent(tools=...) is deprecated: the runtime only type-checks it; hand tools to the conversation "
                "(xr_ai_next.Conversation(tools=...)) or declare them with @xr_ai_next.tool",
                DeprecationWarning,
                stacklevel=2,
            )
        names: set[str] = set()
        for tool in owned:
            if not isinstance(tool, (Tool, AsyncTool)):
                raise TypeError("agents may expose only Tool or AsyncTool instances")
            if tool.name in names:
                raise ValueError(f"duplicate agent tool name: {tool.name}")
            names.add(tool.name)
        self._tools = owned

    @property
    def tools(self) -> tuple[Tool[Any, Any] | AsyncTool[Any, Any], ...]:
        """Return the existing native tools exposed by this agent."""

        return self._tools

__all__ = ["Agent"]
