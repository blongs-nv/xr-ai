# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The raw model interface, for tool loops and adapters: what ``role.as_llm()`` returns and the messages it speaks.

Apps use roles (``role.ask``, ``role.decide``, ``role.stream``) and the kit's ``Conversation``. Code that drives a
model turn by turn (a tool loop such as ``xr_ai_kit.Conversation``, or an adapter for another framework) uses
``role.as_llm()``, an :class:`LLMService` (``chat``, ``stream_events``), with these message types. Wrap its calls in
``xr_ai_app.trace.quiet_role_calls()`` and ``xr_ai_app.clock.held()`` (or ``HeldStream``), as both of those do,
so they are traced once and held on a virtual clock.
"""

from __future__ import annotations

from xr_ai_models import ChatMessage, ChatResponse, ImagePart, LLMService, TextPart, ToolCall, ToolDef

__all__ = ["ChatMessage", "ChatResponse", "ImagePart", "LLMService", "TextPart", "ToolCall", "ToolDef"]
