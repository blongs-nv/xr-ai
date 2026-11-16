# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain and LangGraph at the edges of an ``xr_ai_next`` app (optional package ``xr-ai-langchain``).

======================  ==============================================================================
``chat_model(role)``    a role as a LangChain ``BaseChatModel``: invoke, stream, tool calling
``langchain_tools``     our ``@tool`` functions as LangChain tools bound to one participant
``from_langchain``      a LangChain tool usable in ``Conversation(tools=[...])``
``messages``            our context channels as LangChain messages
``stream_text``         a LangGraph/LangChain message stream as text for ``p.reply``
``GraphBackend``        a compiled LangGraph graph as a ``Delegate`` back end
======================  ==============================================================================

The SDK (``xr_ai_next``) never imports this package.
"""

from __future__ import annotations

from ._chat import RoleChatModel, chat_model, to_chat_messages, to_langchain_messages
from ._graph import GraphBackend
from ._messages import messages, stream_text
from ._tools import from_langchain, langchain_tools

__all__ = [
    "GraphBackend",
    "RoleChatModel",
    "chat_model",
    "from_langchain",
    "langchain_tools",
    "messages",
    "stream_text",
    "to_chat_messages",
    "to_langchain_messages",
]
