# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The conversation with the person, as a step and the escalator beside it see it: ``environment.conversation``.

``await conversation.text()`` is what has been said so far, one line each (``user: ...``, ``assistant: ...``), oldest
first; ``async for _ in conversation.changes():`` runs now, then after each new line. Live it is the app's dialogue
(``SopRunner(conversation=voice.dialogue)``); in an evaluation, a case's fixed text (:class:`FixedConversationHistory`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class ConversationHistory(Protocol):
    """The conversation with the person; see the module docstring."""

    async def text(self) -> str:
        """What has been said so far, one line each, oldest first (``""`` before anything is said)."""
        ...

    def changes(self) -> AsyncIterator[None]:
        """``async for _ in conversation.changes():`` runs now, then after each new line."""
        ...


class FixedConversationHistory:
    """A conversation that never changes: a case's text in an evaluation, or none (``FixedConversationHistory()``)."""

    def __init__(self, text: str = "") -> None:
        self._text = text

    async def text(self) -> str:
        return self._text

    async def changes(self) -> AsyncIterator[None]:
        """Yields once, then ends: it never changes."""
        yield
