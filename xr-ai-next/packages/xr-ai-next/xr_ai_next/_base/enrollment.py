# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connection, enrollment policies.

Three layers that used to be one: a *connection* (a device connected; published by the ``Hub`` component), an
*enrollment* (a participant exists in the app; an ``App`` setting that is one of the policies below) and the *session*
(frames and anything tied to the current connection; the participant). Components never see the policy, only the hooks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["Connection", "Enrollment", "PerConnection", "Persistent"]


@dataclass(frozen=True, slots=True)
class Connection:
    """One device connection to the hub: the client Identity field and the transport session."""

    identity: str
    """The client Identity field; in epic C it is also the participant id."""

    session_id: str
    """Identifier of this connection's session; a second device with the same identity has another one."""


@runtime_checkable
class Enrollment(Protocol):
    """An enrollment policy: when a connecting identity becomes a participant and how long it stays.

    An ``App`` setting, not a component; components never see it. Both members are read-only.
    """

    @property
    def admit(self) -> Callable[[Connection], bool] | None:
        """Rejects a connection when it returns false; ``None`` admits everyone."""
        ...

    @property
    def keep_for_s(self) -> float | None:
        """Seconds a participant stays enrolled after its last disconnect; ``None``: until the app stops."""
        ...


class PerConnection:
    """Enroll on the first connection of an identity, un-enroll when its last connection ends (``keep_for_s == 0``)."""

    def __init__(self, *, admit: Callable[[Connection], bool] | None) -> None:
        self._admit = admit

    @property
    def admit(self) -> Callable[[Connection], bool] | None:
        """The admission function."""
        return self._admit

    @property
    def keep_for_s(self) -> float | None:
        """Always ``0.0``."""
        return 0.0

    def __repr__(self) -> str:
        return f"PerConnection(admit={self._admit!r})"


class Persistent:
    """Enroll on first connection and keep the participant across disconnects for ``keep_for_s`` seconds."""

    def __init__(self, *, keep_for_s: float | None, admit: Callable[[Connection], bool] | None) -> None:
        if keep_for_s is not None and keep_for_s < 0:
            raise ValueError(f"keep_for_s must be None or >= 0, got {keep_for_s!r}")
        self._keep_for_s = keep_for_s
        self._admit = admit

    @property
    def admit(self) -> Callable[[Connection], bool] | None:
        """The admission function."""
        return self._admit

    @property
    def keep_for_s(self) -> float | None:
        """Seconds kept after the last disconnect; ``None``: until the app stops."""
        return self._keep_for_s

    def __repr__(self) -> str:
        return f"Persistent(keep_for_s={self._keep_for_s!r}, admit={self._admit!r})"
