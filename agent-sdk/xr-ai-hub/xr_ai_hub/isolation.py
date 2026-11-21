# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-scoped private hub routing for isolated execution."""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["DEFAULT_ENDPOINTS", "isolated_endpoints"]

DEFAULT_ENDPOINTS = ("ipc:///tmp/xr_hub_pub", "ipc:///tmp/xr_hub_in")
"""Standard hub addresses as ``(sub_addr, push_addr)``; isolated scopes redirect this pair."""

_ADDRESSES: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar("xr_ai_private_hub", default=None)


@contextmanager
def isolated_endpoints(*, sub_addr: str, push_addr: str) -> Iterator[None]:
    """Route standard hub endpoints created in this context to private IPC addresses.

    Explicit foreign addresses and file subscriptions are rejected before connection.
    Tasks inherit the scope; leaving it restores the caller's routing.
    """
    if any(not isinstance(value, str) or not value.startswith("ipc://") for value in (sub_addr, push_addr)):
        raise ValueError("isolated hub addresses must use ipc://")
    token = _ADDRESSES.set((sub_addr, push_addr))
    try:
        yield
    finally:
        _ADDRESSES.reset(token)


def _resolve(sub_addr: str, push_addr: str, file_sub_addr: str | None):
    private = _ADDRESSES.get()
    if private is None:
        return sub_addr, push_addr
    if file_sub_addr is not None or (sub_addr, push_addr) not in {
        private,
        DEFAULT_ENDPOINTS,
    }:
        raise ValueError("isolated worker endpoint must use only the private run hub addresses")
    return private
