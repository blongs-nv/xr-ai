# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test kit: ``FakeHub``, scripted ``FakeModels``, ``test_session`` and the lifecycle contract suite.

``TestSession`` and ``test_session`` start with ``test``; import the module, not the names, into
pytest modules, or set ``__test__ = False`` on the alias.
"""

from __future__ import annotations

from ._hub import FakeHub, HubEvent, HubListener
from ._models import FakeCall, FakeModels, ScriptExhausted, ToolCall
from ._session import TestSession, test_session
from .contract import LIFECYCLE_RULES, run_lifecycle_contract, run_lifecycle_rule

__all__ = [
    "LIFECYCLE_RULES",
    "FakeCall",
    "FakeHub",
    "FakeModels",
    "HubEvent",
    "HubListener",
    "ScriptExhausted",
    "TestSession",
    "ToolCall",
    "run_lifecycle_contract",
    "run_lifecycle_rule",
    "test_session",
]
