# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``role.ask(response_format=...)`` asks the server for that schema when the endpoint declares json_schema."""

from typing import Any

import pytest
from pydantic import BaseModel
from xr_ai_app._base.profiles import Role
from xr_ai_app.testing import solid_frame
from xr_ai_models import Capabilities, ChatResponse


class Reading(BaseModel):
    value: float | None = None
    unit: str | None = None


class _Client:
    def __init__(self, json_schema: bool) -> None:
        self.capabilities = Capabilities(vision=True, json_schema=json_schema)
        self.options: list[dict[str, Any]] = []

    async def ask_image(self, image: Any, question: str, **options: Any) -> ChatResponse:
        self.options.append(options)
        return ChatResponse('{"value": 176, "unit": "F"}', None, None, "stop", {})

    async def chat(self, messages: Any, **options: Any) -> ChatResponse:
        self.options.append(options)
        return ChatResponse('{"value": 80}', None, None, "stop", {})


@pytest.mark.parametrize("declared", [True, False])
async def test_the_schema_goes_to_the_server_only_when_the_endpoint_declares_it(declared: bool) -> None:
    vlm_client, llm_client = _Client(declared), _Client(declared)
    perceive = Role.bind("perceive", entry="vlm", category="vlm", client=vlm_client)
    answer = Role.bind("answer", entry="llm", category="llm", client=llm_client)

    assert await perceive.ask(solid_frame(), "Read the display.", response_format=Reading) == Reading(
        value=176, unit="F"
    )
    assert await answer.ask(None, "Recommend.", response_format=Reading) == Reading(value=80)
    for client in (vlm_client, llm_client):
        sent = client.options[0].get("response_schema")
        assert sent == (Reading.model_json_schema() if declared else None)


async def test_a_plain_ask_never_sends_a_schema() -> None:
    client = _Client(True)
    perceive = Role.bind("perceive", entry="vlm", category="vlm", client=client)
    await perceive.ask(solid_frame(), "What is this?")
    assert "response_schema" not in client.options[0]
