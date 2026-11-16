# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the typed SystemOne decision client and model config."""

from __future__ import annotations

import json
from types import MappingProxyType

import httpx
import pytest
from xr_ai_models import (
    Capabilities,
    DecisionAnswer,
    DecisionInputTooLargeError,
    DecisionQuestion,
    DecisionSpec,
    _openai_compat,
    _systemone,
    load_models_config_from_dict,
    make_decision,
)


def _config():
    return load_models_config_from_dict(
        {
            "clef": {
                "category": "decision",
                "kind": "systemone",
                "model_name": "Cloudflare/Clef",
                "base_url": "http://clef.test:8003",
                "api_key_env": "CLEF_TEST_TOKEN",
                "timeout": 9.0,
                "health_path": "/health",
            }
        }
    )


@pytest.mark.parametrize("enabled,key", [(True, None), (True, "test-token"), (False, "test-token")])
async def test_decision_health_auth_and_disabled_probe(monkeypatch, enabled, key):
    if key is None:
        monkeypatch.delenv("CLEF_TEST_TOKEN", raising=False)
    else:
        monkeypatch.setenv("CLEF_TEST_TOKEN", key)
    seen = []

    def handle(request):
        seen.append(request)
        assert str(request.url) == "https://clef.test/health"
        assert request.headers.get("authorization") == (f"Bearer {key}" if key else None)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False) as http:
        service = _systemone._SystemOneDecision(
            base_url="https://clef.test", model_name="clef", api_key_env="CLEF_TEST_TOKEN",
            timeout=9, health_check=enabled, health_path="/health", client=http,
        )
        assert await service.health()
        await service.close()
        assert not http.is_closed
    assert len(seen) == int(enabled)


@pytest.mark.asyncio
async def test_make_decision_resolves_config_and_validates_systemone_wire_contract(monkeypatch) -> None:
    config = _config()
    assert isinstance(config.decision("clef"), DecisionSpec)
    assert config.decision("clef").model_name == "Cloudflare/Clef"
    monkeypatch.setenv("CLEF_TEST_TOKEN", "test-token")
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/health":
            assert request.url.host == "clef.test"
            assert request.headers["authorization"] == "Bearer test-token"
            return httpx.Response(200)
        assert request.url.path == "/v1/systemone"
        assert request.headers["authorization"] == "Bearer test-token"
        return httpx.Response(
            200,
            json={
                "model": "Cloudflare/Clef",
                "answers": {
                    "scope": {
                        "type": "choice",
                        "choice": "electronics",
                        "confidence": 0.8125,
                        "probabilities": {"general": 0.1875, "electronics": 0.8125},
                    }
                },
                "usage": {"input_tokens": 27, "output_tokens": 0},
            },
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        _systemone.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    service = make_decision(config, "clef")
    try:
        assert await service.health()
        result = await service.decide(
            MappingProxyType({"request": "Which domain owns this?"}),
            {
                "scope": DecisionQuestion(
                    instructions="Select the best destination.",
                    criteria={
                        "general": "Ordinary broad conversation.",
                        "electronics": "A focused question about consumer electronics.",
                    },
                )
            },
            timeout=1.5,
        )
    finally:
        await service.close()

    assert result.answers["scope"].choice == "electronics"
    assert result.answers["scope"].confidence == 0.8125
    assert result.answers["scope"].probabilities == {"general": 0.1875, "electronics": 0.8125}
    assert result.model == "Cloudflare/Clef"
    assert result.usage == {"input_tokens": 27, "output_tokens": 0}
    request = next(item for item in seen if item.url.path == "/v1/systemone")
    assert request.read() == (
        b'{"model":"Cloudflare/Clef","state":{"request":"Which domain owns this?"},'
        b'"questions":{"scope":{"type":"choice","instructions":"Select the best destination.",'
        b'"criteria":{"general":"Ordinary broad conversation.",'
        b'"electronics":"A focused question about consumer electronics."}}}}'
    )
    assert request.extensions["timeout"]["read"] == 1.5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        {"type": "choice", "choice": "unknown", "confidence": 0.5, "probabilities": {"yes": 0.5, "no": 0.5}},
        {"type": "choice", "choice": "yes", "confidence": 0.5, "probabilities": {"yes": 0.5}},
        {"type": "choice", "choice": "yes", "confidence": 0.5, "probabilities": {"yes": float("nan"), "no": 0.5}},
        {"type": "choice", "choice": "yes", "confidence": 0.7, "probabilities": {"yes": 0.7, "no": 0.1}},
        {"type": "score", "choice": "yes", "confidence": 0.5, "probabilities": {"yes": 0.5, "no": 0.5}},
    ],
)
async def test_decision_rejects_malformed_or_unexpected_answer_data(answer) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"model": "Cloudflare/Clef", "answers": {"q": answer}, "usage": {}},
            )
        )
    )
    service = _systemone._SystemOneDecision(
        base_url="http://clef.test:8003",
        model_name="Cloudflare/Clef",
        api_key_env=None,
        timeout=5,
        health_check=False,
        health_path="/health",
        client=client,
    )
    try:
        with pytest.raises(ValueError):
            await service.decide(
                "state",
                {"q": DecisionQuestion(instructions="choose", criteria={"yes": "yes", "no": "no"})},
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_decision_rejects_partial_question_set() -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"model": "Cloudflare/Clef", "answers": {}, "usage": {}},
            )
        )
    )
    service = _systemone._SystemOneDecision(
        base_url="http://clef.test:8003",
        model_name="Cloudflare/Clef",
        api_key_env=None,
        timeout=5,
        health_check=False,
        health_path="/health",
        client=client,
    )
    try:
        with pytest.raises(ValueError, match="exactly one answer"):
            await service.decide(
                "state",
                {"q": DecisionQuestion(instructions="choose", criteria={"yes": "yes", "no": "no"})},
            )
    finally:
        await client.aclose()


def test_systemone_kind_is_restricted_to_decision_category() -> None:
    with pytest.raises(ValueError, match="decision kind"):
        load_models_config_from_dict({"llm": {"category": "llm", "kind": "systemone", "base_url": "http://clef.test"}})
    with pytest.raises(ValueError, match="speech kind"):
        load_models_config_from_dict(
            {"decision": {"category": "decision", "kind": "riva_grpc", "base_url": "http://clef.test"}}
        )


@pytest.mark.asyncio
async def test_decision_images_use_inline_wire_data_and_preserve_order(tmp_path):
    import base64
    import json

    png = b"\x89PNG\r\n\x1a\nexample"
    jpeg = b"\xff\xd8\xffexample"
    path = tmp_path / "example.jpg"
    path.write_bytes(jpeg)
    png_url = "data:image/png;base64," + base64.b64encode(png).decode()
    jpeg_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={
            "model": "clef", "usage": {},
            "answers": {"q": {"type": "choice", "choice": "yes", "confidence": 1.,
                              "probabilities": {"yes": 1., "no": 0.}}},
        })

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        service = _systemone._SystemOneDecision(
            base_url="http://clef.test", model_name="clef", api_key_env=None,
            timeout=9, health_check=False, health_path="/health", client=client,
        )
        questions = {"q": DecisionQuestion(instructions="Choose", criteria={"yes": "Yes", "no": "No"})}
        await service.decide("target then references", questions, images=[png, path, png_url])
        await service.decide("text only", questions)
        for invalid in [["https://example.com/image.png"], ["/server/path.png"], png_url, png]:
            with pytest.raises((ValueError, TypeError)):
                await service.decide("invalid", questions, images=invalid)
    assert bodies[0]["images"] == [png_url, jpeg_url, png_url]
    assert "images" not in bodies[1]
    assert len(bodies) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(413, json={"detail": "request requires 5000 tokens; maximum is 4096"}), "maximum is 4096"),
        (httpx.Response(413, text="body too large"), "body too large"),
        (httpx.Response(413, json=["unexpected"]), r"\[\"unexpected\"\]"),
        (httpx.Response(413), "decision request is too large"),
    ],
    ids=["detail", "text", "json-list", "empty"],
)
async def test_decision_reports_input_too_large_as_a_typed_error(response, message) -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: response))
    service = _systemone._SystemOneDecision(
        base_url="http://clef.test:8003",
        model_name="Cloudflare/Clef",
        api_key_env=None,
        timeout=5,
        health_check=False,
        health_path="/health",
        client=client,
    )
    try:
        with pytest.raises(DecisionInputTooLargeError, match=message):
            await service.decide(
                "state",
                {"q": DecisionQuestion(instructions="choose", criteria={"yes": "yes", "no": "no"})},
            )
    finally:
        await client.aclose()


# ---- a decision entry backed by an LLM (category decision, an openai_compat LLM adapter) ----


def _llm_decision_config(**endpoint):
    return load_models_config_from_dict(
        {
            "router": {
                "category": "decision",
                "adapter": {"preset": "nemotron_omni", "default_extras": {"temperature": 0.6, "max_tokens": 4000}},
                "endpoint": {"base_url": "http://llm.test:8108", "timeout": 9.0, **endpoint},
            }
        }
    )


_ROUTE = {
    "route": DecisionQuestion(
        instructions="Pick the destination for the request.",
        criteria={"notes": "The user dictates a note.", "chat": "Anything else."},
    ),
    "urgent": DecisionQuestion(instructions="", criteria={"yes": "It cannot wait.", "no": "It can wait."}),
}


def _chat_reply(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "llm",
            "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 90, "completion_tokens": 9},
        },
    )


def test_a_decision_entry_may_use_an_llm_adapter_and_other_entries_parse_as_before() -> None:
    spec = _llm_decision_config().decision("router")
    assert isinstance(spec, DecisionSpec)
    assert (spec.kind, spec.model_name, spec.timeout) == ("openai_compat", "llm", 9.0)
    assert spec.base_url == "http://llm.test:8108"
    assert spec.adapter.capabilities["vision"] and spec.adapter.reasoning_field == "reasoning_content"
    assert _config().decision("clef").kind == "systemone"
    # Only an LLM preset may back a decision; any other preset/category mismatch is still an error.
    with pytest.raises(ValueError, match="category mismatch"):
        load_models_config_from_dict(
            {"x": {"category": "decision", "adapter": {"preset": "cosmos_vlm"}, "endpoint": {"base_url": "http://x"}}}
        )
    with pytest.raises(ValueError, match="category mismatch"):
        load_models_config_from_dict(
            {"x": {"category": "vlm", "adapter": {"preset": "nemotron_omni"}, "endpoint": {"base_url": "http://x"}}}
        )


@pytest.mark.asyncio
async def test_make_decision_asks_the_llm_for_one_label_per_question_under_a_schema(monkeypatch) -> None:
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/health":
            return httpx.Response(200)
        assert request.url.path == "/v1/chat/completions"
        return _chat_reply('{"route": "notes", "urgent": "no"}')

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        _openai_compat.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    service = make_decision(_llm_decision_config(), "router")
    try:
        assert await service.health()
        result = await service.decide(MappingProxyType({"utterance": "Note: buy milk."}), _ROUTE, timeout=1.5)
    finally:
        await service.close()

    assert result.answers["route"] == DecisionAnswer(
        choice="notes", confidence=1.0, probabilities={"notes": 1.0, "chat": 0.0}
    )
    assert result.answers["urgent"].choice == "no" and result.answers["urgent"].probabilities == {"yes": 0.0, "no": 1.0}
    assert result.model == "llm" and result.usage == {"prompt_tokens": 90, "completion_tokens": 9}
    request = next(item for item in seen if item.url.path == "/v1/chat/completions")
    body = json.loads(request.read())
    assert body["model"] == "llm" and body["temperature"] == 0.0 and body["max_tokens"] < 4000
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": ["notes", "chat"]},
            "urgent": {"type": "string", "enum": ["yes", "no"]},
        },
        "required": ["route", "urgent"],
        "additionalProperties": False,
    }
    system, user = body["messages"]
    assert system["role"] == "system"
    for text in ('"utterance": "Note: buy milk."', "Pick the destination", '"notes": The user dictates a note.'):
        assert text in user["content"]
    assert request.extensions["timeout"]["read"] == 1.5


def _llm_service(handle, *, vision: bool = True):
    from xr_ai_models._llm_decision import _LLMDecision, _SchemaChat

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    chat = _SchemaChat("http://llm.test", "llm", capabilities=Capabilities(vision=vision), client=client)
    return _LLMDecision(chat, model_name="llm", timeout=5.0), client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        '{"route": "calendar", "urgent": "no"}',  # not a label
        '{"route": "notes"}',  # a question left out
        '{"route": "notes", "urgent": "no", "extra": "yes"}',
        '{"route": ["notes"], "urgent": "no"}',
        "notes",  # not JSON
        '```json\n{"route": "notes", "urgent": "no"}\n```',  # not strictly JSON
    ],
)
async def test_an_llm_decision_with_an_invalid_or_partial_answer_raises(content) -> None:
    service, client = _llm_service(lambda _request: _chat_reply(content))
    try:
        with pytest.raises(ValueError):
            await service.decide("Note: buy milk.", _ROUTE)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_an_llm_decision_sends_images_as_image_parts_or_refuses_them_when_text_only(tmp_path) -> None:
    import base64

    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    sent: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read()))
        return _chat_reply('{"q": "yes"}')

    question = {"q": DecisionQuestion(instructions="Is the sign open?", criteria={"yes": "open", "no": "closed"})}
    service, client = _llm_service(handle)
    try:
        assert (await service.decide("frame 1", question, images=[png])).answers["q"].choice == "yes"
    finally:
        await client.aclose()
    image, text = sent[0]["messages"][1]["content"]
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    assert image == {"type": "image_url", "image_url": {"url": url}}
    assert text["type"] == "text" and "Is the sign open?" in text["text"]

    text_only, client = _llm_service(handle, vision=False)
    try:
        with pytest.raises(ValueError, match="text-only"):
            await text_only.decide("frame 1", question, images=[png])
        with pytest.raises(ValueError, match="JPEG/PNG"):
            await service.decide("frame 1", question, images=["https://example.test/a.png"])
    finally:
        await client.aclose()
    assert len(sent) == 1  # nothing sent for a refused image


# ---- prompt templates: the layout an LLM decision reads travels with the model's entry ----------------


def _templated(templates) -> DecisionSpec:
    return load_models_config_from_dict(
        {
            "router": {
                "category": "decision",
                "adapter": {"preset": "nemotron_omni", "templates": templates},
                "endpoint": {"base_url": "http://llm.test:8108"},
            }
        }
    ).decision("router")


def test_the_default_decision_template_renders_the_questions_then_the_state() -> None:
    from xr_ai_models._llm_decision import _labels, _prompt

    template = _llm_decision_config().decision("router").adapter.templates["decision"]
    system, user = _prompt(template, {"utterance": "hi"}, _ROUTE, _labels(_ROUTE))
    assert system.startswith("You make choice decisions.")
    assert user == (
        "Questions:\n\n"
        'Question id: "route"\nInstructions: Pick the destination for the request.\nLabels:\n'
        '- "notes": The user dictates a note.\n- "chat": Anything else.\n\n'
        'Question id: "urgent"\nInstructions: \nLabels:\n- "yes": It cannot wait.\n- "no": It can wait.\n\n'
        'State:\n{\n  "utterance": "hi"\n}'
    )


def test_an_entry_overrides_template_parts_and_keeps_the_rest() -> None:
    spec = _templated({"decision": {"system": "", "user": "State: {state}\n\n{questions}"}})
    parts = spec.adapter.templates["decision"]
    assert parts["system"] == "" and parts["user"] == "State: {state}\n\n{questions}"
    assert parts["label"] == '- "{label}": {criterion}'  # the default
    # A Clef entry and an llm entry render no template and take none.
    assert _config().decision("clef").adapter.templates == {}


@pytest.mark.parametrize(
    ("templates", "message"),
    [
        ({"decision": {"user": "{questions} {state} {mood}"}}, r"uses \{mood\}"),
        ({"decision": {"user": "{questions}"}}, r"\{state\} must appear exactly once"),
        ({"decision": {"system": "{state}"}}, r"\{state\} must appear exactly once"),
        ({"decision": {"label": "{label}: {criterion} {id}"}}, r"uses \{id\}"),
        ({"decision": {"footer": "x"}}, "is not a part"),
        ({"decision": {"user": "{state.__class__} {questions}"}}, "must be a plain name"),
        ({"decision": {"user": "{state!r} {questions}"}}, "must be a plain name"),
        ({"decision": {"user": 3}}, "must be text"),
        ({"summarize": {}}, "templates.summarize is not rendered"),
        ("decision", "must be a mapping"),
    ],
)
def test_a_bad_template_fails_when_the_configuration_loads(templates, message) -> None:
    with pytest.raises(ValueError, match=message):
        _templated(templates)


def test_an_entry_that_renders_no_template_refuses_one() -> None:
    with pytest.raises(ValueError, match="templates.decision is not rendered"):
        load_models_config_from_dict(
            {
                "x": {
                    "adapter": {"preset": "nemotron_omni", "templates": {"decision": {}}},
                    "endpoint": {"base_url": "http://x"},
                }
            }
        )


@pytest.mark.asyncio
async def test_make_decision_sends_the_entrys_template(monkeypatch) -> None:
    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read()))
        return _chat_reply('{"route": "chat", "urgent": "yes"}')

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        _openai_compat.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    config = load_models_config_from_dict(
        {
            "router": {
                "category": "decision",
                "adapter": {
                    "preset": "nemotron_omni",
                    "templates": {"decision": {"system": "", "user": "Situation: {state}\nAsk: {questions}"}},
                },
                "endpoint": {"base_url": "http://llm.test:8108"},
            }
        }
    )
    service = make_decision(config, "router")
    try:
        result = await service.decide("The kettle is on.", _ROUTE)
    finally:
        await service.close()
    assert result.answers["route"].choice == "chat"
    (body,) = seen
    (user,) = body["messages"]  # no system message
    assert user["role"] == "user" and user["content"].startswith("Situation: The kettle is on.\nAsk: Question id:")
