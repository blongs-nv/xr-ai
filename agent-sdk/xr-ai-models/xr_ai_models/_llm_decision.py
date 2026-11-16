# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choice decisions answered by an OpenAI-compatible chat model.

A ``category: decision`` entry with an LLM adapter (``openai_compat``, e.g.
``"adapter": {"preset": "nemotron_omni"}``) serves the same
:class:`DecisionService` contract as the SystemOne adapter, for a deployment
without room for a decision model. Each ``decide`` is one chat request for all
questions: the questions share one state (and images), so one prefill reads
them, and the reply is one JSON object constrained by a schema
(``response_format``) whose property per question is an enum of its labels.
The prompt is the entry's ``decision`` template (``adapter.templates``; see
``_templates``), by default the questions, then the state.
Reasoning is off and the temperature is 0, whatever the entry's defaults. The
reply is checked strictly (exactly the question ids, each a permitted label);
anything else raises, as the protocol returns no partial decisions. An LLM
reports no calibrated probabilities: the chosen label gets 1.0 and every other
label 0.0.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from ._openai_compat import (
    OpenAICompatLLM,
    _normalize_image,
    _parse_chat_response,
    _request_headers,
)
from ._protocols import (
    ChatMessage,
    ChatResponse,
    ContentPart,
    DecisionAnswer,
    DecisionQuestion,
    DecisionResponse,
    ImageInput,
    ImagePart,
    TextPart,
)
from ._templates import DECISION_DEFAULTS, render_decision


class _SchemaChat(OpenAICompatLLM):
    """The LLM client, plus one non-streamed request with a JSON schema."""

    async def chat_json(
        self,
        messages: Sequence[ChatMessage],
        schema: Mapping[str, Any],
        *,
        max_tokens: int,
        timeout: float,
    ) -> ChatResponse:
        payload = self._build_payload(
            messages,
            tools=None,
            max_tokens=None,
            temperature=None,
            enable_thinking=False,
            thinking_budget=None,
            stream=False,
        )
        # After the entry's default extras: a decision is greedy and short.
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0.0
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "decision", "strict": True, "schema": dict(schema)},
        }
        response = await self._client.post(
            self._chat_url,
            json=payload,
            headers=_request_headers(self._api_key, None),
            timeout=timeout,
        )
        if response.is_error:
            logger.error("decision llm {} {}: {}", self._model, response.status_code, response.text[:300])
        response.raise_for_status()
        return _parse_chat_response(response.json(), self._reasoning_field)


class _LLMDecision:
    """An LLM behind the public :class:`DecisionService` protocol."""

    def __init__(
        self,
        chat: _SchemaChat,
        *,
        model_name: str,
        timeout: float,
        template: Mapping[str, str] = DECISION_DEFAULTS,
    ) -> None:
        self._chat = chat
        self.model_name = model_name
        self.timeout = timeout
        self._template = dict(template)

    async def decide(
        self,
        state: str | Mapping[str, Any],
        questions: Mapping[str, DecisionQuestion],
        *,
        images: Sequence[ImageInput] = (),
        timeout: float | None = None,
    ) -> DecisionResponse:
        """Ask the LLM for one label per question; validate every answer before returning."""

        if not isinstance(state, (str, Mapping)):
            raise TypeError("decision state must be text or a mapping")
        labels = _labels(questions)
        if isinstance(images, (str, bytes, bytearray, memoryview)):
            raise TypeError("images must be a sequence of image inputs")
        system, text = _prompt(self._template, state, questions, labels)
        content: list[ContentPart] = []
        if images:
            if not self._chat.capabilities.vision:
                raise ValueError(
                    f"decision model {self.model_name!r} is text-only (its adapter declares no vision); "
                    "it cannot decide on images"
                )
            urls = [_normalize_image(image) for image in images]
            if any(not url.startswith(("data:image/jpeg;base64,", "data:image/png;base64,")) for url in urls):
                raise ValueError("decision images must be inline JPEG/PNG data URLs, bytes, or pathlib.Path objects")
            content = [*(ImagePart(url=url) for url in urls), TextPart(text=text)]
        messages = [ChatMessage(role="system", content=system)] if system else []
        messages.append(ChatMessage(role="user", content=content or text))
        response = await self._chat.chat_json(
            messages,
            _schema(labels),
            max_tokens=64 + 2 * sum(len(qid) + max(map(len, choices)) for qid, choices in labels.items()),
            timeout=self._bounded_timeout(timeout),
        )
        choices = _parse(response.content, labels)
        answers = {
            qid: DecisionAnswer(
                choice=choice,
                confidence=1.0,
                probabilities={label: 1.0 if label == choice else 0.0 for label in labels[qid]},
            )
            for qid, choice in choices.items()
        }
        raw = response.raw if isinstance(response.raw, Mapping) else {}
        usage = raw.get("usage")
        return DecisionResponse(
            answers=answers,
            model=str(raw.get("model") or self.model_name),
            usage=dict(usage) if isinstance(usage, Mapping) else {},
            raw=raw,
        )

    def _bounded_timeout(self, timeout: float | None) -> float:
        if timeout is None:
            return self.timeout
        if isinstance(timeout, bool):
            raise ValueError("decision timeout must be a positive finite number")
        try:
            requested = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("decision timeout must be a positive finite number") from exc
        if not math.isfinite(requested) or requested <= 0:
            raise ValueError("decision timeout must be a positive finite number")
        return min(requested, self.timeout)

    async def health(self) -> bool:
        """Probe the LLM endpoint as an ``llm`` entry's client does."""

        return await self._chat.health()

    async def close(self) -> None:
        """Close the internally created HTTP client, if the adapter owns it."""

        await self._chat.close()

    async def __aenter__(self) -> _LLMDecision:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


def _labels(questions: Mapping[str, DecisionQuestion]) -> dict[str, list[str]]:
    """The permitted labels per question id, after the SystemOne adapter's checks."""

    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("at least one decision question is required")
    labels: dict[str, list[str]] = {}
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("decision question IDs must be non-empty strings")
        if not isinstance(question, DecisionQuestion):
            raise TypeError(f"question {question_id!r} must be a DecisionQuestion")
        if not isinstance(question.instructions, str):
            raise TypeError(f"question {question_id!r} instructions must be text")
        if not isinstance(question.criteria, Mapping) or not question.criteria:
            raise ValueError(f"question {question_id!r} criteria must be a non-empty mapping")
        for label, criterion in question.criteria.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"question {question_id!r} choice labels must be non-empty strings")
            if not isinstance(criterion, str) or not criterion.strip():
                raise ValueError(f"question {question_id!r} criteria must be non-empty strings")
        if len(question.criteria) < 2:
            raise ValueError(f"question {question_id!r} must define at least two choices")
        labels[question_id] = list(question.criteria)
    return labels


def _prompt(
    template: Mapping[str, str],
    state: str | Mapping[str, Any],
    questions: Mapping[str, DecisionQuestion],
    labels: Mapping[str, list[str]],
) -> tuple[str, str]:
    """The ``(system, user)`` texts, rendered with the entry's ``decision`` template (``_templates``)."""

    state_text = state if isinstance(state, str) else json.dumps(dict(state), ensure_ascii=False, indent=2)
    rendered = [
        (qid, question.instructions.strip(), [(label, question.criteria[label]) for label in labels[qid]])
        for qid, question in questions.items()
    ]
    return render_decision(template, state_text, rendered)


def _schema(labels: Mapping[str, list[str]]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {qid: {"type": "string", "enum": list(choices)} for qid, choices in labels.items()},
        "required": list(labels),
        "additionalProperties": False,
    }


def _parse(content: str, labels: Mapping[str, list[str]]) -> dict[str, str]:
    """Exactly one permitted label per question id, or ``ValueError``."""

    try:
        reply = json.loads(content.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"decision LLM reply is not a JSON object: {content[:200]!r}") from exc
    if not isinstance(reply, dict) or reply.keys() != labels.keys():
        raise ValueError("decision LLM reply must contain exactly one answer for every question")
    for question_id, choice in reply.items():
        if not isinstance(choice, str) or choice not in labels[question_id]:
            raise ValueError(f"decision LLM answer {question_id!r} selected an unknown choice: {choice!r}")
    return reply
