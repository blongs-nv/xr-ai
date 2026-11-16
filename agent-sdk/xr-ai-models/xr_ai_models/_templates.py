# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt templates for prompts an adapter writes itself.

An adapter that turns structured input into model text (today: a decision
answered by an LLM) renders it with named templates. A model entry overrides
them under ``adapter.templates``, part by part; parts it leaves out keep their
defaults, so the right layout for a model travels with the model's entry:

.. code-block:: json

   "adapter": {
     "preset": "nemotron_omni",
     "templates": {"decision": {"user": "State:\\n{state}\\n\\nQuestions:\\n\\n{questions}"}}
   }

Placeholders are :meth:`str.format` fields by plain name (``{{`` and ``}}`` are
literal braces); each part accepts only its own names, and every template is
checked when the configuration loads.

``decision`` parts:

- ``system``: the system message; ``""`` sends none. May use ``{state}`` and
  ``{questions}``.
- ``user``: the user message. ``{state}`` and ``{questions}`` must each appear
  exactly once across ``system`` and ``user``.
- ``question``: one question, with ``{id}``, ``{instructions}`` and
  ``{labels}``; questions are joined by a blank line into ``{questions}``.
- ``label``: one permitted label, with ``{label}`` and ``{criterion}``; labels
  are joined by newlines into ``{labels}``.

The JSON schema that limits answers to the labels, and the strict reading of
the reply, are not templated: a template changes what the model reads, never
what ``decide`` returns.
"""

from __future__ import annotations

import string
from collections.abc import Mapping, Sequence
from typing import Any

DECISION = "decision"

DECISION_DEFAULTS: Mapping[str, str] = {
    "system": (
        "You make choice decisions. Read the state, then answer every question "
        "with exactly one of its labels: the label whose criterion fits the "
        "state best. Reply with one JSON object that maps each question id to "
        "its label, and nothing else."
    ),
    # The questions come before the state: with reasoning off, the omni model picked the right alert-admission label
    # for 10 of 10 probe requests this way and 5 of 10 with the state first.
    "user": "Questions:\n\n{questions}\n\nState:\n{state}",
    "question": 'Question id: "{id}"\nInstructions: {instructions}\nLabels:\n{labels}',
    "label": '- "{label}": {criterion}',
}

_DECISION_FIELDS: Mapping[str, frozenset[str]] = {
    "system": frozenset({"state", "questions"}),
    "user": frozenset({"state", "questions"}),
    "question": frozenset({"id", "instructions", "labels"}),
    "label": frozenset({"label", "criterion"}),
}

_KNOWN: Mapping[str, tuple[Mapping[str, str], Mapping[str, frozenset[str]]]] = {
    DECISION: (DECISION_DEFAULTS, _DECISION_FIELDS),
}


def parse_templates(value: Any, *, allowed: Sequence[str]) -> dict[str, dict[str, str]]:
    """Check an entry's ``templates`` mapping; return it with every part filled in.

    ``allowed`` names the templates this entry's adapter renders. Raises
    :class:`ValueError` for an unknown template or part, a part that is not
    text, a placeholder the part does not accept, or a ``decision`` template
    without exactly one ``{state}`` and one ``{questions}``.
    """

    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("templates must be a mapping of template name to parts")
    for name in value:
        if name not in allowed:
            used = ", ".join(allowed) or "none"
            raise ValueError(f"templates.{name} is not rendered by this entry's adapter (it renders: {used})")
    resolved: dict[str, dict[str, str]] = {}
    for name in allowed:
        defaults, fields = _KNOWN[name]
        overrides = value.get(name, {})
        if not isinstance(overrides, Mapping):
            raise ValueError(f"templates.{name} must be a mapping of part name to text")
        parts = dict(defaults)
        for part, text in overrides.items():
            if part not in fields:
                raise ValueError(f"templates.{name}.{part} is not a part (parts: {', '.join(fields)})")
            if not isinstance(text, str):
                raise ValueError(f"templates.{name}.{part} must be text")
            parts[part] = text
        counts: dict[str, int] = {}
        for part, text in parts.items():
            for field in _fields(text, where=f"templates.{name}.{part}"):
                if field not in fields[part]:
                    accepted = ", ".join("{" + f + "}" for f in sorted(fields[part]))
                    raise ValueError(f"templates.{name}.{part} uses {{{field}}}; it accepts {accepted}")
                if part in ("system", "user"):
                    counts[field] = counts.get(field, 0) + 1
        if name == DECISION:
            for field in ("state", "questions"):
                if counts.get(field, 0) != 1:
                    raise ValueError(f"templates.{name}: {{{field}}} must appear exactly once across system and user")
        resolved[name] = parts
    return resolved


def render_decision(
    parts: Mapping[str, str],
    state: str,
    questions: Sequence[tuple[str, str, Sequence[tuple[str, str]]]],
) -> tuple[str, str]:
    """The ``(system, user)`` texts for ``questions`` (id, instructions, labels with criteria) and ``state``."""

    blocks = [
        parts["question"].format(
            id=question_id,
            instructions=instructions,
            labels="\n".join(parts["label"].format(label=label, criterion=criterion) for label, criterion in labels),
        )
        for question_id, instructions, labels in questions
    ]
    values = {"state": state, "questions": "\n\n".join(blocks)}
    return parts["system"].format(**values), parts["user"].format(**values)


def _fields(text: str, *, where: str) -> list[str]:
    """The placeholder names in ``text``: plain names only (no attribute, index, conversion or format spec)."""

    names = []
    try:
        parsed = list(string.Formatter().parse(text))
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc
    for _literal, field, spec, conversion in parsed:
        if field is None:
            continue
        if not field.isidentifier() or spec or conversion:
            raise ValueError(f"{where}: placeholder {{{field}}} must be a plain name")
        names.append(field)
    return names
