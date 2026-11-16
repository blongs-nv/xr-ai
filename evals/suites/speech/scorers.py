# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python scorers for the speech suite: word error rate and speaker gating.

Both are ``function(case, result) -> (score, evidence)`` for ``scorer.type: python``. Scores are
``1 - WER`` (clamped), so ``pass_threshold: 0.9`` means a word error rate of at most 10 percent.
"""

from __future__ import annotations

import json
import re


def normalize(text: str) -> list[str]:
    """Lower-case words without punctuation; apostrophes dropped (``let's`` -> ``lets``)."""
    return re.findall(r"\w+", text.casefold().replace("'", "").replace("’", ""))


def word_errors(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Word-level edit distance as ``(substitutions, deletions, insertions)``."""
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    cost = [[(0, 0, 0)] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0] = (0, i, 0)
    for j in range(1, cols):
        cost[0][j] = (0, 0, j)
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
                continue
            sub, dele, ins = cost[i - 1][j - 1], cost[i - 1][j], cost[i][j - 1]
            options = [
                (sum(sub) + 1, (sub[0] + 1, sub[1], sub[2])),
                (sum(dele) + 1, (dele[0], dele[1] + 1, dele[2])),
                (sum(ins) + 1, (ins[0], ins[1], ins[2] + 1)),
            ]
            cost[i][j] = min(options, key=lambda option: option[0])[1]
    return cost[-1][-1]


def _score_wer(reference_text: str, hypothesis_text: str) -> tuple[float, dict]:
    reference = normalize(reference_text)
    hypothesis = normalize(hypothesis_text)
    sub, dele, ins = word_errors(reference, hypothesis)
    rate = (sub + dele + ins) / len(reference) if reference else float(bool(hypothesis))
    return max(0.0, 1.0 - rate), {
        "wer": round(rate, 4),
        "substitutions": sub,
        "deletions": dele,
        "insertions": ins,
        "reference_words": len(reference),
    }


def wer(case, result) -> tuple[float, dict]:
    """Score ``1 - WER`` of the transcript against ``scorer.reference`` (empty reference: any word is an error)."""
    return _score_wer(case.scorer["reference"], result.text)


def gating(case, result) -> tuple[float, dict]:
    """Score a speaker-gating run summarised as JSON by ``drivers.speaker_gate``.

    The score is 0 when enrollment differs from ``expect_enrolled``, or when any ``forbid`` text (what a
    non-enrolled speaker said) reached the transcripts; otherwise ``1 - WER`` of the transcripts against
    ``expect_text`` (skipped when absent).
    """
    summary = json.loads(result.text)
    scorer = case.scorer
    evidence = {k: summary.get(k) for k in ("enrolled", "transcripts", "event_kinds")}
    if summary.get("enrolled") != scorer.get("expect_enrolled", True):
        return 0.0, {**evidence, "failure": "enrollment mismatch"}
    heard = " ".join(summary.get("transcripts", []))
    heard_words = " ".join(normalize(heard))
    leaked = [t for t in scorer.get("forbid", []) if normalize(t) and " ".join(normalize(t)) in heard_words]
    if leaked:
        return 0.0, {**evidence, "failure": "non-enrolled speech passed the gate", "leaked": leaked}
    if "expect_text" not in scorer:
        return 1.0, evidence
    score, detail = _score_wer(scorer["expect_text"], heard)
    return score, {**evidence, **detail}
