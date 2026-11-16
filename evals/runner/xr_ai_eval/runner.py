# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a suite against a profile and produce result records."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable
from typing import Any

import httpx

from xr_ai_eval.cases import Case, Suite, Turn
from xr_ai_eval.client import ChatClient, ChatResult, build_messages
from xr_ai_eval.hooks import call_hook, load_hook
from xr_ai_eval.metrics import GpuMemoryReader
from xr_ai_eval.profiles import Endpoint, Profile, ProfileError
from xr_ai_eval.schema import Record
from xr_ai_eval.scorers import score_output


def select_cases(
    suite: Suite, tags: Iterable[str] = (), roles: Iterable[str] = ()
) -> list[Case]:
    """Cases that are not skipped and match any given tag and any given role."""
    wanted_tags, wanted_roles = set(tags), set(roles)
    return [
        case
        for case in suite.cases
        if case.skip_reason is None
        and (not wanted_tags or wanted_tags & set(case.tags))
        and (not wanted_roles or case.role in wanted_roles)
    ]


def utc_now() -> str:
    """Current UTC time, ISO 8601."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


async def run_suite(
    suite: Suite,
    profile: Profile,
    client: ChatClient,
    *,
    commit: str,
    run_id: str,
    tags: Iterable[str] = (),
    roles: Iterable[str] = (),
    gpu_memory: GpuMemoryReader | None = None,
    now: Callable[[], str] = utc_now,
) -> list[Record]:
    """Run every selected case once and return one record each, errors included."""
    records = []
    for case in select_cases(suite, tags, roles):
        records.append(await _run_case(suite, case, profile, client, commit, run_id, gpu_memory, now))
    return records


async def _run_case(
    suite: Suite,
    case: Case,
    profile: Profile,
    client: ChatClient,
    commit: str,
    run_id: str,
    gpu_memory: GpuMemoryReader | None,
    now: Callable[[], str],
) -> Record:
    record = Record(
        suite=suite.name,
        case=case.id,
        role=case.role,
        preset="unknown",
        profile=profile.name,
        passed=False,
        score=0.0,
        latency_ms=0.0,
        tokens={"input": None, "output": None},
        gpu_profile=profile.gpu_profile,
        commit=commit,
        run_id=run_id,
        timestamp=now(),
    )
    try:
        endpoint = profile.endpoint(case.role)
        record.preset = endpoint.preset
        result = await _call(case, endpoint, client)
        record.latency_ms = round(result.latency_ms, 3)
        record.tokens = {"input": result.input_tokens, "output": result.output_tokens}
        if gpu_memory is not None:
            record.gpu_memory_mb = gpu_memory()

        async def judge(role: str, prompt: str) -> tuple[str, str]:
            judge_endpoint = profile.endpoint(role)
            reply = await client.chat(judge_endpoint, build_messages(prompt, (), ()))
            return reply.text, judge_endpoint.preset

        score, detail = await score_output(case.scorer, result.text, judge, case=case, result=result)
        record.score = round(score, 6)
        record.passed = score >= case.pass_threshold
        record.detail = detail
    except (ProfileError, httpx.HTTPError, ValueError, RuntimeError) as exc:
        record.passed, record.score = False, 0.0
        record.error = f"{type(exc).__name__}: {exc}"
    return record


async def _call(case: Case, endpoint: Endpoint, client: ChatClient) -> ChatResult:
    """Make the case's model call: a driver, a transcription, or one or more chat turns."""
    if case.driver is not None:
        reply = await call_hook(load_hook(case.base_dir, case.driver), endpoint, case)
        if isinstance(reply, ChatResult):
            return reply
        if not isinstance(reply, str):
            raise RuntimeError(f"driver must return text, got {type(reply).__name__}")
        return ChatResult(reply, 0.0, None, None, (reply,))
    if case.audio is not None:
        result = await client.transcribe(endpoint, case.audio)
        return ChatResult(result.text, result.latency_ms, None, None, (result.text,))
    turns = case.turns or (Turn(case.text, case.frames),)
    history: list[dict[str, Any]] = [{"role": "system", "content": case.system}] if case.system else []
    history += [{"role": m["role"], "content": m["text"]} for m in case.context]
    replies: list[str] = []
    latency, tokens_in, tokens_out = 0.0, [], []
    for turn in turns:
        history.append(build_messages(turn.text, turn.frames, ())[0])
        result = await client.chat(endpoint, history)
        history.append({"role": "assistant", "content": result.text})
        replies.append(result.text)
        latency += result.latency_ms
        tokens_in.append(result.input_tokens)
        tokens_out.append(result.output_tokens)
    return ChatResult(replies[-1], latency, _total(tokens_in), _total(tokens_out), tuple(replies))


def _total(counts: list[int | None]) -> int | None:
    """Sum of token counts, ``None`` when any turn did not report usage."""
    return None if any(c is None for c in counts) else sum(c for c in counts if c is not None)
