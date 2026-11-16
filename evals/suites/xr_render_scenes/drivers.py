# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Call driver for the xr-render scenario cases: run the real scene supervisor over an in-memory scene.

The scene, tracking, text memory and vision tools are the original harness's fakes (``xr_render_demo_eval``, which
this suite needs importable); only the answer model is real. The driver returns a JSON summary that
``scorers.py:scenario`` checks with the harness's own ``check_corpus``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_EVAL_PROJECT = Path(__file__).resolve().parents[3] / "agent-samples" / "xr-render-demo" / "eval"


def harness() -> Any:
    """The original harness module; a missing project is a case error with the fix in the message."""
    try:
        from xr_render_demo_eval import harness as module  # noqa: PLC0415
    except ImportError:
        sys.path.insert(0, str(_EVAL_PROJECT))
        try:
            from xr_render_demo_eval import harness as module  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "xr_render_demo_eval is not importable: run from an environment with agent-samples/xr-render-demo/eval "
                f"installed ({exc})"
            ) from exc
    return module


def corpus_case(name: str) -> dict[str, Any]:
    """The original corpus entry called ``name``."""
    for case in harness().CORPUS_CASES:
        if case["name"] == name:
            return case
    raise ValueError(f"no corpus case named {name!r}")


def _models_config(endpoint: Any) -> Any:
    from xr_ai_models import load_models_config_from_dict  # noqa: PLC0415

    entry = {"category": "llm", "endpoint": {"base_url": endpoint.base_url}}
    for adapter in ({"preset": endpoint.preset}, {"kind": "openai_compat", "model_name": endpoint.model}):
        try:
            return load_models_config_from_dict({"models": {"llm": {"adapter": adapter, **entry}}})
        except ValueError:
            continue
    raise ValueError(f"endpoint preset {endpoint.preset!r} is not a known model preset")


async def scene_turn(endpoint: Any, case: Any) -> str:
    """Handle ``input.text`` as one turn on the scene named by ``params.case``; return the tool calls and the reply."""
    from xr_ai_models import make_llm  # noqa: PLC0415

    module = harness()
    corpus = corpus_case(case.params["case"])
    scene = module.FakeScene.from_corpus_case(corpus)
    llm = make_llm(_models_config(endpoint), "llm")
    try:
        supervisor = module._make_supervisor(llm, *scene.make_tools())  # noqa: SLF001
        if corpus.get("recent_moves"):
            supervisor._context.set_recent_moves(  # noqa: SLF001
                module._PARTICIPANT,  # noqa: SLF001
                [f"{i}: previously at {a}, now at {b}" for i, a, b in corpus["recent_moves"]],
            )
        request = module.SceneRequest(
            transcript=corpus["user"], participant_id=module._PARTICIPANT, timestamp_us=module.EVAL_REFERENCE_US  # noqa: SLF001
        )
        reply = (await supervisor.handle(request)).response
    finally:
        await llm.close()
    return json.dumps({"calls": scene.calls, "reply": reply})
