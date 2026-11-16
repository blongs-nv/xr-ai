# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""test_session and the lifecycle contract suite.

Against an app without the test seam the suite is an expected failure (``NotImplementedError``).
Against ``RefApp`` (a small correct implementation behind the same seam) it must pass, and each
deliberately broken variant must fail the rule it breaks, so the suite is known to bite.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from _reference_app import RefApp
from xr_ai_app import Component
from xr_ai_app._base.app import App, AppConfig
from xr_ai_app._base.testing import (
    LIFECYCLE_RULES,
    FakeModels,
    run_lifecycle_contract,
    run_lifecycle_rule,
    test_session,
)
from xr_ai_app._base.testing.contract import PENDING_RULES


@pytest.fixture(autouse=True)
def _agent_init(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """``Component.__init__`` is replaced by a no-op; the reference runs need a working one."""
    if "skeleton" not in request.node.name:
        monkeypatch.setattr(Component, "__init__", lambda self: None)


def _rules() -> list[object]:
    """Every rule; a pending placeholder (epic C) is an expected failure until its components exist."""
    return [
        pytest.param(r, marks=pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="pending components"))
        if r in PENDING_RULES
        else r
        for r in LIFECYCLE_RULES
    ]


def _skeleton_app() -> App:
    return App(AppConfig(config_dir=Path(".")))


@pytest.mark.parametrize("rule", _rules())
async def test_contract_rule_on_skeleton_app(rule: str) -> None:
    await run_lifecycle_rule(rule, _skeleton_app)


async def test_contract_suite_on_skeleton_app() -> None:
    await run_lifecycle_contract(_skeleton_app)


@pytest.mark.parametrize("rule", _rules())
async def test_contract_rule_passes_on_reference_app(rule: str) -> None:
    await run_lifecycle_rule(rule, RefApp)


async def test_contract_suite_passes_on_reference_app() -> None:
    await run_lifecycle_contract(RefApp)


_MUTANTS = {
    "no_reverse": "R2_unenroll_sequence",
    "admit_after_leave": "R1_task_ownership",
    "leaky_stop": "R3_stop_sequence",
    "swallow_errors": "R4_error_containment",
    "signals": "R7_only_run_installs_signal_handlers",
}


@pytest.mark.parametrize(("mutation", "rule"), sorted(_MUTANTS.items()))
async def test_contract_rule_fails_on_broken_app(mutation: str, rule: str) -> None:
    with pytest.raises(AssertionError, match=rule.split("_")[0]):
        await run_lifecycle_rule(rule, lambda: RefApp(mutations=frozenset({mutation})))


async def test_contract_suite_reports_every_failing_rule() -> None:
    with pytest.raises(AssertionError) as info:
        await run_lifecycle_contract(lambda: RefApp(mutations=frozenset({"no_reverse", "swallow_errors"})))
    message = str(info.value)
    assert "R2_unenroll_sequence" in message and "R4_error_containment" in message
    assert "R7_only" not in message


async def test_r2_abandon_check_warns_when_drain_is_long() -> None:
    with pytest.warns(UserWarning, match=r"R2\.6 skipped"):
        await run_lifecycle_rule("R2_unenroll_sequence", lambda: RefApp(drain_s=5.0))


async def test_session_attaches_fakes_starts_and_stops_the_app() -> None:
    app = RefApp()
    models = FakeModels({"answer": ["hi"]})
    async with test_session(app, models=models) as s:
        assert s.app is app and s.models is models and not s.hub.closed
        await s.join("a")
        assert s.hub.participants == {"a"}
        await s.settle()
        assert await app.role("answer").ask(None, "q") == "hi"
    assert s.hub.closed
    assert not app.participants


async def test_session_without_the_app_seam_says_what_is_missing() -> None:
    with pytest.raises(NotImplementedError, match="_install_test_doubles"):
        async with test_session(_skeleton_app_stub()):
            pass


def _skeleton_app_stub() -> App:
    return object()  # type: ignore[return-value]


async def test_session_settle_waits_for_in_flight_model_calls() -> None:
    import asyncio

    app = RefApp()
    models = FakeModels({"answer": ["slow"]})
    models.set_latency("answer", 0.05)
    async with test_session(app, models=models) as s:
        task = asyncio.create_task(models.role("answer").ask(None, "q"))
        await models.wait_started("answer")
        await s.settle()
        assert task.done()


def test_r8_is_registered_and_pending() -> None:
    assert "R8_composition" in LIFECYCLE_RULES
    assert PENDING_RULES == {"R8_composition"}


async def test_contract_suite_skips_pending_rules_silently() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await run_lifecycle_contract(RefApp)
    assert not [w for w in caught if "R8_composition" in str(w.message)]
    with pytest.raises(NotImplementedError, match="R8_composition"):
        await run_lifecycle_rule("R8_composition", RefApp)
