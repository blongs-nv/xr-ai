# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Names out of a package's ``__all__`` still import, with one DeprecationWarning naming the replacement."""

from __future__ import annotations

import importlib
import warnings

import pytest

PACKAGES = (
    "xr_ai_gpu",
    "xr_ai_hub",
    "xr_ai_interaction",
    "xr_ai_logging",
    "xr_ai_models",
    "xr_ai_vad",
)
CASES = [(pkg, name) for pkg in PACKAGES for name in sorted(importlib.import_module(pkg)._DEPRECATED)]


@pytest.mark.parametrize(("package", "name"), CASES)
def test_a_deprecated_name_is_out_of_all_but_resolves_with_one_warning(package: str, name: str) -> None:
    module = importlib.import_module(package)
    assert name not in module.__all__
    assert name in dir(module)
    home, note = module._DEPRECATED[name]
    target = getattr(importlib.import_module(home, package), name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert getattr(module, name) is target
    assert [w.category for w in caught] == [DeprecationWarning]
    assert str(caught[0].message) == f"{package}.{name} is deprecated: {note}"
    assert caught[0].filename == __file__


def test_from_import_warns_once_and_a_plain_import_of_the_package_does_not_warn() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from xr_ai_hub import ParticipantEvent  # noqa: F401
    assert len(caught) == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from xr_ai_hub import ProcessorEndpoint  # noqa: F401
    assert caught == []


def test_a_name_that_never_existed_is_still_an_attribute_error() -> None:
    import xr_ai_hub

    with pytest.raises(AttributeError, match="no_such_name"):
        xr_ai_hub.no_such_name  # noqa: B018
    assert not hasattr(xr_ai_hub, "no_such_name")


def test_star_import_skips_deprecated_names_without_warning() -> None:
    namespace: dict[str, object] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec("from xr_ai_hub import *", namespace)  # noqa: S102
    assert caught == []
    assert "ParticipantEvent" not in namespace and "ProcessorEndpoint" in namespace


# --- duplicates of focus area 1 that stay importable and warn --------------------------------------------------------


def test_the_flat_llm_spec_constructor_warns_but_the_structured_one_does_not() -> None:
    from xr_ai_models import AdapterSpec, EndpointSpec, LLMSpec

    with pytest.warns(DeprecationWarning, match="LLMSpec's flat constructor"):
        flat = LLMSpec(base_url="http://x:1/v1", model_name="m")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        structured = LLMSpec(adapter=AdapterSpec(model_name="m"), endpoint=EndpointSpec(base_url="http://x:1/v1"))
        LLMSpec()
    assert flat.adapter.model_name == structured.adapter.model_name == "m"
    assert flat.endpoint.base_url == structured.endpoint.base_url


def test_agent_tools_argument_warns_only_when_given() -> None:
    from xr_ai_runtime import Agent

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Agent().tools == ()
    with pytest.warns(DeprecationWarning, match="Agent\\(tools=...\\)"):
        with pytest.raises(TypeError):  # still type-checked: behavior is unchanged
            Agent(tools=[object()])  # type: ignore[list-item]


def test_context_manager_subscriber_warns_and_still_returns_the_agent() -> None:
    from xr_ai_conversation import ContextManager
    from xr_ai_runtime import Agent

    manager = ContextManager(object(), summary_context_topics=("status",))  # type: ignore[arg-type]
    with pytest.warns(DeprecationWarning, match=r"ContextManager.subscriber\(\)"):
        agent = manager.subscriber()
    assert isinstance(agent, Agent)
