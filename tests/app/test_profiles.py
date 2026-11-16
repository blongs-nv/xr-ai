# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile selection, loading, role resolution and category checks (the package's docs/PROFILES_SCHEMA.md)."""

from __future__ import annotations

import asyncio
import json
import warnings
from pathlib import Path

import pytest
import yaml
from xr_ai_app._base import profiles
from xr_ai_app._base.enrollment import PerConnection
from xr_ai_app._base.profiles import (
    Role,
    load_profile,
    resolve_role,
    resolve_roles,
    select_profile,
)


def _entry(preset: str, url: str = "http://localhost:8100", **extra: object) -> dict:
    return {"adapter": {"preset": preset}, "endpoint": {"base_url": url}, **extra}


_LEGACY = {
    "llm": _entry("nemotron_omni"),
    "vlm": _entry("cosmos3_nano_reasoner"),
    "tts": _entry("pocket_tts"),
    "embedding": _entry("nemotron_embedding"),
}


_ROLES = {"perceive": "vlm", "answer": "llm", "embed": "embedding", "speech_out": "tts"}
"""What a profile with modality-named entries says to fill the standard roles (there are no aliases)."""


def _write(path: Path, body: dict) -> Path:
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


# --- select_profile ---------------------------------------------------------------------------


def test_default_profile_is_models_json(tmp_path, monkeypatch):
    monkeypatch.delenv("XR_AI_PROFILE", raising=False)
    (tmp_path / "models.json").write_text("{}")
    assert select_profile(tmp_path) == tmp_path / "models.json"
    assert select_profile(tmp_path, None) == tmp_path / "models.json"


def test_default_profile_accepts_yaml(tmp_path, monkeypatch):
    monkeypatch.delenv("XR_AI_PROFILE", raising=False)
    (tmp_path / "models.yaml").write_text("{}")
    assert select_profile(tmp_path) == tmp_path / "models.yaml"


def test_named_profile_prefers_json_then_yaml_then_yml(tmp_path):
    (tmp_path / "models.hybrid.yml").write_text("{}")
    assert select_profile(tmp_path, "hybrid").name == "models.hybrid.yml"
    (tmp_path / "models.hybrid.yaml").write_text("{}")
    assert select_profile(tmp_path, "hybrid").name == "models.hybrid.yaml"
    (tmp_path / "models.hybrid.json").write_text("{}")
    assert select_profile(tmp_path, "hybrid").name == "models.hybrid.json"


def test_precedence_argument_over_environment_over_default(tmp_path, monkeypatch):
    for name in ("models.json", "models.local.json", "models.nim.json"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setenv("XR_AI_PROFILE", "nim")
    assert select_profile(tmp_path).name == "models.nim.json"
    assert select_profile(tmp_path, "local").name == "models.local.json"
    monkeypatch.setenv("XR_AI_PROFILE", "")
    assert select_profile(tmp_path).name == "models.json"


def test_path_values_resolve_against_config_dir(tmp_path):
    (tmp_path / "sub").mkdir()
    target = tmp_path / "sub" / "custom.json"
    target.write_text("{}")
    assert select_profile(tmp_path, "sub/custom.json") == tmp_path / "sub" / "custom.json"
    assert select_profile(tmp_path, str(target)) == target
    assert select_profile(tmp_path / "elsewhere", str(target)) == target


def test_missing_profile_lists_candidates(tmp_path, monkeypatch):
    monkeypatch.delenv("XR_AI_PROFILE", raising=False)
    with pytest.raises(FileNotFoundError, match=r"models\.nope\.json.*models\.nope\.yaml.*models\.nope\.yml"):
        select_profile(tmp_path, "nope")
    with pytest.raises(FileNotFoundError, match="missing.json"):
        select_profile(tmp_path, "missing.json")


# --- loading and resolution -------------------------------------------------------------------


def test_modality_named_entries_fill_standard_roles_only_through_a_roles_map(tmp_path):
    profile = load_profile(_write(tmp_path / "models.json", {"models": _LEGACY, "roles": _ROLES}))
    assert profile.budgets == {}
    assert resolve_roles(profile) == {"perceive": "vlm", "answer": "llm", "embed": "embedding", "speech_out": "tts"}
    assert profile.resolve("perceive") == "vlm"


def test_direct_form_still_loads(tmp_path):
    profile = load_profile(_write(tmp_path / "models.json", _LEGACY))
    assert set(profile.models.entries) == set(_LEGACY)


def test_modality_names_are_not_aliases_of_standard_roles(tmp_path):
    assert not hasattr(profiles, "LEGACY_ROLE_ALIASES")
    profile = load_profile(_write(tmp_path / "models.json", {"models": _LEGACY}))
    assert resolve_roles(profile) == {}
    with pytest.raises(KeyError, match="no model for role 'perceive'.*vlm"):
        resolve_role(profile, "perceive")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_role(profile, "vlm") == "vlm"  # an entry name, like any other


def test_explicit_entry_and_roles_win_over_an_entry_of_another_name(tmp_path):
    body = {
        "models": {
            "vlm": _entry("cosmos3_nano_reasoner"),
            "perceive": _entry("cosmos_vlm", "http://localhost:9"),
            "llm": _entry("nemotron_omni"),
            "big": _entry("nemotron3_nano", "https://hosted"),
        },
        "roles": {"extract": "big", "ack": "llm", "answer": "llm"},
    }
    profile = load_profile(_write(tmp_path / "p.json", body))
    assert resolve_role(profile, "perceive") == "perceive"
    assert resolve_role(profile, "extract") == "big"
    assert resolve_role(profile, "ack") == "llm"
    assert resolve_role(profile, "answer") == "llm"


def test_roles_map_beats_an_entry_of_the_same_job(tmp_path):
    body = {"models": {"llm": _entry("nemotron_omni"), "big": _entry("nemotron3_nano")}, "roles": {"answer": "big"}}
    profile = load_profile(_write(tmp_path / "p.json", body))
    assert resolve_role(profile, "answer") == "big"
    assert resolve_role(profile, "llm") == "llm"  # the entry itself, by its own name


def test_custom_role_names_resolve_as_entries(tmp_path):
    body = {"models": {"reader2": _entry("cosmos3_nano_reasoner")}}
    profile = load_profile(_write(tmp_path / "p.json", body))
    assert resolve_role(profile, "reader2") == "reader2"
    with pytest.raises(KeyError, match="no model for role 'perceive'.*reader2"):
        resolve_role(profile, "perceive")


def test_role_that_is_entry_and_roles_key_is_an_error(tmp_path):
    body = {"models": _LEGACY, "roles": {"llm": "vlm"}}
    with pytest.raises(ValueError, match="both a models entry and a roles key"):
        load_profile(_write(tmp_path / "p.json", body))


def test_roles_target_must_be_an_entry(tmp_path):
    with pytest.raises(ValueError, match=r"roles\.extract must name an entry"):
        load_profile(_write(tmp_path / "p.json", {"models": _LEGACY, "roles": {"extract": "nope"}}))


def test_category_mismatch_names_file_and_role(tmp_path):
    body = {"models": {"perceive": _entry("pocket_tts")}}
    path = _write(tmp_path / "bad.json", body)
    with pytest.raises(ValueError, match=r"bad\.json.*role 'perceive'.*'tts'.*\['vlm'\]"):
        load_profile(path)
    body = {"models": {"vlm": _entry("cosmos3_nano_reasoner")}, "roles": {"answer": "vlm"}}
    with pytest.raises(ValueError, match="role 'answer'"):
        load_profile(_write(tmp_path / "bad2.json", body))


def test_extract_accepts_llm_or_vlm(tmp_path):
    for entry in ("llm", "vlm"):
        body = {"models": _LEGACY, "roles": {"extract": entry}}
        assert load_profile(_write(tmp_path / "p.json", body)).resolve("extract") == entry


def test_wrapped_keys_need_the_wrapped_form(tmp_path):
    with pytest.raises(ValueError, match="wrapped form"):
        load_profile(_write(tmp_path / "p.json", {**_LEGACY, "roles": {"extract": "llm"}}))


def test_yaml_profile_with_metadata_and_budgets(tmp_path):
    body = {
        "profile": {"name": "hybrid", "description": "d", "gpu_profile": "spark"},
        "models": _LEGACY,
        "roles": _ROLES,
        "budgets": {"perceive": {"max_concurrency": 2}, "tts": {"max_concurrency": 1}},
    }
    path = tmp_path / "models.hybrid.yaml"
    path.write_text(yaml.safe_dump(body))
    profile = load_profile(path)
    assert (profile.name, profile.description, profile.gpu_profile) == ("hybrid", "d", "spark")
    assert profile.budget("vlm").max_concurrency == 2
    assert profile.budget("speech_out").max_concurrency == 1
    assert profile.budget("answer").max_concurrency is None


@pytest.mark.parametrize(
    "budgets, match",
    [
        ({"perceive": {"max_concurrency": 0}}, "max_concurrency"),
        ({"perceive": {"max_fps": 4}}, "must be a mapping with max_concurrency"),
        ({"perceive": {"speed": 1}}, "must be a mapping"),
        ({"nonexistent": {"max_concurrency": 1}}, "no model for role"),
    ],
)
def test_invalid_budgets(tmp_path, budgets, match):
    with pytest.raises((ValueError, KeyError), match=match):
        load_profile(_write(tmp_path / "p.json", {"models": _LEGACY, "roles": _ROLES, "budgets": budgets}))


def test_unknown_gpu_profile_rejected(tmp_path):
    with pytest.raises(ValueError, match="gpu_profile"):
        load_profile(_write(tmp_path / "p.json", {"profile": {"gpu_profile": "h100"}, "models": _LEGACY}))


def test_speaker_asr_endpoint_comes_from_speech_in(tmp_path):
    speaker = {
        "category": "stt",
        "adapter": {"kind": "openai_compat", "capabilities": {"speaker_stream": True}},
        "endpoint": {"base_url": "http://127.0.0.1:8102"},
        "deployment": {"ownership": "managed", "service": "speaker-stt"},
    }
    profile = load_profile(_write(tmp_path / "p.json", {"models": {**_LEGACY, "speech_in": speaker}}))
    assert profile.speaker_asr_base_url() == "http://127.0.0.1:8102"
    batch = {"models": {**_LEGACY, "stt": _entry("parakeet_stt", "http://localhost:8103")}}
    assert load_profile(_write(tmp_path / "q.json", batch)).speaker_asr_base_url() is None
    assert load_profile(_write(tmp_path / "r.json", {"models": _LEGACY})).speaker_asr_base_url() is None


# --- speech roles have no chat methods ----------------------------------------------------------


def test_a_role_bound_to_a_speech_entry_refuses_chat_methods():
    for name, category in (("speech_in", "stt"), ("speech_out", "tts")):
        role = Role.bind(name, entry="my_speech", category=category, client=None)

        async def use(role=role):
            with pytest.raises(TypeError, match=rf"{role.name}.*speech entry.*ask"):
                await role.ask(None, "hello")
            with pytest.raises(TypeError, match="stream"):
                role.stream(None, "hello")
            with pytest.raises(TypeError, match="decide"):
                await role.decide("q", choices=("a", "b"))
            with pytest.raises(TypeError, match="speech entry"):
                await role.as_llm().chat([])
            with pytest.raises(TypeError, match="speech entry"):
                await role.as_vlm().ask_image(b"x", "what")

        asyncio.run(use())


def test_the_client_pool_hands_a_speech_entry_no_client(tmp_path):
    from xr_ai_app._base._bound import ClientPool

    body = {"models": {"tts": _entry("pocket_tts")}, "roles": {"speech_out": "tts"}}
    built: list[object] = []
    pool = ClientPool(load_profile(_write(tmp_path / "p.json", body)), built.append)
    assert pool.client("tts") is None and built == []


def test_the_test_kit_refuses_chat_methods_on_speech_roles_and_takes_names_as_given():
    from xr_ai_app._base.testing import FakeModels

    models = FakeModels({"answer": ["hi"]})

    async def use():
        for name in ("speech_in", "speech_out"):
            with pytest.raises(TypeError, match="speech entry"):
                await models.role(name).ask(None, "x")
            with pytest.raises(TypeError, match="speech entry"):
                models.role(name).stream(None, "x")
        assert await models.role("answer").ask(None, "x") == "hi"
        assert models.role("vlm").name == "vlm"  # no alias: ``vlm`` is its own script, not ``perceive``
        assert models.calls("perceive") == []

    asyncio.run(use())


def test_profile_precedence_explicit_then_env_then_app_yaml(tmp_path, monkeypatch) -> None:
    """``profile=`` outranks ``XR_AI_PROFILE``, which outranks ``profile:`` in app.yaml."""
    from xr_ai_app._base.app import App

    for name in ("a", "b", "c"):
        (tmp_path / f"models.{name}.json").write_text('{"models": {}}')
    (tmp_path / "app.yaml").write_text("profile: a\nvoice: {enabled: false}\n")

    def chosen(**kw) -> str:
        app = App.from_config(tmp_path / "app.yaml", **kw, enrollment=PerConnection(admit=None))
        return app._load_profile().path.name

    monkeypatch.delenv("XR_AI_PROFILE", raising=False)
    assert chosen() == "models.a.json"
    monkeypatch.setenv("XR_AI_PROFILE", "b")
    assert chosen() == "models.b.json"
    assert chosen(profile="c") == "models.c.json"
