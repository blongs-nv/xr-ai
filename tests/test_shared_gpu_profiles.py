# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three shared GPU placement presets replace the per-sample files, and start any subset of services."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from xr_ai_model_server_stack import _stack
from xr_ai_model_server_stack._stack import run_profile_services, service_key
from xr_ai_next._base import profiles

_ROOT = Path(__file__).resolve().parents[1]
_SHARED = _ROOT / "deployment" / "gpu-profiles"
_HARDWARE = ["dual_48G_ada", "96G_blackwell", "spark"]
_ALL = tuple(_stack._SERVICES)


@pytest.fixture(autouse=True)
def isolate_stack_logging(monkeypatch, tmp_path):
    log_file = tmp_path / "logs" / "orchestrator.log"
    log_file.parent.mkdir()
    monkeypatch.setattr(_stack, "setup_logging", lambda *_a, **_kw: log_file)


def test_the_presets_are_the_three_named_in_the_schema():
    assert {p.stem for p in _SHARED.glob("*.yaml")} == set(_HARDWARE) == set(profiles.GPU_PROFILES)


@pytest.mark.parametrize("hardware", _HARDWARE)
def test_shared_preset_names_every_launchable_service(hardware):
    assert set(yaml.safe_load((_SHARED / f"{hardware}.yaml").read_text())) == set(_ALL)


@pytest.mark.parametrize("hardware", _HARDWARE)
def test_shared_preset_matches_every_per_sample_file(hardware):
    shared = yaml.safe_load((_SHARED / f"{hardware}.yaml").read_text())
    per_sample = sorted((_ROOT / "agent-samples").glob(f"*/model-servers/yaml/{hardware}.yaml"))
    assert per_sample
    for path in per_sample:
        for service, settings in yaml.safe_load(path.read_text()).items():
            assert shared[service] == settings, f"{path}: {service}"


@pytest.mark.parametrize("hardware", _HARDWARE)
def test_any_subset_resolves_with_distinct_ports(hardware):
    configs = _stack._configs(_ROOT, _ALL, hardware, shared=_SHARED)
    assert set(configs) == set(_ALL)
    assert len({c["port"] for c in configs.values()}) == len(_ALL)
    subset = _stack._configs(_ROOT, ("tts", "omni"), hardware, shared=_SHARED)
    assert set(subset) == {"tts", "omni"}
    assert subset["omni"]["port"] == configs["omni"]["port"]


def test_shared_mode_reports_unknown_preset_and_missing_services(tmp_path):
    with pytest.raises(ValueError, match="unknown GPU profile 'nope'; available: 96G_blackwell, dual_48G_ada, spark"):
        _stack._configs(_ROOT, ("tts",), "nope", shared=_SHARED)
    (tmp_path / "tiny.yaml").write_text("tts: {}\n")
    with pytest.raises(ValueError, match="missing servers: omni"):
        _stack._configs(_ROOT, ("tts", "omni"), "tiny", shared=tmp_path)


def test_per_sample_mode_still_requires_exact_services(tmp_path):
    (tmp_path / "yaml").mkdir()
    (tmp_path / "yaml/custom.yaml").write_text("tts: {}\nomni: {}\n")
    with pytest.raises(ValueError, match="must contain exactly these servers"):
        _stack._configs(tmp_path, ("tts",), "custom")


def test_service_names_accept_directory_aliases():
    assert [service_key(n) for n in ("vlm-server", "clef-server", "pocket-tts", "tts", "speaker-stt")] == [
        "vlm", "clef", "tts", "tts", "speaker-stt",
    ]
    with pytest.raises(ValueError, match="unknown model service 'bogus'"):
        service_key("bogus")


def test_profile_services_start_in_stack_order_without_duplicates(monkeypatch):
    started = {}
    monkeypatch.setattr(_stack, "detect_gpu_config", lambda: "spark")
    monkeypatch.setattr(_stack, "require_credentials", lambda *_: None)
    monkeypatch.setattr(_stack, "run_stack", lambda processes, base, **kw: started.update(
        names=[p.name for p in processes], base=base, kw=kw,
    ))
    run_profile_services(_ROOT, ("embedding-server", "omni", "vlm", "pocket-tts", "tts"))
    assert started["names"] == ["tts", "omni", "vlm", "embedding"]
    assert started["kw"] == {"exit_after_ready": True}


def test_profile_services_stop_uses_configured_ports(monkeypatch):
    stopped = []
    monkeypatch.setattr(_stack, "stop_persistent_servers", lambda targets: stopped.extend(targets) or True)
    run_profile_services(_ROOT, ("tts", "speaker-stt"), stop=True, gpu_profile="spark")
    assert stopped == [("speaker-stt", 8102), ("tts", 8105)]
    stopped.clear()
    run_profile_services(_ROOT, ())
    assert stopped == []


def test_no_services_starts_nothing(monkeypatch):
    monkeypatch.setattr(_stack, "run_stack", lambda *a, **k: pytest.fail("started"))
    run_profile_services(_ROOT, ())


@pytest.mark.parametrize("hardware", _HARDWARE)
def test_profile_service_ports_follow_the_launcher_order(hardware):
    from xr_ai_model_server_stack._stack import profile_service_ports

    ports = profile_service_ports(_ROOT, ("tts", "pocket-tts", "speaker-stt", "omni"), gpu_profile=hardware)
    assert ports == {"speaker-stt": 8102, "tts": 8105, "omni": 8108}
    assert profile_service_ports(_ROOT, ()) == {}
