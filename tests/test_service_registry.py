# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""deployment/services.yaml is the one port registry and the declarative launch spec of every model server."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from xr_ai_model_server_stack import _container, _registry, _stack
from xr_ai_model_server_stack._stack import plan_profile_services, service_key

_ROOT = Path(__file__).resolve().parents[1]
_SHARED = _ROOT / "deployment" / "gpu-profiles"
_HARDWARE = ["dual_48G_ada", "96G_blackwell", "spark"]
_REG = _registry.load_registry()
_PROCESS = [s for s in _REG.values() if s.kind == "process"]


def test_registry_has_the_required_and_optional_servers_in_launch_order():
    assert list(_REG) == [
        "speaker-stt", "stt", "tts", "magpie", "omni", "nano", "vlm", "clef", "embedding", "ocr",
    ]
    assert _REG["ocr"].kind == "container"
    assert [service_key(n) for n in ("nemotron3-nano-llm", "magpie-tts", "ocr-server", "stt-server")] == [
        "nano", "magpie", "ocr", "stt",
    ]


@pytest.mark.parametrize("spec", _PROCESS, ids=lambda s: s.key)
def test_service_yaml_and_code_default_agree_with_the_registry(spec):
    """The service's own port is only for running it by hand; the launcher never reads it."""
    directory = _ROOT / "services" / spec.project
    own = yaml.safe_load((directory / f"{spec.config}.yaml").read_text())
    assert own["port"] == spec.port
    code = "\n".join(p.read_text() for p in directory.glob("*/__main__.py"))
    default = re.search(r"_DEFAULT_PORT\s*=\s*(\d+)", code)
    assert default is None or int(default.group(1)) == spec.port  # clef has no code default
    assert (directory / "pyproject.toml").is_file()
    assert spec.command in (directory / "pyproject.toml").read_text()


@pytest.mark.parametrize("hardware", _HARDWARE)
def test_launched_ports_come_from_the_registry_only(hardware, tmp_path):
    configs = _stack._configs(_ROOT, tuple(_REG), hardware, shared=_SHARED)
    assert {k: c["port"] for k, c in configs.items()} == {k: s.port for k, s in _REG.items()}
    bad = tmp_path / "x.yaml"
    bad.write_text("omni:\n  port: 9999\n")
    with pytest.raises(ValueError, match="ports are set only in deployment/services.yaml"):
        _stack._configs(_ROOT, ("omni",), "x", shared=tmp_path)


def test_no_other_launcher_file_names_a_service_port():
    for path in [*(_ROOT / "utils/xr-ai-model-server-stack/xr_ai_model_server_stack").glob("*.py")]:
        for port in {s.port for s in _REG.values()}:
            assert not re.search(rf"\b{port}\b", path.read_text()), f"{path.name} hard-codes {port}"


def test_registry_validation_rejects_duplicates_and_bad_entries():
    ok = {"project": "p", "command": "c", "config": "c", "port": 8000}
    with pytest.raises(ValueError, match="port 8000 is used by a and b"):
        _registry.parse_registry({"services": {"a": ok, "b": ok}})
    with pytest.raises(ValueError, match="name 'a' is used by a and b"):
        _registry.parse_registry({"services": {"a": ok, "b": {**ok, "port": 8001, "aliases": ["a"]}}})
    with pytest.raises(ValueError, match="process services need command"):
        _registry.parse_registry({"services": {"a": {"project": "p", "config": "c", "port": 1}}})
    with pytest.raises(ValueError, match="container services need image"):
        _registry.parse_registry({"services": {"a": {"project": "p", "kind": "container", "port": 1}}})
    with pytest.raises(ValueError, match="kind must be"):
        _registry.parse_registry({"services": {"a": {**ok, "kind": "vm"}}})
    with pytest.raises(ValueError, match="non-empty 'services'"):
        _registry.parse_registry({})


def test_optional_servers_start_from_a_profile_in_registry_order(monkeypatch):
    started = {}
    monkeypatch.setattr(_stack, "setup_logging", lambda *_a, **_k: Path("/tmp") / "orchestrator.log")
    monkeypatch.setattr(_stack, "detect_gpu_config", lambda: "96G_blackwell")
    monkeypatch.setattr(_stack, "require_credentials", lambda *_: None)
    monkeypatch.setattr(_stack, "run_stack", lambda processes, base, **kw: started.update(
        names=[p.name for p in processes], commands=[p.command for p in processes]))
    containers = []
    monkeypatch.setattr(_container, "start", lambda spec, config, ns, models: containers.append((spec.key, config)))
    _stack.run_profile_services(_ROOT, ("ocr-server", "nemotron3-nano-llm", "magpie-tts", "stt-server"))
    assert started["names"] == ["stt", "magpie", "nano"]
    assert started["commands"] == ["stt_server", "magpie_tts_server", "nemotron3_nano_llm_server"]
    assert containers == [("ocr", {"cuda_visible_devices": "0", "port": 8112})]


def test_stop_stops_processes_and_the_ocr_container(monkeypatch):
    monkeypatch.setattr(_stack, "setup_logging", lambda *_a, **_k: Path("/tmp") / "orchestrator.log")
    stopped, docker = [], []
    monkeypatch.setattr(_stack, "stop_persistent_servers", lambda targets: stopped.extend(targets) or True)
    monkeypatch.setattr(_container, "stop", lambda spec, ns: docker.append((spec.key, ns)) or True)
    _stack.run_profile_services(_ROOT, ("nano", "ocr"), stop=True, gpu_profile="spark", namespace="n")
    assert stopped == [("nano", 8107)] and docker == [("ocr", "n")]


def test_ocr_docker_command_and_readiness_without_docker():
    spec = _REG["ocr"]
    command = _container.run_command(spec, {"cuda_visible_devices": "1"}, "tea", Path("/repo/models"))
    assert command[:5] == ["docker", "run", "-d", "--rm", "--name"] and "xr-ai-ocr-tea" in command
    assert command[command.index("--gpus") + 1] == "device=1"
    assert "127.0.0.1:8112:8000" in command and command[-1] == "xr-ai-nemotron-ocr:local"
    assert "OCR_INFER_LENGTH=1024" in command

    calls, ticks = [], iter([False, False, True])
    _container.start(spec, {}, "tea", Path("/tmp/xr-ai-ocr-models-test"), runner=lambda c, **k: calls.append(c),
                     ready=lambda port: next(ticks), sleep=lambda s: None)
    assert calls[0][0:2] == ["docker", "run"]

    stops = []
    with pytest.raises(RuntimeError, match="not ready on port 8112"):
        _container.start(spec, {}, "tea", Path("/tmp/xr-ai-ocr-models-test"), timeout_s=-1,
                         runner=lambda c, **k: stops.append(c), ready=lambda p: False, sleep=lambda s: None)
    assert stops[-1] == ["docker", "stop", "xr-ai-ocr-tea"]


def test_dry_run_plan_lists_servers_without_starting_anything(monkeypatch):
    monkeypatch.setattr(_stack, "run_stack", lambda *a, **k: pytest.fail("started"))
    monkeypatch.setattr(_stack, "detect_gpu_config", lambda: pytest.fail("detected"))
    lines = plan_profile_services(_ROOT, ("ocr", "nano", "magpie"), gpu_profile="spark")
    assert [line.split()[0] for line in lines] == ["magpie", "nano", "ocr"]
    assert "docker run" in lines[2] and ":8112" in lines[2]
    undetected = plan_profile_services(_ROOT, ("nano", "ocr"))
    assert all("GPU profile detected at launch" in line for line in undetected)
    assert undetected[0].startswith("nano :8107") and "--gpus all" in undetected[1]
    assert plan_profile_services(_ROOT, ()) == []
