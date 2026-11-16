# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose sample configs and delegate lifecycle to existing server utilities."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import yaml
from xr_ai_launcher import Process, detect_gpu_config, require_credentials, run_stack
from xr_ai_logging import setup_logging
from xr_ai_vllm import stop_persistent_servers

from . import _container
from ._registry import ROOT, load_registry

_REGISTRY = load_registry()
_SERVICES_DIR = ROOT / "services"
# Launch order and (project, command, config basename) come from deployment/services.yaml.
_SERVICES = {key: (spec.project, spec.command, spec.config) for key, spec in _REGISTRY.items()}
_SERVICE_ALIASES = {name: spec.key for spec in _REGISTRY.values() for name in (spec.key, *spec.aliases)}


def service_key(name: str) -> str:
    """Return the launcher key for a profile's ``deployment.service`` value."""
    key = _SERVICE_ALIASES.get(name)
    if key is None:
        raise ValueError(f"unknown model service {name!r}; known: {', '.join(sorted(_SERVICE_ALIASES))}")
    return key


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: could not read YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def _configs(
    base: Path, services: tuple[str, ...], profile: str | None, *, shared: Path | None = None
) -> dict[str, dict]:
    """Merge each service's YAML with the GPU profile's overrides.

    ``shared`` is a directory of shared GPU presets (``deployment/gpu-profiles``); its files may name
    more services than are launched. Without it the sample's ``base/yaml`` profile must name exactly
    the launched services.
    """
    profile_dir = shared if shared is not None else base / "yaml"
    overrides_path = profile_dir / f"{profile}.yaml"
    overrides = {}
    if profile is not None:
        profiles = {path.stem for path in profile_dir.glob("*.yaml")}
        if profile not in profiles:
            raise ValueError(f"unknown GPU profile {profile!r}; available: {', '.join(sorted(profiles))}")
        overrides = _read_yaml(overrides_path)
        if shared is not None:
            missing = [name for name in services if name not in overrides]
            if missing:
                raise ValueError(f"{overrides_path}: missing servers: {', '.join(missing)}")
            overrides = {name: overrides[name] for name in services}
        elif set(overrides) != set(services):
            raise ValueError(f"{overrides_path}: must contain exactly these servers: {', '.join(services)}")
    configs = {}
    for service in services:
        spec = _REGISTRY.get(service)
        if spec is None:
            raise ValueError(f"unknown model service {service!r}")
        override = overrides.get(service, {})
        if not isinstance(override, dict):
            raise ValueError(f"{overrides_path}: {service} must be a mapping")
        # Per-sample presets (legacy launch scripts) may still override a port; shared presets may not.
        if shared is not None and "port" in override:
            raise ValueError(f"{overrides_path}: {service}: ports are set only in deployment/services.yaml")
        port = override["port"] if "port" in override and shared is None else spec.port
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"{service}: port must be an integer between 1 and 65535")
        if spec.kind == "container":
            configs[service] = {**override, "port": port}
            continue
        common_path = _SERVICES_DIR / spec.project / f"{spec.config}.yaml"
        config = _read_yaml(common_path)
        config.update(override)
        # Servers resolve relative cache paths against the supplied YAML. Resolve
        # before materializing in a temporary directory; overrides belong to the sample.
        cache_source = overrides_path if "model_cache" in override else common_path
        cache_value = config.get("model_cache")
        if not isinstance(cache_value, str) or not cache_value:
            raise ValueError(f"{service}: model_cache must be a non-empty path")
        cache = Path(cache_value)
        config["model_cache"] = str((cache_source.parent / cache).resolve())
        config["port"] = port  # the registry, not the service's own YAML, is the source of a launched port
        configs[service] = config
    if len({config["port"] for config in configs.values()}) != len(configs):
        raise ValueError("model services must use distinct ports")
    return configs


def _processes(configs: dict[str, dict], directory: Path, root: Path) -> list[Process]:
    processes = []
    for service, (project, command, basename) in _SERVICES.items():
        if service not in configs or _REGISTRY[service].kind == "container":
            continue
        config = configs[service]
        path = directory / f"{basename}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        path.chmod(0o600)
        processes.append(Process(
            service, str(root / "services" / project), command,
            config=path, launch_mode="persist", port=config["port"],
        ))
    return processes


def run_model_stack(
    base: Path,
    services: tuple[str, ...],
    *,
    stop: bool = False,
    gpu_profile: str | None = None,
) -> None:
    """Start or stop the configured local inference stack for a developer sample.

    ``base`` is the sample's model-server project directory, conventionally
    ``<repository>/agent-samples/<sample>/model-servers`` or
    ``<repository>/apps/<application>/model-servers``. Relative paths are
    resolved against the current working directory. Service projects are
    resolved under that repository's ``services/`` directory.

    ``services`` selects from ``speaker-stt``, ``stt``, ``tts``, ``omni``, ``vlm``,
    ``clef``, and ``embedding``. The selected ``base/yaml/<gpu_profile>.yaml`` must contain
    exactly those server mappings. Each mapping overrides the corresponding
    service's own YAML in ``services/``. Launch order is speaker ASR, batch STT, TTS,
    Omni, VLM, Clef, then embedding, omitting unselected services.

    ``gpu_profile`` selects a checked-in profile by filename stem; ``None``
    detects the hardware automatically for startup. ``stop=False`` requires ``HF_TOKEN``,
    waits for every server to become ready, and returns with servers running.
    ``stop=True`` needs neither GPU detection nor download credentials. With no
    explicit profile it uses the selected services' shared default ports; pass
    ``gpu_profile`` to honor profile-specific port overrides. It stops only recognized
    servers on those selected ports. It can stop servers used
    by another sample sharing those ports; no launch history is recorded.

    Stop the current stack before switching samples, editing its server
    settings, or updating the checkout. This convenience API does not read
    consumer endpoint profiles or launch independently deployed NIM endpoints.

    Raises:
        ValueError: The selected profile or server configuration is invalid.
        RuntimeError: GPU inventory or configured server cleanup fails.
        SystemExit: Required credentials are absent or stack startup aborts.
    """
    base = base.resolve()
    _run(base, base.parents[2], f"{base.parent.name}-models", services, stop, gpu_profile, None)


def _ordered(services: tuple[str, ...]) -> tuple[str, ...]:
    keys = {service_key(name) for name in services}
    return tuple(name for name in _SERVICES if name in keys)


def profile_service_ports(
    root: Path, services: tuple[str, ...], *, gpu_profile: str | None = None
) -> dict[str, int]:
    """Return launcher key to port for ``services`` (ports come from ``deployment/services.yaml``).

    Lets a caller see which servers already listen before starting, so it stops only its own.
    """
    ordered = _ordered(services)
    return {name: _REGISTRY[name].port for name in ordered}


def plan_profile_services(
    root: Path, services: tuple[str, ...], *, gpu_profile: str | None = None, namespace: str = "profile"
) -> list[str]:
    """Describe what :func:`run_profile_services` would start, without starting, detecting or pulling anything.

    A dry run never detects hardware: without ``gpu_profile`` the GPU placement is left out (containers use all
    GPUs in the printed command) and each line says so. One line per server, in launch order.
    """
    ordered = _ordered(services)
    if not ordered:
        return []
    root = root.resolve()
    configs = (
        _configs(root, ordered, gpu_profile, shared=root / "deployment" / "gpu-profiles")
        if gpu_profile is not None else {name: {"port": _REGISTRY[name].port} for name in ordered}
    )
    note = "" if gpu_profile is not None else " (GPU profile detected at launch; placement not shown)"
    lines = []
    for name in ordered:
        spec, config = _REGISTRY[name], configs[name]
        if spec.kind == "container":
            command = _container.run_command(spec, config, namespace, root / "models")
            lines.append(f"{name} :{spec.port} container: {' '.join(command)}{note}")
        else:
            lines.append(f"{name} :{spec.port} uv run in services/{spec.project}: {spec.command}{note}")
    return lines


def run_profile_services(
    root: Path,
    services: tuple[str, ...],
    *,
    gpu_profile: str | None = None,
    stop: bool = False,
    namespace: str = "profile",
) -> None:
    """Start or stop the model servers a profile marks ``ownership: managed``.

    Same behavior as :func:`run_model_stack` (persistent servers, ``HF_TOKEN``, readiness, scoped stop),
    but the GPU placement comes from the shared presets in ``<root>/deployment/gpu-profiles`` instead of
    a sample's ``model-servers/yaml``, and ``services`` may be any subset of what the preset names.
    ``services`` are launcher keys, aliases or ``services/`` directory names; the launch order is the
    one in ``deployment/services.yaml``. Servers of kind ``container`` (OCR) are started with ``docker run``
    after the process servers are ready. ``root`` is the repository root.
    """
    ordered = _ordered(services)
    if not ordered:
        return
    root = root.resolve()
    _run(root, root, namespace, ordered, stop, gpu_profile, root / "deployment" / "gpu-profiles")


def _run(
    base: Path,
    root: Path,
    namespace: str,
    services: tuple[str, ...],
    stop: bool,
    gpu_profile: str | None,
    shared: Path | None,
) -> None:
    log_file = setup_logging("orchestrator", namespace=namespace)
    extra = {} if shared is None else {"shared": shared}
    if stop:
        configs = _configs(base, services, gpu_profile, **extra)
        containers = [name for name in configs if _REGISTRY[name].kind == "container"]
        processes = [(name, config["port"]) for name, config in configs.items() if name not in containers]
        stopped = stop_persistent_servers(processes) if processes else True
        for name in containers:
            stopped = _container.stop(_REGISTRY[name], namespace) and stopped
        if not stopped:
            raise RuntimeError("one or more configured model servers could not be stopped")
        return
    configs = _configs(base, services, gpu_profile or detect_gpu_config(), **extra)
    require_credentials("HF_TOKEN")
    process_configs = {name: c for name, c in configs.items() if _REGISTRY[name].kind != "container"}
    # Configs stay available through child initialization and readiness. Services
    # read their config at startup; persistent servers do not reopen it afterward.
    directory = tempfile.mkdtemp(prefix="model-configs-", dir=log_file.parent)
    try:
        if process_configs:
            run_stack(_processes(process_configs, Path(directory), root), base, exit_after_ready=True)
        for name, config in configs.items():
            if name not in process_configs:
                _container.start(_REGISTRY[name], config, namespace, root / "models")
    except BaseException:
        # Keep failure-summary Config paths usable, without retaining inline HF credentials.
        for service, config in process_configs.items():
            path = Path(directory) / f"{_SERVICES[service][2]}.yaml"
            if path.exists() and config.get("hf_token"):
                path.write_text(yaml.safe_dump({**config, "hf_token": ""}, sort_keys=False), encoding="utf-8")
        raise
    else:
        shutil.rmtree(directory)
