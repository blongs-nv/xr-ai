# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job-named model roles and profile selection (see the package's ``docs/PROFILES_SCHEMA.md``)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import yaml
from pydantic import BaseModel
from xr_ai_models import (
    DecisionSpec,
    EmbeddingSpec,
    LLMSpec,
    ModelsConfig,
    OCRSpec,
    STTSpec,
    TTSSpec,
    VLMSpec,
    load_models_config_from_dict,
)

if TYPE_CHECKING:
    from xr_ai_models import LLMService, VLMService

    from .frames import Frame

ROLE_CATEGORIES: Final[Mapping[str, tuple[str, ...]]] = {
    "perceive": ("vlm",),
    "decide": ("decision",),
    "extract": ("llm", "vlm"),
    "answer": ("llm",),
    "ack": ("llm",),
    "embed": ("embedding",),
    "speech_in": ("stt",),
    "speech_out": ("tts",),
}
"""Standard role name to the ``xr_ai_models`` categories that may fill it."""

DEFAULT_PROFILE: Final = "models"
"""Profile name used when none is chosen; resolves to ``models.json`` or ``models.yaml``."""


GPU_PROFILES: Final[tuple[str, ...]] = ("dual_48G_ada", "96G_blackwell", "spark")
"""The shared GPU placement presets under ``deployment/gpu-profiles``."""

PROFILE_ENV: Final = "XR_AI_PROFILE"
"""Environment variable that selects a profile when neither the command line nor ``app.yaml`` does."""

_PROFILE_SUFFIXES: Final = (".json", ".yaml", ".yml")
_WRAPPED_KEYS: Final = ("profile", "roles", "budgets")
_SPEAKER_SERVICE: Final = "speaker-stt"


@dataclass(frozen=True)
class RoleBudget:
    """Inference budget for one role; ``None`` means unlimited."""

    max_concurrency: int | None = None
    """Calls in flight at once."""


@dataclass(frozen=True)
class Profile:
    """A loaded profile file: the unchanged ``models`` config plus ``profile``, ``roles`` and ``budgets``."""

    path: Path
    name: str
    description: str = ""
    gpu_profile: str | None = None
    models: ModelsConfig = field(default_factory=lambda: ModelsConfig(entries={}))
    roles: Mapping[str, str] = field(default_factory=dict)
    budgets: Mapping[str, RoleBudget] = field(default_factory=dict)

    def resolve(self, role: str) -> str:
        """Return the ``models`` entry name that fills ``role`` (see :func:`resolve_role`)."""
        return resolve_role(self, role)

    def resolve_all(self) -> dict[str, str]:
        """Return role to entry name for every standard role this profile fills."""
        return resolve_roles(self)

    def budget(self, role: str) -> RoleBudget:
        """Return the budget of ``role``, set under any name that resolves to the same entry; else unlimited."""
        if role in self.budgets:
            return self.budgets[role]
        entry = resolve_role(self, role)
        for key, budget in self.budgets.items():
            if resolve_role(self, key) == entry:
                return budget
        return RoleBudget()

    def speaker_asr_base_url(self, role: str = "speech_in") -> str | None:
        """Return the speaker-ASR endpoint when ``role`` is the enrolled-speaker service, else ``None``."""
        try:
            entry = resolve_role(self, role)
        except KeyError:
            return None
        spec = self.models.entries[entry]
        if spec.deployment.service == _SPEAKER_SERVICE or spec.adapter.capabilities.get("speaker_stream"):
            return spec.endpoint.base_url or None
        return None


_CATEGORY_OF: Final = (
    (LLMSpec, "llm"),
    (VLMSpec, "vlm"),
    (STTSpec, "stt"),
    (TTSSpec, "tts"),
    (EmbeddingSpec, "embedding"),
    (DecisionSpec, "decision"),
    (OCRSpec, "ocr"),
)


def _category(spec: object) -> str:
    for cls, name in _CATEGORY_OF:
        if isinstance(spec, cls):
            return name
    raise TypeError(f"unknown model spec {type(spec).__name__}")


def resolve_role(profile: Profile, name: str) -> str:
    """Return the ``models`` entry that fills role ``name``.

    ``name`` is a ``roles`` key or the name of a ``models`` entry (a standard role is just a name that
    follows the convention in ``PROFILES_SCHEMA.md``); nothing else resolves, so a modality name such as
    ``vlm`` works only when the profile has an entry or a role of that name. Raises :class:`KeyError`
    listing what the profile has.
    """
    found = profile.roles.get(name) or (name if name in profile.models.entries else None)
    if found is None:
        have = sorted({*profile.roles, *profile.models.entries})
        raise KeyError(f"{profile.path}: no model for role {name!r}; the profile defines {have}")
    return found


def resolve_roles(profile: Profile) -> dict[str, str]:
    """Return role to entry name for every standard role the profile fills (missing ones are left out)."""
    out: dict[str, str] = {}
    for job in ROLE_CATEGORIES:
        try:
            out[job] = resolve_role(profile, job)
        except KeyError:
            continue
    return out


def _read(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: cannot read profile: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    return raw


def _budgets(path: Path, raw: object) -> dict[str, RoleBudget]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: budgets must be a mapping of role to settings")
    out: dict[str, RoleBudget] = {}
    for role, body in raw.items():
        if not isinstance(body, dict) or not body.keys() <= {"max_concurrency"}:
            raise ValueError(f"{path}: budgets.{role} must be a mapping with max_concurrency")
        conc = body.get("max_concurrency")
        if conc is not None and (isinstance(conc, bool) or not isinstance(conc, int) or conc < 1):
            raise ValueError(f"{path}: budgets.{role}.max_concurrency must be a positive integer")
        out[str(role)] = RoleBudget(conc)
    return out


def load_profile(path: Path | str) -> Profile:
    """Load a profile (YAML or JSON) and validate roles, budgets and categories.

    The ``models`` mapping goes through the unchanged ``xr_ai_models`` loader, so every existing
    ``models.json`` loads. Errors name the file and the role.
    """
    path = Path(path)
    raw = _read(path)
    if "models" not in raw and raw.keys() & set(_WRAPPED_KEYS):
        raise ValueError(f"{path}: 'profile', 'roles' and 'budgets' need the wrapped form with a 'models' key")
    models = load_models_config_from_dict(raw, source=str(path))

    meta = raw.get("profile") or {}
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: profile must be a mapping")
    gpu = meta.get("gpu_profile")
    if gpu is not None and gpu not in GPU_PROFILES:
        raise ValueError(f"{path}: profile.gpu_profile {gpu!r} is not one of {list(GPU_PROFILES)}")

    roles_raw = raw.get("roles") or {}
    if not isinstance(roles_raw, dict):
        raise ValueError(f"{path}: roles must be a mapping of role to model entry")
    roles: dict[str, str] = {}
    for role, target in roles_raw.items():
        if role in models.entries:
            raise ValueError(f"{path}: role {role!r} is both a models entry and a roles key")
        if not isinstance(target, str) or target not in models.entries:
            raise ValueError(f"{path}: roles.{role} must name an entry in models, got {target!r}")
        roles[str(role)] = target

    profile = Profile(
        path=path,
        name=str(meta.get("name") or path.stem.removeprefix("models.") or path.stem),
        description=str(meta.get("description") or ""),
        gpu_profile=gpu,
        models=models,
        roles=roles,
    )
    for job, entry in resolve_roles(profile).items():
        category = _category(models.entries[entry])
        if category not in ROLE_CATEGORIES[job]:
            raise ValueError(
                f"{path}: role {job!r} is filled by entry {entry!r} of category {category!r}; "
                f"it needs one of {list(ROLE_CATEGORIES[job])}"
            )
    budgets = _budgets(path, raw.get("budgets"))
    for role in budgets:
        resolve_role(profile, role)
    return Profile(
        path=path,
        name=profile.name,
        description=profile.description,
        gpu_profile=gpu,
        models=models,
        roles=roles,
        budgets=budgets,
    )


class Decision(str):
    """The choice a decision model made: the label itself, plus what the model reported about it.

    It is a ``str``, so ``decision == "start"`` and ``choices.index(decision)`` work unchanged.
    """

    confidence: float | None
    """Probability of the chosen label, or ``None`` when the model does not report it."""

    probabilities: Mapping[str, float] | None
    """Probability of every choice, or ``None`` when the model does not report them."""

    def __new__(
        cls,
        choice: str,
        *,
        confidence: float | None = None,
        probabilities: Mapping[str, float] | None = None,
    ) -> Decision:
        self = super().__new__(cls, choice)
        self.confidence = confidence
        self.probabilities = None if probabilities is None else dict(probabilities)
        return self


SPEECH_CATEGORIES: Final = ("stt", "tts")
"""The ``xr_ai_models`` categories of ``speech_in`` and ``speech_out``; a role bound to one has no chat methods."""


def speech_role_error(name: str, category: str, method: str) -> TypeError:
    """The error ``ask``, ``stream`` and ``decide`` raise on a role bound to a speech entry."""
    return TypeError(
        f"role {name!r} is bound to a speech entry (category {category!r}); {method} is a chat method. "
        "Speech roles are not asked: speech entries are used by the voice agent"
    )


class StructuredOutputError(ValueError):
    """A model reply did not validate against the ``response_format`` of :meth:`Role.ask`."""


class Role:
    """One named model role resolved from the active profile."""

    @classmethod
    def bind(
        cls,
        name: str,
        *,
        entry: str,
        category: str,
        client: Any,
        limiter: Any = None,
    ) -> Role:
        """Return a role whose calls go to ``client``, the real client of profile entry ``entry``.

        ``category`` is the ``xr_ai_models`` category of the entry (``llm``, ``vlm`` or ``decision``);
        ``limiter`` is an optional ``asyncio.Semaphore`` that enforces ``max_concurrency``. ``App``
        builds and closes the clients; this is the path it binds them through.
        """
        from ._bound import BoundRole  # noqa: PLC0415 - ``_bound`` imports this module

        return BoundRole(name, entry=entry, category=category, client=client, limiter=limiter)

    @property
    def name(self) -> str:
        """The job-named role."""
        raise NotImplementedError

    async def ask(
        self,
        frame: Frame | Sequence[Frame] | None,
        prompt: str,
        *,
        response_format: type[BaseModel] | None = None,
        system: str | None = None,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        """Ask the role once; with ``response_format`` return a validated instance of it.

        ``frame`` is one frame, several frames (compared in order by a vision model) or ``None`` for
        a text model. ``system`` is sent as a separate system message, not joined to ``prompt``.
        ``thinking`` requests or suppresses reasoning for this call and is ignored when the preset
        has no switch for it. ``max_tokens`` and ``temperature`` override the profile entry for this
        call. A call that takes longer than ``timeout_s`` raises ``TimeoutError``. ``response_format``'s schema is
        always in the prompt; when the role's model entry declares ``capabilities: {json_schema: true}`` it is also
        sent for the server to constrain the reply to (declare it only for a deployment that passes). A reply that does
        not validate against ``response_format`` is retried once: the same prompt is sent again with a
        reminder to answer with the JSON object only (the bad reply and the schema error are not fed
        back), so ``timeout_s`` applies to each attempt and the worst case is twice ``timeout_s``. A
        second invalid reply raises :class:`StructuredOutputError`
        (a ``ValueError``).
        """
        raise NotImplementedError

    def stream(
        self,
        frame: Frame | Sequence[Frame] | None,
        prompt: str,
        *,
        system: str | None = None,
        thinking: bool | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream text for one prompt, cancelled with the calling turn.

        The keyword options mean what they do on :meth:`ask`; ``timeout_s`` bounds the whole stream
        and raises ``TimeoutError`` from the iterator when it runs out.
        """
        raise NotImplementedError

    def as_llm(self) -> LLMService:
        """Return this role as an ``LLMService``: a tool-calling chat for code that runs its own tool loop.

        ``chat`` and ``stream_events`` take a message transcript plus ``tools``, ``max_tokens``, ``temperature`` and
        ``enable_thinking``. A bound role goes to its client under the role's concurrency limit and, like ``ask`` and
        ``stream``, is cancelled with the calling task; closing the client stays with the app. A role of the test kit
        answers through ``ask`` with the transcript rendered as one prompt, and a scripted ``ToolCall`` reply comes
        back as a tool call. The role is resolved when the service is first used.
        """
        from ._role_llm import _RoleLLM  # noqa: PLC0415 - ``_role_llm`` imports this module

        return _RoleLLM(lambda: self)  # type: ignore[return-value]

    def as_vlm(self) -> VLMService:
        """Return this role as a ``VLMService``: ``ask_image``, ``ask_images`` and ``ask_video``.

        Use it for capability tools that want a vision service (``ImageQueryTool``, ``VideoQueryTool``). A bound role
        goes to its client under the role's concurrency limit with every option passed through; closing the client
        stays with the app. A role of the test kit answers ``ask_image`` and ``ask_images`` through ``ask`` (the
        images become frames; only ``system_prompt``, ``max_tokens``, ``temperature`` and ``timeout`` are kept) and
        has no ``ask_video``. The role is resolved when the service is first used.
        """
        from ._role_vlm import _RoleVLM  # noqa: PLC0415 - ``_role_vlm`` imports this module

        return _RoleVLM(lambda: self)  # type: ignore[return-value]

    async def decide(
        self,
        question: str,
        *,
        choices: Sequence[str] | Mapping[str, str],
        context: str = "",
        frames: Sequence[Frame] = (),
    ) -> Decision:
        """Pick one of ``choices``; only valid on a role backed by a decision model.

        A sequence names the choices; a mapping gives each choice its criteria text. ``context`` is
        the state the question is judged against and ``frames`` are images shown to the model in
        order. The result is a :class:`Decision`, a ``str`` that also carries the model's confidence
        and probabilities when it reports them.
        """
        raise NotImplementedError


def select_profile(config_dir: Path, profile: str | None = None) -> Path:
    """Return the model profile file for ``profile`` using the documented precedence.

    ``profile`` is the highest-priority choice (``--profile``; ``App`` passes ``XR_AI_PROFILE``, then ``profile:`` from
    ``app.yaml``, when none is given); ``XR_AI_PROFILE`` and then ``DEFAULT_PROFILE`` follow. A path (a
    separator or a ``.json``, ``.yaml`` or ``.yml`` suffix) resolves against ``config_dir``; a name
    resolves to the first existing ``models.<name>`` file with those suffixes, and the default name
    resolves to ``models.json`` or ``models.yaml``. A miss raises :class:`FileNotFoundError` listing
    the candidates.
    """
    chosen = profile or os.environ.get(PROFILE_ENV) or DEFAULT_PROFILE
    if os.sep in chosen or "/" in chosen or chosen.lower().endswith(_PROFILE_SUFFIXES):
        path = Path(chosen).expanduser()
        path = path if path.is_absolute() else config_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"profile file not found: {path}")
        return path
    stem = "models" if chosen == DEFAULT_PROFILE else f"models.{chosen}"
    candidates = [config_dir / f"{stem}{suffix}" for suffix in _PROFILE_SUFFIXES]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"profile {chosen!r} not found; looked for " + ", ".join(str(c) for c in candidates))
