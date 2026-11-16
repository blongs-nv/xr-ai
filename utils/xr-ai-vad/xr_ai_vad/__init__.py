# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Silero-VAD utterance detector for xr-ai agent workers.

Consumes int16 LE PCM bytes and emits int16 PCM utterance bytes via an
async callback when speech ends.
"""
__all__ = []

# Names that left the public surface: still importable, each with a DeprecationWarning that names the
# replacement. `__all__` no longer lists them (docs/source/reference/migrations.md has the before and after).
_NOTE = (
    "internal to the voice pipeline; configure voice in the voice: section of an xr_ai_app app.yaml "
    "(VoiceAgent builds the VAD). "
    "The voice package imports it from xr_ai_vad.detector"
)
_DEPRECATED: dict[str, tuple[str, str]] = {
    "VadDetector": (".detector", _NOTE),
    "OnUtteranceCb": (".detector", _NOTE),
    "OnSpeechStartCb": (".detector", _NOTE),
}


def __getattr__(name: str) -> object:
    entry = _DEPRECATED.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import warnings  # noqa: PLC0415

    module, replacement = entry
    # `from pkg import name` first probes with hasattr(); importlib calls it from a `<frozen importlib...` frame. Skip
    # the warning for that probe only: the real lookup that follows warns, so an import warns once, not twice.
    if not sys._getframe(1).f_code.co_filename.startswith("<frozen importlib"):
        warnings.warn(f"{__name__}.{name} is deprecated: {replacement}", DeprecationWarning, stacklevel=2)
    return getattr(importlib.import_module(module, __package__), name)


def __dir__() -> list[str]:
    return sorted([*globals(), *_DEPRECATED])
