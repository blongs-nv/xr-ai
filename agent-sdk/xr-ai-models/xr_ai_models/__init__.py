# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified typed service protocols and model clients for XR AI.

Repository code talks to the typed ``*Service`` protocols. The concrete
``OpenAICompat*`` clients cover chat, embeddings, and HTTP speech;
OCR uses the Nemotron v2 protocol and Riva speech uses optional gRPC.
"""
from ._protocols import (
    Capabilities,
    ChatMessage,
    ChatResponse,
    DecisionAnswer,
    DecisionInputTooLargeError,
    DecisionQuestion,
    DecisionResponse,
    DecisionService,
    ChatStreamEvent,
    ContentPart,
    EmbeddingService,
    ModelServiceError,
    ImageInput,
    ImagePart,
    LLMService,
    OCRService,
    OCRRegion,
    STTService,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolDef,
    TTSService,
    VideoInput,
    VideoPart,
    VLMService,
)
from ._openai_compat import (
    OpenAICompatLLM,
    OpenAICompatEmbedding,
    OpenAICompatSTT,
    OpenAICompatTTS,
    OpenAICompatVLM,
)
from ._config import (
    AdapterSpec,
    Category,
    DecisionSpec,
    DeploymentSpec,
    EmbeddingSpec,
    EndpointSpec,
    LLMSpec,
    ModelsConfig,
    OCRSpec,
    Spec,
    STTSpec,
    TTSSpec,
    VLMSpec,
    load_models_config,
    load_models_config_from_dict,
)
from ._thinking import ThinkingSpec
from ._factory import make_decision, make_embedding, make_llm, make_ocr, make_stt, make_tts, make_vlm
from ._ocr import OCRRequestError, OCRResponseError
from ._riva_grpc import RivaSTT, RivaTTS

__all__ = [
    "ModelServiceError",
    "Capabilities",
    "ChatMessage",
    "ChatResponse",
    "DecisionAnswer",
    "DecisionInputTooLargeError",
    "DecisionQuestion",
    "DecisionResponse",
    "DecisionService",
    "ChatStreamEvent",
    "ContentPart",
    "EmbeddingService",
    "ImageInput",
    "ImagePart",
    "LLMService",
    "OCRService",
    "OCRRegion",
    "OCRSpec",
    "OCRRequestError",
    "OCRResponseError",
    "make_ocr",
    "STTService",
    "TextPart",
    "ToolCall",
    "ToolChoice",
    "ToolDef",
    "TTSService",
    "VideoInput",
    "VideoPart",
    "VLMService",
    "OpenAICompatLLM",
    "OpenAICompatEmbedding",
    "OpenAICompatSTT",
    "OpenAICompatTTS",
    "OpenAICompatVLM",
    "AdapterSpec",
    "Category",
    "DecisionSpec",
    "DeploymentSpec",
    "EmbeddingSpec",
    "EndpointSpec",
    "RivaSTT",
    "RivaTTS",
    "LLMSpec",
    "ModelsConfig",
    "Spec",
    "STTSpec",
    "TTSSpec",
    "VLMSpec",
    "ThinkingSpec",
    "load_models_config",
    "load_models_config_from_dict",
    "make_llm",
    "make_decision",
    "make_embedding",
    "make_stt",
    "make_tts",
    "make_vlm",
]

# Names that left the public surface: still importable, each with a DeprecationWarning that names the
# replacement. `__all__` no longer lists them (docs/source/reference/migrations.md has the before and after).
_KIND = (
    "internal adapter-kind constant; choose the adapter with `kind:` in the profile. "
    "The package imports it from xr_ai_models._config"
)
_CONTROL = (
    "internal literal; set thinking control with ThinkingSpec(control=...). "
    "The package imports it from xr_ai_models._thinking"
)
_DEPRECATED: dict[str, tuple[str, str]] = {
    "KIND_OPENAI_COMPAT": ("._config", _KIND),
    "KIND_SYSTEMONE": ("._config", _KIND),
    "ModelKind": ("._config", _KIND),
    "ThinkingControl": ("._thinking", _CONTROL),
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
