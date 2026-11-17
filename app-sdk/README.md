<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# The app SDK

The app surface for XR AI agents, built beside the existing SDK rather
than inside it. It sits on top of the foundation packages under `agent-sdk/`;
nothing under `agent-sdk/` depends on it.

Three layers: the **SDK** (`xr-ai-app`: the core, the primitives every app is
built from, and the developer tools), **kits** (the other packages, built on
the SDK's public names only: `xr-ai-kit`, the common patterns such as
`Conversation`, `Router` and `Delegate`, with optional LangChain and LangGraph
adapters, and kits for one kind of app), and **your app** (`../app-samples/`
shows one per shape). The SDK is one package
directly on the foundation: its app core is private (`xr_ai_app._base`), and
each core concept (`App`, `Component`, `Participant`, `Session`, `Turn`) is one
class whose public names the package README lists.

| Path | What |
|---|---|
| [`xr-ai-app`](xr-ai-app/README.md) | the SDK: import `xr_ai_app`; the `xr-ai-dev` local device |
| [`xr-ai-kit`](xr-ai-kit/README.md) | the kit: `Conversation` (with its tool loop), `Router` and destinations, `Delegate` and job back ends, import `xr_ai_kit`; procedures (SOPs: steps that watch the camera, a runner, and capturing a demonstration as an evidence packet), import `xr_ai_kit.sop`; LangChain and LangGraph adapters, import `xr_ai_kit.langchain` with the extra `xr-ai-kit[langchain]` ([README](xr-ai-kit/xr_ai_kit/langchain/README.md)) |
| [`../app-samples`](../app-samples/README.md) | apps grouped by the model servers they use: `sdk-tour/` (a numbered walk through the SDK, one small app per piece: voice, vision, monitoring, routing, multi-party, delegation, tracing, procedure guidance and capture), `simple-vlm`, `conversation`, `lab-instruments` |

Each sample is its own uv project: its `pyproject.toml` names the packages
here (and from PyPI) that it needs, and `uv run ...` from the sample's
directory (or from its group's, for `app-samples/sdk-tour`) runs it (see
`app-samples/README.md`). Tests live
with the rest of the suite in `tests/app/surface/`.

Status: a preview. Everything here runs end to end on scripted models in the
tests, and every sample also runs on DeviceIOHub with real models (the
twelve in `app-samples/sdk-tour` with `uv run sdk-tour <sample>`).
