<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-next

The next app surface for XR AI agents, built beside the existing SDK rather
than inside it. It sits on top of the foundation packages under `agent-sdk/`;
nothing under `agent-sdk/` depends on it.

Three layers: the **SDK** (`packages/xr-ai-next`: the core, the primitives
every app is built from, and the developer tools), **kits** (the other packages,
built on the SDK's public names only: `packages/xr-ai-kit`, the common patterns
such as `Conversation`, `Router` and `Delegate`, and kits for one kind of app or
one framework), and **your app** (`samples/` shows one per shape). The SDK is one package
directly on the foundation: its app core is private (`xr_ai_next._base`), and
each core concept (`App`, `Component`, `Participant`, `Session`, `Turn`) is one
class whose public names the package README lists.

| Path | What |
|---|---|
| [`packages/xr-ai-next`](packages/xr-ai-next/README.md) | the SDK: import `xr_ai_next`; the `xr-ai-dev` local device |
| [`packages/xr-ai-kit`](packages/xr-ai-kit/README.md) | the kit: `Conversation` (with its tool loop), `Router` and destinations, `Delegate` and job back ends, import `xr_ai_kit` |
| [`packages/xr-ai-langchain`](packages/xr-ai-langchain/README.md) | kit: LangChain and LangGraph adapters, import `xr_ai_langchain` |
| [`packages/xr-ai-procedures`](packages/xr-ai-procedures/README.md) | kit: procedures (SOPs), capture a demonstration as an evidence packet and guide someone through an approved one, import `xr_ai_procedures` |
| [`samples`](samples/README.md) | apps grouped by the model servers they use: `sdk-tour/` (a numbered walk through the SDK, one small app per piece: voice, vision, monitoring, routing, multi-party, delegation, tracing, procedure guidance and capture), `simple-vlm`, `conversation`, `lab-instruments` |

Each sample is its own uv project: its `pyproject.toml` names the packages
here (and from PyPI) that it needs, and `uv run ...` from the sample's
directory (or from its group's, for `samples/sdk-tour`) runs it (see
`samples/README.md`). Tests live
with the rest of the suite in `tests/next/surface/`.

Status: a preview. Everything here runs end to end on scripted models in the
tests, and every sample also runs on DeviceIOHub with real models (the
twelve in `samples/sdk-tour` with `uv run sdk-tour <sample>`).
