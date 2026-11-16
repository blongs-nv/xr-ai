<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Model servers for the SDK tour samples

One launcher for every sample in `sdk-tour/` (`01` to `12`). Their shared
profile, `../models.json`, names upstream `agent-samples/conversation-example`'s
models (Cosmos on :8100 for `perceive`, Nemotron-3 Nano Omni on :8108 for
the two `answer_*` roles, Clef on :8120 for `decide`, Pocket TTS on :8105 for
`speech_out`, speaker ASR on :8102 for `speech_in`), so this launcher starts
exactly that sample's stack: it runs
`agent-samples/conversation-example/model-servers/yaml/<gpu-profile>.yaml`
unchanged, under the same stack name, so servers started by either launcher
(or by `../../conversation/model-servers`) are reused or stopped by the other.

Refer to the
[model-server workflow](https://nvidia.github.io/xr-ai/latest/guides/customizing-model-servers.html)
for startup, shutdown, switching samples, and configuration composition.

## Run

From `app-samples/sdk-tour/` (once for all twelve):

```bash
uv run --project model-servers sdk_tour_model_servers
```

Wait for the launcher to report that all processes are ready and return. Use
`--gpu-profile NAME` for an explicit profile on both start and stop. Stop the
servers before switching to a sample with another stack or editing server
settings:

```bash
uv run --project model-servers sdk_tour_model_servers --stop
```
