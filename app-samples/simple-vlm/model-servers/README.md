<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# simple-vlm model servers

This launcher starts the same local inference stack as upstream's
`agent-samples/simple-vlm-example`: speaker ASR, Pocket TTS and the Cosmos VLM.
It runs that sample's model-server configuration
(`agent-samples/simple-vlm-example/model-servers/yaml/<gpu-profile>.yaml`)
unchanged, under the same stack name, so servers started by either launcher
are reused or stopped by the other.

Refer to the
[model-server workflow](https://nvidia.github.io/xr-ai/latest/guides/customizing-model-servers.html)
for startup, shutdown, switching samples, and configuration composition.

## Run

Run all commands from `app-samples/simple-vlm/`:

```bash
uv run --project model-servers simple_vlm_model_servers
```

Wait for the launcher to report that all processes are ready and return. Use
`--gpu-profile NAME` for an explicit profile on both start and stop. Stop the
servers before switching samples or editing server settings:

```bash
uv run --project model-servers simple_vlm_model_servers --stop
```
