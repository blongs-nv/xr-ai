<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# lab instruments model servers

This launcher starts the same local inference stack as upstream's
`agent-samples/lab-instrument-monitoring`: speaker ASR, Pocket TTS,
Nemotron-3 Nano Omni and the Cosmos VLM. It runs that sample's model-server
configuration
(`agent-samples/lab-instrument-monitoring/model-servers/yaml/<gpu-profile>.yaml`)
unchanged, under the same stack name, so servers started by either launcher
are reused or stopped by the other.

Refer to the
[model-server workflow](https://nvidia.github.io/xr-ai/latest/guides/customizing-model-servers.html)
for startup, shutdown, switching samples, and configuration composition.

## Run

Run all commands from `xr-ai-next/samples/lab-instruments/`:

```bash
uv run --project model-servers lab_instruments_model_servers
```

Wait for the launcher to report that all processes are ready and return. Use
`--gpu-profile NAME` for an explicit profile on both start and stop. Stop the
servers before switching samples or editing server settings:

```bash
uv run --project model-servers lab_instruments_model_servers --stop
```
