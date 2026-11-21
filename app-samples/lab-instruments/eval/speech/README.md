<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Real speech replay

This sample-owned five-minute case uses the [devkit real-audio contract](../../../../app-sdk/xr-ai-app/docs/REAL_AUDIO.md). Prepare fixed fixtures once, then reuse them across model and hardware profiles. Models and auxiliary services must already be running.

```bash
uv --config-file uv.toml run --project tests xr-ai-dev prepare app-samples/lab-instruments/app.py app-samples/lab-instruments/eval/speech/realtime.yaml --out /path/to/media
uv --config-file uv.toml run --project tests xr-ai-dev run app-samples/lab-instruments/app.py app-samples/lab-instruments/eval/speech/realtime.yaml \
    --real-audio --real-models --clock real --media-dir /path/to/media --out /path/to/new-run
```

Use `--profile` for local or hosted model endpoints. See the contract for generated image names, report interpretation and accuracy-only evaluation. Fixture and run media stay local.
