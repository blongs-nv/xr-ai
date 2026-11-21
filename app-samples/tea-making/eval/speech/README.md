<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Real speech replay

This sample-owned five-minute case uses the [devkit real-audio contract](../../../../app-sdk/xr-ai-app/docs/REAL_AUDIO.md). Prepare fixed fixtures once, then reuse them across model and hardware profiles. Models and auxiliary services must already be running.

```bash
uv --config-file uv.toml run --project tests xr-ai-dev prepare app-samples/tea-making/app.py app-samples/tea-making/eval/speech/realtime.yaml --out /path/to/media
uv --config-file uv.toml run --project tests xr-ai-dev run app-samples/tea-making/app.py app-samples/tea-making/eval/speech/realtime.yaml \
    --real-audio --real-models --clock real --media-dir /path/to/media --out /path/to/new-run
```

Use `--profile` for local or hosted model endpoints. See the contract for required image inputs, report interpretation and accuracy-only evaluation. Fixture and run media stay local.

Preparation requires these images of the same tea-making setup:

```bash
--image tea_package.png=/path/to/branded-package.jpg \
--image kettle-water.png=/path/to/water-visible-in-kettle.jpg \
--image kettle-hot.png=/path/to/hot-water-display.jpg \
--image tea-in-water.png=/path/to/tea-touching-water-in-cup.jpg
```

The package must have a legible brand, blend and brewing instructions. The display must show the actual water temperature at or above that package's target with a visible C/F unit, and the cup must show actual tea-water contact. Keep all four images fixed across configurations. The procedure advances from camera evidence; there are no Next commands. Assertions check identification, water visibility, target temperature and timer start. The run stops at five minutes, so steep completion is not asserted.
