<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Real speech replay

This sample-owned five-minute case uses the [devkit real-audio contract](../../../../../app-sdk/xr-ai-app/docs/REAL_AUDIO.md). Prepare fixed fixtures once, then reuse them across model and hardware profiles. Models and auxiliary services must already be running.

```bash
uv --config-file uv.toml run --project tests xr-ai-dev prepare app-samples/sdk-tour/10-guided-procedure/app.py app-samples/sdk-tour/10-guided-procedure/eval/speech/realtime.yaml --out /path/to/media
uv --config-file uv.toml run --project tests xr-ai-dev run app-samples/sdk-tour/10-guided-procedure/app.py app-samples/sdk-tour/10-guided-procedure/eval/speech/realtime.yaml \
    --real-audio --real-models --clock real --media-dir /path/to/media --out /path/to/new-run
```

Use `--profile` for local or hosted model endpoints. See the contract for generated image names, report interpretation and accuracy-only evaluation. Fixture and run media stay local.

Preparation generates three deterministic geometric printer scenes: an orange object on the bed with the door open, an empty bed with the door open, and an empty bed with the door closed. The camera, printer geometry and lighting stay fixed. Door angle and bed occupancy supply the visual evidence; the pictures contain no answer labels. The case checks the spoken start, bed-clear confirmation and door-closed confirmation while questions and a warm-up timer run on the fixed timeline.

For recorded-camera qualification, replace the generated pictures with `--image printer-bed-occupied.png=PATH`, `--image printer-bed-clear-door-open.png=PATH`, and `--image printer-bed-clear-door-closed.png=PATH`. Use the same printer in each state. Results with generated scenes measure those scenes and do not establish performance on real printer photographs.
