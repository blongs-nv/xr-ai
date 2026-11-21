<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Five-minute speech performance

`realtime.yaml` drives the app's conversation, camera questions, notes and visual
alert while its real STT and TTS run. Enrollment/stop phrases are not expected to
reach the app handler. Use the same prepared inputs and timeline for each profile and establish a
baseline before comparing model replacements.

Each `audio.path` is the SHA256 of its `text`, followed by `.wav`. Supply those
recordings or synthesize them once using a chosen TTS service, then keep exactly
the same WAVs for every configuration. The files themselves are hashed by the
runner; a filename is not a content hash. The camera inputs are `blue.png` and
`red.png`, each showing a single filled rectangle in that color. Keep fixtures
under the ignored `media/` directory or pass an external `--media-dir`.

From the repository root, with the selected model services already running:

```bash
uv --config-file uv.toml run --project tests xr-ai-dev run app-samples/conversation/app.py \
    app-samples/conversation/eval/speech/realtime.yaml \
    --real-audio --real-models --clock real \
    --media-dir /path/to/fixed-fixtures --out /path/to/new-run
```

Use `--profile NAME_OR_PATH` for another local or hosted profile. Accuracy-only
scenarios remain under `../../scenarios/`; their typed turns bypass STT. See the
[real audio guide](../../../../app-sdk/xr-ai-app/docs/REAL_AUDIO.md) for the report,
source scheduling, supported actions and measurement limits.

Prepare reusable fixtures with `xr-ai-dev prepare app-samples/conversation/app.py app-samples/conversation/eval/speech/realtime.yaml --out /path/to/media`. The scenario generates its rectangle images; preparation synthesizes and pins its speech clips.
