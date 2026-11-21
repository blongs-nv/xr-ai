<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Speech replay

`realtime.yaml` schedules the existing [`guide.yaml`](../../scenarios/guide.yaml) case through real STT/TTS over five minutes. Prompts, media and assertions come from that case. This file adds timing, voice controls and reply expectations; it contains no separate task definitions or scene generator.

Use the [real-audio guide](../../../../app-sdk/xr-ai-app/docs/REAL_AUDIO.md) to prepare pinned speech and run the replay. Preparation copies the case's existing media. Recorded replacements can be supplied using the same media names; a generated placeholder remains a placeholder and is not evidence of real-scene accuracy.

Source images retain their streamed or on-demand camera mode during speech replay.
