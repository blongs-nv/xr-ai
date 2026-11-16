<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# tea_visual

Tea visual checks on frames, using the shipped tea guide questions. Run with
`xr-ai-eval run --suite tea_visual --evals-dir evals/suites --profile <profile>`.

PR 711 keeps the visual oracles in its procedure-capture-guide tea evaluation fixtures
(`example-guides/tea-making/evaluation/fixtures/visual_expectations.yaml` and `image_scenes.yaml`); its
`tea-making-sample/eval/cases.yaml` is routing-only and has no visual cases. The scenes here are ported from
those fixtures (read them with `git show pr711:<path>`).

Not ported from PR 711: `breakfast`, `green` (more other-blend packets), `current20`, `current99` (more
below-target readings) and `set100` (a SET label; `target_only` covers the TARGET case).
