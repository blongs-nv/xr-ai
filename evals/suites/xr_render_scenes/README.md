<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr_render_scenes

69 scenario cases of the xr-render supervisor, generated from the original corpus (`python generate_cases.py`). Run with `xr-ai-eval run --suite xr_render_scenes --evals-dir evals/suites --profile local`; only the `answer` role is a real model.

Not standalone: `drivers.py` imports `xr_render_demo_eval` (the original harness) and, when it is not installed, adds `agent-samples/xr-render-demo/eval` to `sys.path`; the scorer reuses the harness's `check_corpus`.

The driver differs from the harness's `run_corpus_case`: a supervisor exception is a case error (the harness scores it as a failed case), and the LLM client is built from the runner's endpoint (preset, else `openai_compat` with the endpoint's model name), so options in the original `yaml/models.json` apply only when the profile entry carries them.
