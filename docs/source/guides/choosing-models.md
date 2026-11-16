<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Choosing models per role

## Overview

Applications ask for a model by job, and a profile says which model fills each job. This guide lists the roles, the
categories of model that may fill them, the three standard profiles, and how to read the scorecard that compares
models on the same cases. Use it to decide what to put in `yaml/models.<profile>.json` for your application.

:::{important}
The scorecard has not been measured yet. No baselines are recorded in `evals/baselines/` (the runs need a free GPU
host), so every number in this guide is marked "not yet measured". Do not choose a model from this page on
performance grounds until the table is filled by `xr-ai-eval report`.
:::

## Roles

| Role | Job | Categories that may fill it | Earlier name |
| --- | --- | --- | --- |
| `perceive` | Describe or question what a frame shows | `vlm` | `vlm` |
| `decide` | Pick one of N choices (intent, routing, judge) | `decision` | `decision` |
| `extract` | Turn text or an image into structured fields | `llm`, `vlm` | none |
| `answer` | Conversational reply and tool loop | `llm` | `llm` |
| `ack` | Short acknowledgment while the answer is produced | `llm` | `acknowledgment_llm` |
| `embed` | Text embeddings for retrieval | `embedding` | `embedding` |
| `speech_in` | Speech to text | `stt` | `stt` |
| `speech_out` | Text to speech | `tts` | `tts` |

An application names only the roles it uses: a camera-question app might need `perceive`, `decide`, `speech_in`
and `speech_out`, and a conversation adds `answer` and `ack`. Other role names are allowed for sample-specific jobs.
A profile can fill several roles with one entry through its `roles` map. An entry whose category does not match
the role fails when the profile loads, with the file and role name in the message.

<!-- docs-snippet: run group=roles cwd=yaml -->
```python
from xr_ai_next import App, PerConnection

app = App.from_config("app.yaml", profile="local", enrollment=PerConnection(admit=None))
print(app.role("decide").name)
try:
    app.role("not-a-role")
except KeyError as error:
    print(error)  # names the profile file and lists the roles it defines
```

## Standard profiles

| Profile | Meaning | Choose it when |
| --- | --- | --- |
| `local` | Every model on one Spark or RTX PRO 6000 (or two 48 GB GPUs); the launcher starts them | You have the GPU and want nothing hosted |
| `hybrid` | `answer` and `extract` on a hosted large model; perception, decisions, speech and embeddings local | Recommended: the best language model without giving up local perception |
| `nim` | NIM endpoints for every role; nothing is started | Models are deployed and operated elsewhere |

`hybrid` needs `NVIDIA_API_KEY` for the hosted entry. A sample without a hosted role (one that only
perceives and decides locally) has a `hybrid` file equal to `local`. Refer to {doc}`profiles-and-deployment` for the
file format, selection order and managed servers.

## How to decide

1. Start from the standard profile that matches where the models run.
2. Run the eval suite for the role you are changing on both candidates, with the same profile and GPU placement.
3. Prefer the model with the higher pass rate on the suite closest to your use; break ties on latency and GPU memory.
4. Record a role's known limitation on the scorecard rather than working around it in the prompt.

## Read the scorecard

`xr-ai-eval report` writes one row per suite, role and model, from the JSONL results that `xr-ai-eval run` or
`matrix` produced. Columns:

| Column | Meaning |
| --- | --- |
| Suite, role, model | The cases run, the role under test, and the preset or model that filled it |
| Pass rate | Cases that met their pass threshold, over cases run |
| Errors | Cases that errored (counted as failed) |
| Median latency | Wall time of the model call, over cases without an error |
| Median tokens | Input and output tokens, when the endpoint reports usage |
| Median GPU memory | Host-wide used GPU memory in MiB right after the call (includes other processes) |

Pass rates are comparable only for the same suite version, profile and GPU placement. Cases skipped with a
`skip_reason` write no record and do not appear.

## The scorecard

| Suite | Role | Model | Pass rate | Median latency | Median GPU memory |
| --- | --- | --- | --- | --- | --- |
| `tea-visual` | `perceive` | not yet measured | not yet measured | not yet measured | not yet measured |
| `speech` | `speech_in` | not yet measured | not yet measured | not yet measured | not yet measured |
| `conversation-context` | `answer` | not yet measured | not yet measured | not yet measured | not yet measured |

## Regenerate the scorecard

On a GPU host with no other model stack running (the procedure and the layout of `evals/baselines/` are in
`evals/baselines/README.md`):

```bash
xr-ai-eval matrix --suite tea-visual --suite conversation-context \
    --profiles local,hybrid --presets cosmos3_nano_reasoner,nemotron_omni --out matrix.jsonl
xr-ai-eval report --results matrix.jsonl --out scorecard.md
```

`matrix` runs every suite against every profile and, with `--presets`, requests each preset as the model name in
place of the profile's own. The report is deterministic: the same results produce the same file. Paste the table
from `scorecard.md` over the placeholder rows above, and record the commit and GPU profile beside it. Refer to
{doc}`/reference/configuration` for the profile fields.
