<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Model profiles and deployment

## Overview

How an `xr_ai_next` application finds its models: the profile file, how a profile is selected, the model servers a
profile can name, and how to point a profile at hosted or NIM endpoints. This page is the single place for the
profile file format; {doc}`choosing-models` says which model to put in it, and {doc}`deploying-with-nim` covers
operating NIM endpoints. Samples that have not moved to `xr_ai_next` still use `yaml/models.json` with
`model-servers/` launchers, documented in {doc}`customizing-model-servers`.

## The profile file

A profile is JSON or YAML named `models.<name>.json` (or `.yaml`, `.yml`) beside the sample's `yaml/app.yaml`. The
`models` mapping holds one entry per role; the optional keys `profile`, `roles` and `budgets` add metadata, role
sharing and per-role limits. A trimmed example of a `models.local.json`:

```json
{
  "profile": {"name": "local", "description": "Everything on one GPU."},
  "models": {
    "perceive": {
      "category": "vlm",
      "adapter": {"kind": "openai_compat", "model_name": "llm"},
      "endpoint": {"base_url": "http://localhost:8108"},
      "deployment": {"ownership": "managed", "service": "omni"}
    }
  },
  "budgets": {"perceive": {"max_concurrency": 2}}
}
```

- `adapter` selects the wire protocol, `endpoint` says where the server is, and `deployment` says who starts it.
- `deployment.service` names the server an entry runs on; `deployment.ownership` is `managed`, `reused` or
  `external` (the default).
- `roles` maps a role to another entry so one server can fill several roles.
- `budgets` caps `max_concurrency` per role so background work cannot starve the foreground turn.

## How a profile is selected

The first of these that is set wins:

1. `--profile <name-or-path>` on the command line.
2. The `XR_AI_PROFILE` environment variable.
3. `profile:` in `app.yaml`.
4. The default name `models`, which is the older `models.json`.

A value with a path separator or a `.json`, `.yaml` or `.yml` suffix is a path, relative to the directory of
`app.yaml`. Any other value is looked up as `models.<name>.json`, `.yaml` or `.yml` in that directory. Refer to {doc}`/reference/configuration` for the generated field
reference.

## Model servers

`deployment/services.yaml` is the registry of servers, their ports and their launch commands; a profile entry names a
server in `deployment.service`. GPU placement for locally started servers comes from three shared presets in
`deployment/gpu-profiles/`: `dual_48G_ada`, `96G_blackwell` and `spark`.

The optional servers include `nano` (a text model for `answer`), `magpie` (Magpie TTS for `speech_out`), `stt`
(batch speech to text for `speech_in`) and `ocr`, a container whose image is built once (refer to
{ref}`ocr-serving`).

## Hosted and NIM endpoints

A `hybrid` profile points `answer` at a hosted endpoint, with `api_key_env` naming the environment variable that
holds the key (never put the key in the file):

```json
{
  "answer": {
    "adapter": {"kind": "openai_compat", "model_name": "nvidia/llama-3.3-nemotron-super-49b-v1.5"},
    "endpoint": {"base_url": "https://integrate.api.nvidia.com/v1", "api_key_env": "NVIDIA_API_KEY"}
  }
}
```

A `nim` profile has `external` entries for every role, so no model server runs locally for it. Deploy and validate the NIM endpoints first; {doc}`deploying-with-nim` explains ownership, credentials
and the smoke check.

## Related pages

- {doc}`choosing-models`: roles, categories and the scorecard.
- {doc}`/components/ai-services`: the bundled model servers and OCR serving.
- {doc}`customizing-model-servers`: the earlier `model-servers/` launchers.
