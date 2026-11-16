<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Profiles schema

Implemented by `xr_ai_app._base.profiles` (and the launcher) and read by
`App`. The constants `ROLE_CATEGORIES` and `DEFAULT_PROFILE` in
`xr_ai_app._base.profiles` are the executable form of the tables below.

## Roles

Roles are named by job. A role is a key in the profile's `models` mapping (or a `roles` key that points at one);
several roles may point at one endpoint.

| Role | Job | Categories that may fill it | Today's name |
| --- | --- | --- | --- |
| `perceive` | Describe or question what a frame shows | `vlm` | `vlm` |
| `decide` | Pick one of N choices (intent, routing, judge) | `decision` (Clef's `systemone` kind, or an LLM adapter: `"category": "decision", "adapter": {"preset": "nemotron_omni"}`, whose `adapter.templates.decision` may set the prompt the LLM reads) | `decision` |
| `extract` | Turn text or an image into structured fields | `llm`, `vlm` | (cloned `vlm` or `llm` specs in code) |
| `answer` | Conversational reply, tool loop | `llm` | `llm` |
| `ack` | Short acknowledgment while the answer is produced | `llm` | `acknowledgment_llm` |
| `embed` | Text embeddings for retrieval | `embedding` | `embedding` |
| `speech_in` | Speech to text | `stt` | `stt` (and the speaker STT endpoint in `voice_gate.yaml`) |
| `speech_out` | Text to speech | `tts` | `tts` |

Other role names are allowed (the schema already accepts any name), for sample-specific roles such as
a second reader of the same VLM. The eight names above are reserved for these jobs.

## Profile file format

A profile is YAML or JSON. The `models` mapping is exactly today's `models.json` schema and is parsed
by the unchanged `xr_ai_models.load_models_config`. The other top-level keys are optional and are
ignored by that loader, which only reads `models` when the key is present.

```yaml
profile:                       # optional metadata
  name: hybrid
  description: answer and extract on a hosted large model; perception and decisions local
  gpu_profile: dual_48G_ada    # shared GPU placement preset: dual_48G_ada | 96G_blackwell | spark
models:                        # required; same schema as models.json today
  perceive:
    adapter: {preset: cosmos3_nano_reasoner}
    endpoint: {base_url: "http://localhost:8100"}
    deployment: {ownership: managed, service: vlm-server}
  decide:
    adapter: {kind: systemone}      # Clef decision server
    endpoint: {base_url: "http://localhost:8110"}
    deployment: {ownership: managed, service: clef-server}
  answer:
    adapter: {kind: openai_compat, model_name: "nvidia/llama-3.3-nemotron-super-49b-v1.5"}
    endpoint: {base_url: "https://integrate.api.nvidia.com/v1", api_key_env: NVIDIA_API_KEY}
  speech_out:
    adapter: {preset: pocket_tts}
    endpoint: {base_url: "http://localhost:8105"}
    deployment: {ownership: managed, service: pocket-tts}
roles:                         # optional: role -> entry name in `models`
  extract: answer
  ack: answer
budgets:                       # optional: per-role inference budget
  perceive: {max_concurrency: 2}
```

Rules:

1. `models` entries follow the existing structured form (`adapter`, `endpoint`, `deployment`). Nothing
   about an entry changes; only the role names the app asks for change.
2. `roles` maps a role name to another entry name. It exists so one entry can fill several roles without
   being repeated. A role that is both an entry and a `roles` key is an error.
3. `budgets` sets `max_concurrency` (in-flight calls per role) so background work cannot starve the
   foreground turn. Missing means unlimited.
4. `deployment.ownership: managed` marks the entries the launcher starts, by `deployment.service`,
   once per service even when several roles share it. `external` (default) and `reused` are never started.
5. Category checks: when an entry fills a standard role, its resolved category must be one of the
   categories in the table above; a mismatch fails at load time with the file path and role name.
6. `profile.gpu_profile` names one of three shared GPU placement presets that replace the 18
   per-sample files. Hosted-only profiles omit it.

### Standard profiles

| Profile | File | Meaning |
| --- | --- | --- |
| `local` | `models.local.json` | Everything on one Spark or RTX PRO 6000 |
| `hybrid` | `models.hybrid.json` | `answer` and `extract` on a hosted large model; perception and decisions local; the recommended path |
| `nim` | `models.nim.json` | NIM endpoints; replaces the script in `deploying-with-nim.md` |

## How `--profile` picks a file

`App.run` and `App.from_config` resolve the profile name with this precedence (highest first):

1. `--profile <name-or-path>` on the command line (or `profile=` passed to `App.from_config`).
2. The `XR_AI_PROFILE` environment variable.
3. `profile:` in `app.yaml`, a default that explicit and environment choices outrank.
4. `DEFAULT_PROFILE` (`models`), which resolves to today's `models.json`.

A value containing a path separator or ending in `.json`, `.yaml` or `.yml` is a path (relative paths
resolve against the directory of `app.yaml`). Any other value `<name>` resolves to the first existing
file of `models.<name>.json`, `models.<name>.yaml`, `models.<name>.yml` in the directory of `app.yaml`;
none found is an error that lists the candidates.

## Backward compatibility with today's `models.json`

- `app.role(name)` resolves `name` to a `roles` key, else to an entry of that exact name; there are no
  legacy aliases. A `models.json` with modality-named entries (`vlm`, `llm`, `tts`, ...) and no
  `roles` map still loads, but those entries fill a standard role only when named for it: add a
  `roles` map (`perceive: vlm`) or rename the entry. `app.role("vlm")` still works as the entry name.
- A role bound to a speech entry (`speech_in`, `speech_out`: categories `stt`, `tts`) has no chat
  methods: `ask`, `stream` and `decide` on it raise a `TypeError` that says so.
- The direct (unwrapped) form of `models.json`, without the `models:` key, keeps loading for old
  callers, but `roles`, `profile` and `budgets` need the wrapped form.
- The speaker STT endpoint that `voice_gate.yaml` holds today is read as a fallback for `speech_in`
  with a deprecation warning; once a profile declares `speech_in`, the voice-gate key is ignored.
- Existing `xr_ai_models` public names and `load_models_config` are unchanged.
