<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Eval schema

Implemented by the runner (`evals/runner`) and filled by the suites and the sample
migrations. One contract replaces six eval layouts: cases in YAML, a `--profile` flag, results in JSONL.

## Result record (JSONL)

One JSON object per line, one line per case per run. Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `suite` | string | Suite name, e.g. `tea-visual`; equals the directory under `evals/` |
| `case` | string | Case id, unique within the suite |
| `role` | string | Job-named role under test (`perceive`, `decide`, ...; see `PROFILES_SCHEMA.md`) |
| `preset` | string | Model preset or model name that filled the role |
| `profile` | string | Profile name used (`local`, `hybrid`, `nim`, or a path stem) |
| `pass` | boolean | Whether `score` met the case's `pass_threshold` |
| `score` | number | 0.0 to 1.0 |
| `latency_ms` | number | Wall time of the model call(s) for the case |
| `tokens` | object | `{"input": int, "output": int}`; `null` values when the endpoint does not report usage |
| `gpu_profile` | string or null | GPU placement preset (`dual_48G_ada`, `96G_blackwell`, `spark`); `null` for hosted runs |
| `commit` | string | Git commit of the repo that ran the eval |

Optional fields: `schema_version` (integer, currently `1`), `run_id`, `timestamp` (UTC ISO 8601), `error`
(string; set when the case errored, with `pass: false` and `score: 0`), `detail` (free-form object with the
scorer's evidence, for example the model output), `gpu_memory_mb`.

A case that errors is still one record. Readers must ignore unknown fields. Example:

```json
{"schema_version": 1, "suite": "tea-visual", "case": "kettle-boiling-positive", "role": "perceive", "preset": "cosmos3_nano_reasoner", "profile": "local", "pass": true, "score": 1.0, "latency_ms": 812.4, "tokens": {"input": 1530, "output": 41}, "gpu_profile": "dual_48G_ada", "commit": "62bc3920"}
```

## Case file (YAML)

Layout: `evals/<suite>/cases/*.yaml` with fixtures beside them in `evals/<suite>/fixtures/`.

```yaml
suite: tea-visual              # required; equals the directory name
version: 1                     # required; bump when cases change meaning
description: Recorded-frame checks for the tea steps
defaults:                      # optional values applied to every case
  role: perceive
  pass_threshold: 1.0
cases:
  - id: kettle-boiling-positive          # required, unique in the suite
    role: perceive                        # required unless set in defaults
    input:
      text: "Is the kettle boiling?"     # prompt or user turn
      frames: [fixtures/kettle_boil_01.jpg]   # optional, paths relative to the case file
      context: []                         # optional prior messages: [{role: user, text: ...}]
    system: null                          # optional; case-level system prompt, sent first
    scorer:
      type: choice                        # exact | contains | regex | choice | json_match | judge | python
      expect: positive                    # scorer-specific; for choice, one of the case's choices
      choices: [positive, negative, unknown]
    pass_threshold: 1.0                  # optional, 0.0 to 1.0; default 1.0
    tags: [visual, kettle]                # optional; used to select subsets
    skip_reason: null                     # optional; a skipped case writes no record
```

Scorer types: `exact` (string equality after trimming), `contains` (every `expect` substring present),
`regex` (`expect` pattern matches), `choice` (a decision role returns the expected choice),
`json_match` (structured output equals or includes the expected fields; partial score per field), and
`judge` (a named role scores 0 to 1 against `rubric`; the judge's role and preset go in `detail`).

`python` (escape hatch, gap theme M): `scorer: {type: python, function: "scorers.py:wer", ...}`; the file is
resolved relative to the case file and loaded by path. `function(case, result) -> score | (score, evidence)`
where `case` is the loaded case (`case.scorer` carries any extra keys, such as the reference text) and
`result` has `text` (the last reply), `replies`, `latency_ms` and token counts. No per-sample scorer types.

Input extensions (the model call, in this precedence):

- `input.driver: "drivers.py:fn"`: `fn(endpoint, case) -> str` makes the call itself (sync or async) and its
  text is scored. For transports the runner does not speak (a private WebSocket).
- `input.params: {...}`: free-form mapping handed to drivers and python scorers as `case.params` (paths in it
  are relative to the case file; the hook resolves them).
- `input.audio: fixtures/clip.wav`: posted to the role's `/v1/audio/transcriptions`; the returned `text` is the output.
- `input.turns: [{text, frames}, ...]` (instead of `text`/`frames`): sent in order, each reply appended to the
  history as an assistant message; the last reply is scored, `result.replies` holds all. Latency and tokens sum.
- `input.frames` order is as listed: earlier frames first, after the text.
- A case with `skip_reason` may name fixtures that do not exist yet; a missing fixture on a live case is an
  invalid case (exit 2).

Rules from `AGENTS.md` carry over: a rule in a sample prompt needs a case, and fixtures must not reuse
the prompt's worked-example specifics.

## Runner contract

- `xr-ai-eval run --suite <name> --profile <name> [--out results.jsonl] [--tags a,b] [--roles perceive,decide]` (an `xr-ai eval` alias is deferred).
- Exit code 0 when every executed case passed, 1 when any failed, 2 for invalid cases or schema errors.
- Results append to `--out` (default `evals/results/<UTC date>-<commit>.jsonl`); baselines live in
  `evals/baselines/<suite>.<profile>.jsonl` and use the same schema.
- Matrix mode runs each suite against several profiles and aggregates `(suite, role, preset)` pass rate,
  median `latency_ms` and tokens into the scorecard page.
