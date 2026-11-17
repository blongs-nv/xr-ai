<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Evaluating the tea guide

Two suites, one per layer:

| Suite | What it evaluates | Cases | Script |
|---|---|---|---|
| Steps | each step on its own: the real step, on a recording and a scripted person | `steps/<step id>.yaml` | `run_steps.py` |
| Dialogue | the guide around the steps: what the user says during each step | `dialogue/<step>.yaml` | `run_dialogue.py` |

```text
steps/               one case file per step: scenes, person, expectations
dialogue/            one packet per step state: utterances and expectations
  common.yaml        utterances asked during every step; step words
recordings/
  green-tea-04.yaml  what the recording shows, when, and where the times came from
run_steps.py         builds the steps with procedure.build_steps; xr_ai_kit.sop.eval runs them
run_dialogue.py      the Router, the guide's controls and the Conversation, in a given state
```

## Step cases

```yaml
step: heat_water
build: {temperature_c: 80}          # passed to procedure.build_steps
cases:
  - name: unit_unknown_then_answered
    scene: [26, 346]                # seconds of the recording
    person: {answers: [Fahrenheit]}
    expect: {asks: [unit], completes: {after: 330}}
  - name: looked_away
    scene: [[26, 36], [5, 12], [5, 12], [5, 12]]   # parts played back to back
    build: {unit: fahrenheit}       # this case's own build values
    person: {after_tell: [[36, 80]]}               # what the camera shows once told
    expect: {says: [kettle, display], completes: false}
```

Each case runs the step that ships (built by `procedure.build_steps`, as
the procedure builds it) on the devkit's virtual clock: frames from the
scene at the recording's pace, the person's answers after
`answer_after_s` (3 s), idle waits skipped, model calls in real time (so
their latency counts as it would live). The case ends when the step
finishes or the scene does.

**Scenes** are times in the recording: `[start, end]` plays one part as
recorded; a list of parts plays them back to back, in any order and
repeated, to build what the recording never had (the person looking away
for a minute while the kettle heats). Scene time starts at the first
part's start, so for one part it is the recording's own time.

**Person**: `answers` (to the step's questions, in order; beyond them,
none), `answer_after_s`, `after_tell` (the scene parts shown once the
step told the person to do something).

**Expectations** belong to the step's type (`xr_ai_kit.sop.steps`), so
each type is scored by its own rules:

| Step type | `expect` keys |
|---|---|
| watch (`fill_water`, `heat_water`, `start_steeping`) | `completes`: true / false / `{after, before}` in scene time; `asks`: words its question must include (`[]`: any); `says`: the same for an instruction; `quiet`: says nothing |
| read (`identify`) | `completes`; `reads` / `derives`: per field, words its value must include (`[]`: not empty) |
| wait (`steep`) | `completes` |

A case without `expect` is run and reported only (ambiguous footage); give
it a `why`.

## Dialogue cases

A packet puts the guide in a state and lists what the user may say:

```yaml
step: heat_water
state:
  steps: {identify: done, fill_water: done, heat_water: running}
  detail: "running: 0 of 2 confirmations ..."   # the running step's latest report
  asking: "..."            # optional: a question the step is waiting on
  noted: {temperature_c: 80, duration_s: 120}
utterances:
  - say: What temperature are we waiting for?
    expect: {route: conversation, answer: {mentions: ["80", "176"]}}
```

Each utterance goes through the app's own decisions, performing nothing:
the Router's choice (`Router.choose`), the guide's control decision
(`SopRunner.classify`; a non-control passes the turn on, as live) and the
Conversation's answer.

| `expect` key | Meaning |
|---|---|
| `route` | where the turn ends up: `tea` (a control was performed), `observation` (the camera now), `conversation`; a list allows any |
| `control` | the control asked for: `start`, `stop`, `skip`, `restart`, `answer` (a reply to the guide's pending question), or `none` (the default); a list allows any |
| `answer.mentions` | the conversation's answer contains at least one of these |
| `answer.not` | it contains none of these |

In `common.yaml`, `{current}` and `{next}` are the packet's step and the
next one, and in an answer check a step id stands for any of its
`step_words`. So "What's the next step?" during `fill_water` must be
answered (not acted on) with words of `heat_water`.

## Adding cases

- Something a step must handle: one case in `steps/<step>.yaml`; a scene
  the recording lacks is spliced from its parts.
- Something the user may say during one step: one entry in that packet's
  `utterances`; during every step: one entry in `dialogue/common.yaml`.
- A new step: its words in `steps.yaml`, one line in
  `procedure.build_steps`, one case file, one packet.

## Run

The frames are not in the repository. Extract the recording at 1 frame per
second or more and write an index CSV with `source_time_s` and
`frame_image` (paths relative to the CSV). From the sample's folder, with
the models in `yaml/models.json` running:

```bash
uv --config-file ../../uv.toml run python eval/run_steps.py \
    --frames-index /path/to/frames/index.csv --out runs/steps/
uv --config-file ../../uv.toml run python eval/run_dialogue.py --out runs/dialogue/
```

`--only heat_water` (a step) or `--only looked_away` (a case) runs some;
`--fakes eval/smoke.fakes.yaml` runs the plumbing on scripted replies (the
numbers then mean nothing). Each writes `report.md` and `results.json`
(every line said, every reported state, timings) and exits 0 when
everything passed. The labels come from one recording, derived from
captions and checked by eye; read the numbers with that in mind.
