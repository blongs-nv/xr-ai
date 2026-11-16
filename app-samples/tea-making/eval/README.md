<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Test packets for the tea guide

One packet per step says what can reasonably happen while that step is
active, and what the guide should do about it. `harness.py` runs every
packet against the app's own code and models and reports what passed.

```text
packets/
  common.yaml          utterances asked during every step; step words
  not_started.yaml     before the guide starts
  identify.yaml        one file per step: its state, frames, utterances
  fill_water.yaml
  heat_water.yaml
  heat_water.asking.yaml  the same step while it waits for the person's answer
  start_steeping.yaml
  steep.yaml
  completed.yaml       after the guide finished
recordings/
  green-tea-04.yaml    what the recording shows, when, and where the times came from
```

## A packet

```yaml
step: heat_water
state:                       # the run while this step is active
  steps: {identify: done, fill_water: done, heat_water: running}
  detail: "running: 0 of 2 confirmations ..."   # the running step's latest report
  asking: "..."            # optional: a question the step is waiting on
  noted: {temperature_c: 80, duration_s: 120}
frames:                      # what the camera may show during this step
  - {name: rising, at: [26, 321], expect: not_done}
  - {name: at_target, at: [339, 346], expect: done}
  - name: setpoint_flash     # looks done, is not
    at: [22.5, 25.5]
    expect: not_done
    why: 176 is the kettle's target setting while the button is pressed
utterances:                  # what the user may say during this step
  - say: What temperature are we waiting for?
    expect: {route: conversation, answer: {mentions: ["80", "176"]}}
```

**Frames** are times in the recording. The step's check (exactly the live
one: `judge` with `steps.yaml`'s wording) is run on a frame every 2 s of
the group, and the live rule (2 positive among the last 4 decidable
frames) is replayed over them:

| `expect` | Passes when |
|---|---|
| `done` | the rule completes the step on the group |
| `not_done` | it does not: completing here is a false completion |
| `either` | always (ambiguous footage, reported only) |

A group may also give `facts` (what the person already said, typed as the
check's `needs`, for example `{unit: fahrenheit}`), and test the side
escalation that runs beside the check: `asks: true` means it should ask
the person something during this group (`asks_about` lists words its
question must include), `asks: false` that it must not. The harness replays
its pacing (one judgement every `review_every_s` seconds of the recording,
with the latest frame, the step's state and the check's latest report, and
the facts) on the same reasoning model as live (`question_for`); nothing
is asked of anyone. The escalation never decides the step, so `expect`
scores the fast check alone.

Groups from other steps' parts of the recording test what happens when a
step sees the wrong scene: frames of the label while the guide waits for
hot water, frames of the heating kettle while it waits for the tea bag.

**Utterances** are judged in the packet's state, through the app's own
decisions, performing nothing: the Router's choice (`Router.choose`), the
guide's control decision (`SopRunner.classify`; a non-control passes the
turn on, as live) and the Conversation's answer (`Conversation.answer`
with the guide's state as context).

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

- A new thing the user might say during one step: one entry under that
  packet's `utterances`; during every step: one entry in `common.yaml`.
- A new part of a recording: one entry under a packet's `frames`, with its
  `why` when it is a trap.
- A new step: one packet file, and its words in `common.yaml`.
- A new recording: one file in `recordings/` and frame groups that use its
  times (one recording per run, `--recording`).

No code changes for any of these.

## Run

The frames are not in the repository. Extract the recording at 1 frame per
second or more and write an index CSV with `source_time_s` and
`frame_image` (paths relative to the CSV). From the sample's folder, with
the models in `yaml/models.json` running:

```bash
uv --config-file ../../uv.toml run python eval/harness.py \
    --frames-index /path/to/frames/index.csv --out runs/packets/
```

Without `--frames-index` only the utterances run. `--only heat_water
steep` runs some packets; `--fakes eval/smoke.fakes.yaml` runs the plumbing
on scripted replies (the numbers then mean nothing). The exit code is 0 when
everything passed.

## Output

`runs/packets/report.md`: the totals (frame groups and utterances passed;
false completions; unintended state changes, a control from something
that is not one; missed controls), then each packet's frame groups (judged
positive, negative, unknown; when the rule completed) and utterances
(route, control, what was wrong). `results.json` has every frame's verdict
with the model's observation and every answer, for reading why.

The labels come from one recording, derived from captions and checked by
eye; read the numbers with that in mind.
