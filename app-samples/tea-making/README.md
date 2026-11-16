<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# tea-making

Tea-making guidance on the core and the kit: a procedure that watches the
camera (`xr_ai_kit.sop`), next to a conversation that answers questions
about it. The design follows the tea sample's independent-flows
architecture: the procedure is ordinary Python calling plain async steps,
a small runner owns the run and its controls, and the conversation reads
the run's state. There is no step language and no workflow engine.

## How it works

```text
Voice -> Router (Clef)
           |-- tea           controls only: start, stop, skip, restart
           |                 -> SopRunner -> procedure.py (one run per person)
           |                                   identify, verify x3, timer
           |                                   |
           |                     tea.context <-+ the run's state
           |                         |
           |-- conversation  <-------+  every question about the guide
           |-- observation   what the camera shows now
```

| Step | Function (`xr_ai_kit.sop`) | Done when |
|---|---|---|
| identify | `identify`: read the label into `TeaLabel`, then recommend `Brewing` settings | a tea type is read; skipping uses the defaults |
| fill_water | `verify` | water is seen in the kettle in 2 of the last 4 decidable frames |
| heat_water | `verify` on a `Thermometer` reading | the display reads at least the target, with a unit (printed, or `display_unit` in `app.yaml`); a setpoint or a reading with no unit is not judged |
| start_steeping | `verify` | a real tea bag touches real water (not packaging art) |
| steep | `timer` | the recommended time has passed; it starts after the tea goes in, or when that check is skipped |

Each check is `judge` in a loop: the omni model describes the frame (or
reads it into a schema), and Clef judges that description against the
step's criteria (positive, negative or unknown). Unknowns take no slot, so
an obscured view neither confirms nor resets a check. The wording of every
question, criterion and message is in `steps.yaml`; the procedure itself
is `procedure.py`, front to back.

Starting the guide starts a run that outlives the turn: a question or an
interruption never cancels it. "Skip this step" cancels only the current
step; "stop" ends the run. Each change is traced and written to
`artifacts/<participant>/sop-tea.jsonl`.

Compared with the original tea sample, this port does not yet retrieve
brewing settings from documents (the model recommends them; `identify`
takes a `retrieve` function when an embedding service is running), does not
ask the user which unit the kettle displays (set `display_unit` instead),
and has no live event viewer.

## Run

The roles are named by job, as in the teaching samples, and
`yaml/models.json` binds them: `perceive` and `answer_with_reasoning`
(Nemotron-3 Nano Omni, with and without images), `decide` (Clef) and
`speech_out` (Pocket TTS); speaker ASR takes voice input.
Start them, then the sample, from this directory:

```bash
uv run --project model-servers tea_making_model_servers
uv run tea_making
```

The launcher starts DeviceIOHub and the app (`tea_making_app`). Open the
authenticated web-client URL that DeviceIOHub prints, allow the microphone
and camera, and connect. After the agent reports ready, say "Start talking
to me", then "Help me make tea", "What step am I on?", "What do you see?",
"Skip this step" or "Stop". Stop with Ctrl+C.

## Without a device

```bash
uv --config-file ../../uv.toml run xr-ai-dev run app.py scenarios/ --out runs/
uv --config-file ../../uv.toml run pytest
```

| Scenario | What it checks | Media |
|---|---|---|
| `guide.yaml` | the whole guide on scripted models: label read and settings recommended, water seen, the display at the target, tea in the water, the countdown (virtual time), and a question answered from the run's state | synthetic pictures (`scenarios/make_images.py`), sent as on-demand pictures |

With `--real-models` and your own pictures (`--media-dir`, same file
names) the same scenario runs on the real models.

## Test packets

`eval/` holds one test packet per step: the frames the camera may show
while that step is active (its own scene, other steps' scenes, look-alikes
such as the kettle's target setting flashing on its display) and what the
user may say then (questions such as "what's the next step?", controls,
chatter, negated or quoted controls), each with what the guide should do.
`eval/harness.py` runs them against the app's own decisions and models and
reports false completions, unintended state changes, missed controls and
ungrounded answers. See `eval/README.md`.
