<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# tea-making

Tea-making guidance on the core and the kit: a procedure that watches the
camera (`xr_ai_kit.sop`), next to a conversation that answers questions
about it. Three layers, each with one job: **steps** (kit step types,
each verified on its own), the **procedure** (`procedure.py`: which steps,
in what order, what passes between them) and the **runner** (`SopRunner`:
one run per person, its controls, its state for the conversation). There
is no step language and no workflow engine.

## How it works

```text
Voice -> Router (Clef)
           |-- tea           controls only: start, stop, skip, restart
           |                 -> SopRunner -> procedure.py (one run per person)
           |                                   read, transform, watch x3, wait
           |                                   |
           |                     tea.context <-+ the run's state
           |                         |
           |-- conversation  <-------+  every question about the guide
           |-- observation   what the camera shows now
```

| Step | Kit step type | Done when |
|---|---|---|
| identify | `ReadStep`: read the package frame after frame into a `TeaPackageReading` (each field's source, then its value only when the package prints it), merging what each frame shows | a kind of tea is read |
| choose_settings | `TransformStep`: fill in the brewing settings the package does not print, keeping what it does; asks the person what tea it is when nothing was read | both settings are set; it cannot be skipped |
| fill_water | `WatchStep` | water is seen in the kettle in 2 of the last 4 decidable frames |
| heat_water | `WatchStep` on a `Thermometer` reading | the display reads at least the target, with a unit (printed, or said by the person when its escalator asks); a setpoint is not judged |
| start_steeping | `WatchStep` | a real tea bag touches real water (not packaging art) |
| steep | `WaitStep` | the steeping time has passed (the middle of a range); it starts after the tea goes in, or when that check is skipped |

A watch step's **check** judges a frame every 2 s: the omni model
describes it (or reads it into a schema) and Clef labels that description
against the step's criteria (positive, negative or unknown); unknowns take
no slot, so an obscured view neither confirms nor resets it. Beside each
camera step runs its **escalator** (`StepWithEscalator`), an agent of its
own that shares only an escalation channel with the step: it reads a
snapshot of the step (its task, its latest evidence, what it knows) and,
when the evidence is mostly unknown, asks the person for what is missing
(the kettle's unit; the answer is added to the step's context, which the
check reads) or tells them what to bring into view. It never completes a
step. What every escalator reads besides its step (`context`: the
procedure's summary `run`, the recent `dialogue`, or neither) and how often
it may speak are app.yaml's `settings.tea.escalation`.

Every step's words are in `steps.yaml`: what it says as it starts
(`start_line`) and when it finishes (`done_line`, filled from its result:
identify says the tea it read, choose_settings the settings, heat_water the
reading that completed it), and one section per owner (`read`, `transform`
or `watch` for the step, `escalation` for its escalator). Each step is
defined once in `procedure.py` (`TeaProcedure`): its type, models and
schemas, and what it runs with (`HeatInputs`, `SteepInputs`). The procedure
itself (`make_tea`) is the order of the steps and what passes between them:
the `Tea` read from the package goes to choose_settings, and its
temperature and steeping time to heat_water and steep. There are no default
settings in code: the model fills in what the package does not print.

Starting the guide starts a run that outlives the turn: a question or an
interruption never cancels it. "Skip this step" cancels only the current
step (and any question it is waiting on); "stop" ends the run. Each
change is traced and written to `artifacts/<participant>/sop-tea.jsonl`.

Compared with the original tea sample, this port does not yet retrieve
brewing settings from documents (the model fills them in from what it
knows), and has no live event viewer.

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

## Evaluation

`eval/` evaluates each layer on its own. **Step cases** (`eval/steps/`,
one file per step) run the real step, taken from `procedure.TeaProcedure`, on
scenes from a recording (its own scene, other steps' scenes, look-alikes
such as the kettle's target setting, and spliced scenes such as the person
looking away) with a scripted person, scored by the step type's own rules
(`xr_ai_kit.sop.evaluation`). **Dialogue cases** (`eval/dialogue/`) put the guide
in a state and check what it does with what the user says (questions such
as "what's the next step?", controls, chatter, negated or quoted
controls). See `eval/README.md`.
