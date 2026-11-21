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
| identify | `ReadStep`: read the package frame after frame (`TeaPackageLook`: is a package in view, its words, then each value with the words it comes from, and brewing directions only when printed), into a `Tea` merged across frames | a kind of tea is read |
| choose_settings | `TransformStep`: fill in the brewing settings the package does not print, keeping what it does; asks the person what tea it is when nothing was read | both settings are set; it cannot be skipped |
| fill_water | `WatchStep` | water is seen in the kettle in 2 of the last 4 decidable frames |
| heat_water | `WatchStep` that describes the display in a schema (`DisplayCaption`: is a lit display in view, what it is on, its characters, the digits, whether a unit letter is printed) and judges it in code (`reached_target`) | the digits reach the target, in a unit printed or said by the person (in the conversation, or when its escalator asks) |
| start_steeping | `WatchStep` | a real tea bag touches real water (not packaging art) |
| steep | `WaitStep` | the steeping time has passed (the middle of a range); it starts after the tea goes in, or when that check is skipped |

A watch step's **check** judges a frame every 2 s: the omni model
describes it (or fills in a schema) and Clef labels that description
against the step's criteria, or, for heat_water, code compares the digits
with the target (positive, negative or unknown); unknowns take
no slot, so an obscured view neither confirms nor resets it. Beside each
camera step runs its **escalator** (`StepWithEscalator`), an agent of its
own that shares only an escalation channel with the step: it reads a
snapshot of the step (its task, its latest evidence, what it knows). As
the step starts, and each time the conversation changes, it reads what the person has said into the facts
the step accepts (`learns_from_conversation`: the kettle's unit, said when
the target was announced or while the water heats), so the check has it
without anyone asking. When the evidence is mostly unknown, it asks the
person for what is missing (the answer is added to the step's context,
which the check reads) or tells them what to bring into view. It never
completes a step. Whether every escalator reads the run's summary and the
conversation, whether it learns from the conversation, and how often it
may speak are app.yaml's `settings.tea.escalation`; the conversation is
`SopRunner(conversation=voice.dialogue)` in app.py.

```text
procedure.py      make_tea (the order of the steps, what passes between them); TeaProcedure builds them
tea.py            what passes between steps: Tea, WaterTemperature, SteepTime
steps/<step>/     everything about one step:
  words.yaml        what it says and asks its models (does, start_line, done_line, question, ...)
  escalation.yaml   how the person can help when its camera check is stuck (camera steps)
  schemas.py        its own types (identify: the package reply; heat_water: the display
                    caption and its judge, HeatWaterInputs; steep: SteepInputs)
eval/steps/       each step's evaluation cases
```

Each step's words are loaded into its type's words schema
(`load_words(..., WatchWords)`), so a misspelled key fails when the app
starts. Each step is defined once in `procedure.py` (`TeaProcedure`): its
type, its models, and its schemas from its folder. The procedure itself
(`make_tea`) is the order of the steps and what passes between them:
the `Tea` read from the package goes to choose_settings, and its
temperature and steeping time to heat_water and steep. There are no default
settings in code: the model fills in what the package does not print.

Starting the guide starts a run that outlives the turn: a question or an
interruption never cancels it. "Skip this step" cancels only the current
step (and any question it is waiting on); "stop" ends the run. Each
change is traced and written to `artifacts/<participant>/sop-tea.jsonl`.

Brewing settings come from the model's knowledge, not from documents, and
there is no live event viewer.

## Adding a step

1. `steps/<id>/words.yaml`: its words, in its type's words schema
   (`WatchWords`, `ReadWords`, `TransformWords`, or `StepWords` for a wait).
2. `steps/<id>/escalation.yaml`, for a camera step with an escalator: how
   the person can help when its check is stuck.
3. `steps/<id>/schemas.py`, when it has its own types: its inputs, the facts
   the person may give it, or a caption schema and its code judge.
4. Build it in `TeaProcedure.__init__` and add it to `self.steps` (that
   tuple drives both the plan the conversation sees and `eval/run_steps.py`).
5. Run it in `make_tea`, where it gets what earlier steps produced; then add
   its cases in `eval/steps/<id>.yaml`.

The sdk-tour's guided procedure keeps every step's words in one file
(`load_words(..., key="bed_clear.step")`); this sample gives each step a
folder. Copy the folder layout once a step has types of its own.

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

For speech replay and performance reports, see [`eval/speech/`](eval/speech/).
