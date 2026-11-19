<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-kit

Kit components for XR AI apps (import name `xr_ai_kit`): the common patterns,
built only from the public names of the core, `xr-ai-app` (`xr_ai_app`).
The core holds the primitives that need the runtime, the hub or the model
binding (App, Component, Participant, Role with `ask`/`stream`/`decide`,
outputs, context channels, frames, `@tool`, tracing and the clock, the devkit
and test kit); anything composed from those lives here, however common.

| Module | What |
|---|---|
| `xr_ai_kit` | `Conversation`, `Router`, `OneShotDestination`, `StartStopDestination`, `RouterDestination`, `DestinationResult`, `Admission`, `Ack`, `Delegate`, `Job` |
| `xr_ai_kit.conversation` | `Conversation`: multi-turn answers from one role over the context channels it lists, spoken on an output, with a streaming tool loop over `@tool` functions (`xr_ai_app.tools` calls each tool); a `return_direct` tool's result (text, or a stream of text) is spoken as the answer and ends the loop; the loop speaks the answer, the tool returns it |
| `xr_ai_kit.routing` | the routing names, and `choose` (one pick for a turn: `role.decide` plus focus, retries and the dialogue); the devkit's `destination:<name>` exercise target (`resolve_destination`, `destinations`) |
| `xr_ai_kit.delegate` | `Delegate`, `Job`, `JobUpdate` and the job back ends: `FunctionBackend`, `A2ABackend` (remote agents over A2A; extra `xr-ai-kit[a2a]`) |
| `xr_ai_kit.sop` | standard operating procedures: steps (`steps`: what every step shares; `step_types`: `WatchStep`, `ReadStep`, `TransformStep`, `WaitStep`), an optional escalator beside a step (`escalation`), and `SopRunner`, which runs a procedure written as plain Python, one run per participant, with controls and a summary for the conversation (see below) |

Routing by hand, without a `Router`, is the core's `role.decide`.

## Procedures (`xr_ai_kit.sop`)

A procedure is an ordinary `async def` that runs its steps in order with
`await run.step(step, inputs)`. A step's words live in a YAML file, loaded
into the words schema of its type; its type, its models and its schemas are
chosen in code:

```yaml
# steps/fill_water/words.yaml: a watch step's words (WatchWords)
does: fill the kettle with fresh water            # the plan the conversation reads
start_line: Fill the kettle and keep it in view.  # spoken as the step starts
done_line: Water is visible in the kettle.        # spoken when it finishes
question: Describe whether water is visibly inside the kettle.
criteria: {question: Is there water?, positive: ..., negative: ..., unknown: ...}
```

```python
from xr_ai_kit.sop import (
    EscalationWords, ReasoningEscalator, Run, SopRunner, StepWithEscalator, StepWords, WaitInputs, WaitStep,
    WatchModels, WatchStep, WatchWords, load_words,
)

class SteepInputs(WaitInputs):          # what steep runs with: duration_s, and {minutes} for its words
    @property
    def minutes(self) -> float:
        return self.duration_s / 60

fill_water = StepWithEscalator(
    WatchStep("fill_water", load_words("steps/fill_water/words.yaml", WatchWords),
              WatchModels(perceive=vlm, decide=classifier)),
    ReasoningEscalator(load_words("steps/fill_water/escalation.yaml", EscalationWords), reasoning),
)
steep = WaitStep("steep", load_words("steps/steep/words.yaml", StepWords), inputs_schema=SteepInputs)

async def make_tea(run: Run) -> None:
    await run.step(fill_water)
    await run.step(steep, SteepInputs(duration_s=180))   # inputs: what earlier steps produced

steps = (fill_water, steep)
tea = app.add(SopRunner("tea", make_tea, title="Tea guidance", decide=classifier,
                        plan=[(step.id, step.words.does) for step in steps],
                        conversation=voice.dialogue))   # what each step and its escalator see
# tea.control is a router destination (start, stop, skip, restart; anything else passes on);
# tea.context is the run's summary, for a Conversation's context=.
```

- **Steps.** A step is defined once and run with inputs, like a function:
  any object with `id`, `inputs_schema` (a pydantic model of what each run
  starts with, or `None`), `start_line(inputs)`, `done_line(inputs, result)`
  and `run(environment, inputs)`. Its `StepEnvironment` gives it the camera,
  speaking to the person, asking them something, reporting its progress, the
  conversation with the person (`environment.conversation`: its `text()` and
  its `changes()`), and the run's summary (`environment.run_summary()`).
  The kit's step types are built from the step's id, its words (each type
  defines a words schema, such as `WatchWords`; `load_words` reads one from
  YAML, and a misspelled key fails when it is loaded), the models it uses,
  and its `inputs_schema`; the `{placeholders}` in its words name fields or
  properties of its inputs and are checked when it is built. The procedure
  passes each step what earlier steps produced: `run.step(step, inputs)`.
  `WatchStep` labels a frame every few seconds (perceive, then decide) until
  N of the last M decidable frames are positive, optionally reading each
  frame into a schema (`StructuredObservation`); `ReadStep` reads into a
  schema until it has enough, merging what each frame shows;
  `TransformStep` has a reasoning model turn one schema into another
  following its prompt (filling in what a reading left empty, keeping what
  it has), may ask the person a question, and cannot be skipped; `WaitStep`
  counts down. An app's own step type works the same way. Each step type's
  README (`xr_ai_kit/sop/step_types/<type>/README.md`) covers its patterns,
  what works and what does not (measured on real models), smaller models and
  speed; `step_types/README.md` covers shaping a schema a vision model fills
  in.
- **Escalation** is optional and separate from the steps. An escalator runs
  beside one step (`StepWithEscalator(step, escalator)`) and shares only an
  `EscalationChannel` with it: it reads a `StepSnapshot` of the step (what
  the step does, its recent `Evidence`, each with the frame behind it, what
  it knows) and gives it `AddedContext` (what it learned). It talks to the
  person itself, and can look at the camera (`environment.frames`). An
  escalator is anything with `run(channel, environment)`: the kit's
  `ReasoningEscalator` reasons in text over the evidence and asks the person
  or tells them what to bring into view, with a prompt each step may replace
  (its words, `EscalationWords`: `guidance`, and optionally `instructions` or
  the whole `prompt`), and gives the step what the person says in the
  conversation; another could send the frame behind the evidence to a model
  of its own. A step type supports escalation by implementing
  `run_with_escalation(environment, channel, inputs)`; the watch step does.
- `SopRunner` owns the run: each step is its own task, so skip cancels only
  the step and stop cancels the run; a new turn never cancels either. Every
  change is traced, written to the participant's artifacts (`sop-<name>.jsonl`)
  when the app has an artifacts folder (each line said or asked, with who said
  it: the step or its escalator), and summarized in `runner.context`.
  The runner gives each step the app's conversation
  (`SopRunner(conversation=voice.dialogue)`) and the run's summary.
- `xr_ai_kit.sop.evaluation` runs one step at a time on a recorded scene and
  a scripted person, scored by the step type's own scoring module (and the
  escalator's, for an escalated step); it runs the step that ships. A
  case gives the conversation and the run's summary as fixed text
  (`conversation:`, `run_summary:`).

## Routing helpers

A `Router` takes `RouterDestination`s. `Conversation` knows nothing about
routing (it is a turn handler, `conversation.answer`); the common patterns
are helpers, and anything else is a `RouterDestination` with its hooks.

- **`OneShotDestination(name, description, handler)`**: a plain turn handler,
  routed turn by turn by its description.

  ```python
  chat = OneShotDestination("conversation", "Chat and questions.", conversation.answer)
  routing = app.add(Router(decide=decide, destinations=[chat, ...], default=chat))
  app.on_turn(routing.handle)  # the Router does not register itself
  ```

- **`StartStopDestination(name, description, handler, *, decide, output, start, stop, content, other=..., started=None, stopped=None, prompt=...)`**:
  a mode the user starts and stops by asking, such as note-taking. Each
  turn it gets, `decide` reads it against the choice descriptions `start`,
  `stop`, `content` and `other`. Off: `start` starts the mode (says
  `started`), `content` starts it and runs `handler` on the turn, `other` is
  declined (the Router routes the turn elsewhere). On, it holds the Router's
  focus: `stop` ends it (says `stopped`), `content` runs `handler`, `other`
  is declined and the mode stays on. `handler` knows nothing about routing.
  The flag is in `p.state`, so the mode ends when the participant leaves.

  ```python
  async def take_note(turn, p):
      await notes.publish(p, turn.text)
      await p.reply("Noted.", to=voice)

  note_mode = StartStopDestination(
      "notes", "Note taking.", take_note, decide=decide, output=voice,
      start="Start taking notes.", stop="Stop taking notes.",
      content="A statement to write down.", started="Okay, I'm taking notes.",
      stopped="Okay, notes stopped.")
  ```

- **Custom: `RouterDestination(name, description, handler, *, admit=None, focused=None)`**:
  a component builds it in `__init__` from its own methods and keeps it as
  an attribute; `focused(p) -> bool` holds the focus, `admit(turn, p) ->
  Admission` accepts or declines before the handler runs; the handler may
  return `DestinationResult("pass", reason=...)` to hand the turn back. The
  procedure runner's `SopRunner.control` and the conversation sample's
  observation, alert and note destinations are written this way.

  ```python
  class NoteTaking(Component):
      def __init__(self, decide, output, log):
          super().__init__()
          ...
          self.notes = RouterDestination(
              "notes", ROUTING, self._take,
              admit=self._admit, focused=self.holds_focus)
  ```

An app depends on this package next to `xr-ai-app`, both as path sources
(as in `../../app-samples/*/pyproject.toml`):

```toml
dependencies = ["xr-ai-app", "xr-ai-kit"]

[tool.uv.sources]
xr-ai-app = { path = "../../app-sdk/xr-ai-app", editable = true }
xr-ai-kit = { path = "../../app-sdk/xr-ai-kit", editable = true }
```

The kit imports nothing private from the core:
`tests/app/test_contract_surface.py` checks its imports and its attribute
reaches, as for every kit.
