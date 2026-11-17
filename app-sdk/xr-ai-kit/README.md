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
| `xr_ai_kit.sop` | standard operating procedures: steps (`steps`: what every step shares; `step_types`: `WatchStep`, `ReadStep`, `WaitStep`), an optional escalator beside a step (`escalation`), and `SopRunner`, which runs a procedure written as plain Python, one run per participant, with controls and a summary for the conversation (see below) |

Routing by hand, without a `Router`, is the core's `role.decide`.

## Procedures (`xr_ai_kit.sop`)

A procedure is an ordinary `async def` that runs its steps in order with
`await run.step(step)`. A step's words live in a YAML file; its type, its
models and its code hooks are chosen in code:

```yaml
fill_water:
  does: fill the kettle with fresh water     # the plan the conversation reads
  say: Fill the kettle and keep it in view.  # spoken as the step starts
  done: Water is visible in the kettle.      # spoken when it finishes
  watch:                                     # the watch step's own section
    question: Describe whether water is visibly inside the kettle.
    criteria: {instructions: Is there water?, positive: ..., negative: ..., unknown: ...}
  escalation: The kettle has to be in view; the person knows if it is full.
steep:
  say: Steeping for {minutes:g} minutes.
  done: The tea is ready.
```

```python
from xr_ai_kit.sop import Escalated, Escalator, Run, SopRunner, StepDefinitions, WaitStep, WatchStep

STEPS = StepDefinitions.load("steps.yaml")
fill = Escalated(
    WatchStep.define(STEPS["fill_water"], observe=vlm, decide=classifier),
    Escalator.define(STEPS["fill_water"], reasoning),
)
steep = WaitStep.define(STEPS["steep"]).lasting(180).format(minutes=3)
STEPS.check_read()  # every section of the file has a reader

async def make_tea(run: Run) -> None:
    await run.step(fill)
    await run.step(steep)

tea = app.add(SopRunner("tea", make_tea, title="Tea guidance", decide=classifier, plan=STEPS.plan()))
# tea.control is a router destination (start, stop, skip, restart; anything else passes on);
# tea.context is the run's summary, for a Conversation's context=.
```

- **Steps.** A step is any object with `id`, `say` and `run(io)`; `io` gives
  it the camera frames, speech and a progress callback. Each step type names
  the models it uses as its own fields (a watch step's `observe` and
  `decide`), reads its own section of the YAML, and may have
  `closing(result)`, the line spoken when it finishes. `WatchStep` labels a
  frame every few seconds (observe, then decide) until N of the last M
  decidable frames are positive; `ReadStep` reads into a schema and derives a
  result; `WaitStep` counts down. An app's own step type works the same way.
- **Escalation** is optional and separate from the steps. An `Escalator`
  runs beside one step (`Escalated(step, escalator)`) and shares only a
  `Link` with it: it reads the step's **outbound** context (what the step
  does, its recent reports, what it knows) and gives it **inbound** context
  (what the person said). It talks to the person itself: when the step is
  stuck, it asks for what is missing or tells them what to bring into view.
  A step type supports it by implementing `linked(link)`; the watch step does.
- `SopRunner` owns the run: each step is its own task, so skip cancels only
  the step and stop cancels the run; a new turn never cancels either. Every
  change is traced, written to the participant's artifacts (`sop-<name>.jsonl`)
  when the app has an artifacts folder, and summarized in `runner.context`.
  An escalator reads the channels named in `escalation_context` (the
  dialogue) and the run's summary, as its settings choose.
- `xr_ai_kit.sop.evaluation` runs one step at a time on a recorded scene and
  a scripted person, scored by the step type's own scoring module (and the
  escalator's, for an escalated step); it runs the step that ships.

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
