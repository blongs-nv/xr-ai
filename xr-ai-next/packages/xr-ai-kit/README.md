<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-kit

Kit components for XR AI apps (import name `xr_ai_kit`): the common patterns,
built only from the public names of the core, `xr-ai-next` (`xr_ai_next`).
The core holds the primitives that need the runtime, the hub or the model
binding (App, Component, Participant, Role with `ask`/`stream`/`decide`,
outputs, context channels, frames, `@tool`, tracing and the clock, the devkit
and test kit); anything composed from those lives here, however common.

| Module | What |
|---|---|
| `xr_ai_kit` | `Conversation`, `Router`, `OneShotDestination`, `StartStopDestination`, `RouterDestination`, `DestinationResult`, `Admission`, `Ack`, `Delegate`, `Job` |
| `xr_ai_kit.conversation` | `Conversation`: multi-turn answers from one role over the context channels it lists, spoken on an output, with a streaming tool loop over `@tool` functions (`xr_ai_next.tools` calls each tool); a `return_direct` tool's result (text, or a stream of text) is spoken as the answer and ends the loop; the loop speaks the answer, the tool returns it |
| `xr_ai_kit.routing` | the routing names, and `choose` (one pick for a turn: `role.decide` plus focus, retries and the dialogue); the devkit's `destination:<name>` exercise target (`resolve_destination`, `destinations`) |
| `xr_ai_kit.delegate` | `Delegate`, `Job`, `JobUpdate` and the job back ends: `FunctionBackend`, `A2ABackend` (remote agents over A2A; extra `xr-ai-kit[a2a]`) |
| `xr_ai_kit/procedures/` | a placeholder: the procedures kit is still the separate package `xr-ai-procedures` |

Routing by hand, without a `Router`, is the core's `role.decide`.

## Routing helpers

A `Router` takes `RouterDestination`s. `Conversation` knows nothing about
routing (it is a turn handler, `conversation.answer`); the common patterns
are helpers, and anything else is a `RouterDestination` with its hooks.

- **`OneShotDestination(name, description, handler)`**: a plain turn handler,
  routed turn by turn by its description.

  ```python
  chat = OneShotDestination("conversation", "Chat and questions.", conversation.answer)
  Router(decide=decide, destinations=[chat, ...], default=chat)
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
  procedures kit's `procedure.control` and the conversation sample's
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

An app depends on this package next to `xr-ai-next`, both as path sources
(as in `../../samples/*/pyproject.toml`):

```toml
dependencies = ["xr-ai-next", "xr-ai-kit"]

[tool.uv.sources]
xr-ai-next = { path = "../../packages/xr-ai-next", editable = true }
xr-ai-kit  = { path = "../../packages/xr-ai-kit", editable = true }
```

The kit imports nothing private from the core:
`tests/next/test_contract_surface.py` checks its imports and its attribute
reaches, as for every kit.
