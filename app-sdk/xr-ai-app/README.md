<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# xr-ai-app

The app surface for XR AI agents (import name `xr_ai_app`), built on
the foundation packages under `agent-sdk/`. This package is the core: the
primitives that need the runtime, the hub or the model binding. Patterns
built from them (`Conversation` with its tool loop, `Router` and destinations,
`Delegate` and its job back ends) are the kit, `../xr-ai-kit` (`xr_ai_kit`);
a sample imports from these two. The app core (lifecycle, participants, enrollment,
model roles, frames, the hub adapter, tracing and the test kit) is the
private `xr_ai_app._base` package; its written contracts are
`docs/LIFECYCLE_CONTRACT.md` and `docs/PROFILES_SCHEMA.md`.

| Module | What |
|---|---|
| `xr_ai_app` | for every app: `App`, `Component`, `Participant`, `Turn`, `Session`, `current_turn`, `Hub`, `Frame`, `FrameStream`, `Voice`, `TextSink`, `output_started`, `Role` (routing by hand is `role.decide`), `tool`, context channels (`ContextLog`, `ContextValue`) |
| `xr_ai_app.context` | `render` (the listed channels as the text a model reads) and the channel protocol |
| `xr_ai_app.trace` | trace events with causal ids, sinks, NeMo Relay export; `emit`, `span` for your own code |
| `xr_ai_app.clock` | `now`, `wall_us`, `held`: time that stays right on the devkit's virtual clock |
| `xr_ai_app.tools`, `xr_ai_app.llm` | for tool loops and adapters: tools as data (`describe`, `ToolSpec.call`/`.stream`, `ToolSpec.return_direct`), and the messages of `role.as_llm()` |
| `xr_ai_app.devkit` | the local device: scenarios, `exercise`, a virtual-time loop, an HTTP endpoint and the `xr-ai-dev` command |
| `xr_ai_app.testing` | scripted models and test sessions |

The core classes, and every public name each one has
(`tests/app/surface/test_public_surface.py` checks that they have no other):

| Class | Public names |
|---|---|
| `App` | `App.from_config`, `app.settings`, `app.role`, `app.add`, `app.on_turn`, `app.on_start`, `app.state`, `app.participants`, `app.trace`, `app.run`, `app.start`, `app.stop`, `app.test_session` |
| `Component` | `self.app`, `self.spawn`, and the hooks `on_start`, `on_participant_enrolled`, `on_participant_unenrolled`, `on_stop` |
| `Participant` | `p.id`, `p.name`, `p.connected`, `p.frames`, `p.state`, `p.session`, `p.say`, `p.reply`, `p.withdraw`, `p.spawn`, `p.cancel`, `p.artifacts` |
| `FrameStream` | `frames.current` (one frame now: a fresh streamed frame, or one picture asked of the device), `frames.every` (a frame per tick, streamed or, on demand, a picture) |
| `Session` | `session.id`, `session.state` |
| `Turn` | `turn.text`, `turn.id`, `turn.timestamp_us`, `turn.state`, `turn.replace` |

`app.start()` and `app.stop()` run the app inside your own event loop (`app.run()`
does both and handles signals; the stop sequence is in
`docs/LIFECYCLE_CONTRACT.md`). `p.connected` is false while a `Persistent`
participant has no device connected. `p.artifacts` writes files under the
participant's `artifacts_dir` folder (set in `app.yaml`). `p.say`, `p.reply`
and `p.withdraw` take `to=` (a sink, such as `Voice` or `TextSink`, or a list
of them); without it they use the app's speech output.

An app depends on this package through its own `pyproject.toml` (a path
source, as in `../../app-samples/*/pyproject.toml`), and then has the `xr-ai-dev`
command. Run an app without a headset, from the app's directory:

```bash
uv --config-file <repo>/uv.toml run xr-ai-dev run app.py scenario.yaml --fakes fakes.yaml --out runs/x/
```

See `skills/test-with-local-device/SKILL.md` for the scenario format, the
HTTP endpoint and the report. The samples are in `../../app-samples/`. Watching the camera over time (a
step that judges frames until enough agree) is in the kit's procedure library,
`xr_ai_kit.sop`.

Real microphone-speed replay is available through `xr-ai-app[audio]` and
`xr-ai-dev run --real-audio --real-models --clock real`. It retains the app's
configured STT, enrollment, voice gate and TTS, with an isolated hub and a sender
in another process. Accuracy-only scenarios keep their current behavior. See
[`docs/REAL_AUDIO.md`](docs/REAL_AUDIO.md) and the five-minute Conversation case
in `app-samples/conversation/eval/speech/`.
