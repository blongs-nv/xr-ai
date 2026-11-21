<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# App SDK samples, grouped by the model servers they use

Apps on the `xr_ai_app` SDK (`../app-sdk/xr-ai-app`) and its kits. Each
group below runs on one model stack and has its own model-server launcher
and its own launcher for DeviceIOHub and the app; every sample runs on a
headset or in the browser (the web client) with the real models, and without
a device on scripted or real models (`scenarios/`).
The core package README (`../app-sdk/xr-ai-app/README.md`) lists every
public name of the core classes.

| Folder | Samples | Model stack (its `model-servers/`) | Run on DeviceIOHub |
|---|---|---|---|
| [`sdk-tour/`](sdk-tour/README.md) | 01-12: a numbered walk through the SDK, one small sample per piece (voice, vision, monitoring, routing, multi-party, delegation, LangChain, procedures, tracing, capture) | upstream conversation-example's: Cosmos, Nemotron-3 Nano Omni, Clef, Pocket TTS, speaker ASR | `uv run sdk-tour 01` ... `uv run sdk-tour 12` |
| [`simple-vlm/`](simple-vlm/README.md) | upstream's simple-vlm-example on the core only | upstream simple-vlm-example's: Cosmos, Pocket TTS, speaker ASR | `uv run simple_vlm` |
| [`conversation/`](conversation/README.md) | upstream's conversation-example on the core and the kit | upstream conversation-example's | `uv run conversation` |
| [`lab-instruments/`](lab-instruments/README.md) | upstream's lab-instrument-monitoring on the core and the kit | upstream lab-instrument-monitoring's | `uv run lab_instruments` |
| [`tea-making/`](tea-making/README.md) | tea-making guidance: a procedure that watches the camera (`xr_ai_kit.sop`) next to a conversation | Nemotron-3 Nano Omni, Clef, Pocket TTS, speaker ASR (upstream conversation-example's servers, without Cosmos) | `uv run tea_making` |

Each folder's README has its run commands: first its model servers
(`uv run --project model-servers <...>_model_servers`, which reuses servers
already running), then its launcher, which starts DeviceIOHub and the app and
prints the web client's URL. Every sample is its own uv project: its
`pyproject.toml` names the packages of this repository (by path) and from
PyPI that it needs. `ETHOS.md` here is how every sample should behave (NeMo
Compass compares traces against it).

| Sample | Needs from this repo | Needs from PyPI (beyond what those bring) |
|---|---|---|
| `sdk-tour/` | see [its README](sdk-tour/README.md) | |
| simple-vlm | `xr-ai-app` only (no kit), `xr-ai-launcher` and `xr-ai-logging` for its launcher | none (`pytest`, `pytest-asyncio` as dev tools). Its `model-servers/` is upstream's launcher, a separate uv project |
| conversation | `xr-ai-app`, `xr-ai-kit`, `xr-ai-launcher` and `xr-ai-logging` for its launcher | `pydantic` (`pytest`, `pytest-asyncio` as dev tools). Its `model-servers/` is upstream's launcher, a separate uv project |
| lab-instruments | `xr-ai-app`, `xr-ai-kit`, `xr-ai-tools[marker-tracking]` (the QR/ArUco detector: OpenCV, zxing-cpp), `xr-ai-launcher` and `xr-ai-logging` for its launcher | `pydantic`, `pyyaml`, `Pillow` (`pytest`, `pytest-asyncio` as dev tools). Its `model-servers/` is upstream's launcher, a separate uv project |
| tea-making | `xr-ai-app`, `xr-ai-kit`, `xr-ai-launcher` and `xr-ai-logging` for its launcher | `Pillow` (the test packets), `pydantic`, `pyyaml` (`pytest`, `pytest-asyncio` as dev tools). Its `model-servers/` runs upstream conversation-example's servers, a separate uv project |

| Sample | Scenarios | Evaluates | Record for a real run |
|---|---|---|---|
| simple-vlm | `typed.yaml`, `spoken.yaml`, `on_demand.yaml` | a question about a streamed frame, typed and spoken; with no stream, the app asks for one picture | nothing (synthetic pictures, `scenarios/make_images.py`) |
| conversation | `general.yaml`, `view.yaml`, `view_on_demand.yaml`, `notes.yaml`, `alert.yaml` | each destination: a general question; a question about the view (streamed, and one picture on demand); a note taken, recalled by the conversation, note mode left; an alert that fires when the picture changes | nothing (synthetic pictures, `scenarios/make_images.py`) |
| lab-instruments | `read_one.yaml`, `read_two.yaml`, `view.yaml`, `unknown_marker.yaml`, `no_marker.yaml`, `unreadable.yaml`, `monitor_change.yaml`, `monitor_lost.yaml`, `watch.yaml` | marker detection then one vision call: one and two instruments (two from one picture on demand), the view, an unknown marker, no marker, an unreadable display; the instrument monitor's change (said once) and lost instrument; the silent visual watch, asked about later | nothing (synthetic pictures with upstream's printable markers, `scenarios/make_images.py`); for your own, photograph instruments wearing `sample-markers/` and edit the expected readings |
| tea-making | `guide.yaml` | the whole guide: label read and settings recommended, water seen, the display at the target, tea in the water, the countdown, a question answered from the run's state | nothing (synthetic pictures, `scenarios/make_images.py`, sent on demand) |

`xr-ai-app` in turn brings the foundation packages it uses (`agent-sdk/`;
the earlier `xr-ai` layer was folded into it); see `DEPENDENCIES.md` at the
repository root for the full graph.

| Sample | Shape | What it shows |
|---|---|---|
| simple-vlm | the core only | upstream's simple-vlm-example on `xr_ai_app` alone: `app.on_turn`, `p.frames.current()` (a streamed frame, or one picture asked for), `role.stream`, `Voice(tts=...)`; upstream's `yaml/` and `model-servers/`, a launcher (`uv run simple_vlm`) for DeviceIOHub and the app |
| conversation | the kit's showcase | upstream's conversation-example on the core and the kit: `Router` (Clef) over `Conversation` (dialogue and notes as context), the current view (`p.frames.current()`, `role.stream`), a note-taking destination that holds focus (`RouterDestination` with `admit=` and `focused=`, `ContextLog`), a one-shot visual alert (`p.spawn`, `role.decide(frames=...)`, `p.say(urgent=)`), optional `Ack`; upstream's `yaml/`, prompts and `model-servers/`, a launcher (`uv run conversation`) |
| lab-instruments | a deterministic detector, then a model; the kit's tool loop | upstream's lab-instrument-monitoring on the core and the kit: `Conversation` with nine `@tool`s, QR/ArUco markers found with `xr_ai_tools.marker_tracking` on `p.frames.current()`, one `vlm.ask` on the marked frame, two monitors as `p.spawn` tasks (`p.say`), a `ContextLog` the conversation reads, JSONL records with `p.artifacts`; upstream's `yaml/`, prompts and `model-servers/`, a launcher (`uv run lab_instruments`) |
| tea-making | a procedure next to a conversation | `SopRunner` (`xr_ai_kit.sop`) running `procedure.py`: a `ReadStep` (the package into a `Tea` schema), a `TransformStep` (the brewing settings it does not print), three `WatchStep`s (the omni model perceives, Clef decides; one reads a `Thermometer` schema and checks it in code), each with a `ReasoningEscalator` beside it, a `WaitStep`; the runner's controls as a router destination, its state as the `Conversation`'s context, its record in `p.artifacts`; a launcher (`uv run tea_making`) |

## The rules every sample follows

- **Roles, not models.** Code asks for a role (`perceive`, `decide`,
  `answer_lightweight`, `answer_with_reasoning`); the group's `models.json`
  (`sdk-tour/models.json` for 01-12, `yaml/models.json` in the others) says
  which model fills it.
- **Output is explicit.** Everything spoken names where it goes:
  `p.reply(..., to=voice)` answers a turn, `p.say(..., to=voice)` speaks on
  the assistant's own initiative.
- **Context is explicit.** A model sees exactly the channels listed in its
  `context=[...]`. Channels are `ContextLog` (append, summarized when full)
  or `ContextValue` (replaced); each names its scope: `app`, `participant`,
  `session`.
- **State has a lifetime.** `app.state`, `p.state` (the participant),
  `p.session.state` (the connection), `turn.state` (one turn).
- **Background work is named.** `p.spawn(coro, key=...)` outlives the turn;
  `p.cancel(key)` stops it; it is cleaned up when the participant leaves.
- **Routing is optional, and always visible.** One handler:
  `app.on_turn(fn)`. Several: `app.on_turn(routing.handle)` for a
  `routing = app.add(Router(...))`, or `decide.decide(...)` in your own
  handler (05). Every app says in one line where its turns go.
- **Long work goes behind the front end.** `Delegate` runs it as a job and
  turns its updates into context and speech; the back end never speaks.
- **Everything is traced** with ids that tie each event to its cause.
- **Frameworks plug in at the edges.** LangChain/LangGraph through the
  optional `xr_ai_kit.langchain` adapters; the core never imports them.

## Compare speech performance across models and hardware

The devkit also replays fixed WAVs and camera images through the apps' real speech pipeline. See [real audio evaluation](../app-sdk/xr-ai-app/docs/REAL_AUDIO.md) for fixture preparation, profile selection, report interpretation and the five sample-owned cases. Existing accuracy scenarios remain usable with local or hosted models independently of real-time speech.
