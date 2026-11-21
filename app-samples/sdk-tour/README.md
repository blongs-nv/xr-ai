<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# SDK tour: a numbered walk through the SDK, on one model stack

The SDK's pieces one at a time, in order: what people build with a
voice/video-first front end, one small sample per piece,
on the `xr_ai_app` SDK (`../../app-sdk/xr-ai-app`) and its kit, `xr_ai_kit`
(`../../app-sdk/xr-ai-kit`: `Conversation`, `Router`, `Delegate`). Samples 08
and 09 also use the optional LangChain adapters
(`xr_ai_kit.langchain`, the kit's `langchain` extra); 10 and 12 its procedure
library (`xr_ai_kit.sop`). The core package README
(`../../app-sdk/xr-ai-app/README.md`) lists every public name of the core
classes.

The twelve share one model stack, so they live in one folder with one of each
thing a stack needs:

| Here | What it is |
|---|---|
| `models.json` | the shared model profile: roles `perceive` (Cosmos, :8100), `answer_with_reasoning` and `answer_lightweight` (Nemotron-3 Nano Omni, :8108), `decide` (Clef, :8120), `speech_out` (Pocket TTS, :8105), `speech_in` (speaker ASR, :8102): upstream `agent-samples/conversation-example`'s stack |
| `model-servers/` | the one launcher for that stack (upstream conversation-example's, unchanged) |
| `yaml/` | DeviceIOHub, voice-gate and session-recording settings, upstream conversation-example's (recordings go to `~/.local/share/xr-ai/captures/xr-ai-app-sdk-tour`) |
| `launch.py`, `pyproject.toml` | the one launcher for DeviceIOHub plus any sample's app: `uv run sdk-tour <sample>` |
| `NN-name/` | the samples; each is its own uv project (`app.py`, `app.yaml`, `scenarios/`) |

## Run a sample on a headset or in a browser

Every sample runs on DeviceIOHub with the real models and the web client.
From `app-samples/sdk-tour/`:

```bash
uv run --project model-servers sdk_tour_model_servers   # once, for all twelve; reuses running servers
uv run sdk-tour 01-hello-voice                            # or: uv run sdk-tour 01
```

The first command starts the model servers (or finds them running) and
returns; they keep running until
`uv run --project model-servers sdk_tour_model_servers --stop`. The second
starts DeviceIOHub, then the sample's app (`uv run --project 01-hello-voice
hello_voice_app`: its own uv project, synced on first use), and prints the
web client's URL. Open it, allow the microphone (and the camera for 02, 03,
09, 10, 12), and connect. Type a question, or say "Start talking to me" and
then speak without a wake phrase ("Stop talking to me" ends it;
`yaml/voice_gate.yaml`). The status shows loading until
every model the app uses answers its health check, then ready. Stop with
Ctrl+C. `--session-recording` records the session (`yaml/media_capture.yaml`;
under `~/.local/share/xr-ai/captures/xr-ai-app-sdk-tour`),
`--hub-config PATH` uses another hub file.

| # | Command | On the device |
|---|---|---|
| 01 | `uv run sdk-tour 01` | talk; it remembers earlier turns |
| 02 | `uv run sdk-tour 02` | "What am I holding?": camera in **Live video** or **On-demand images** mode |
| 03 | `uv run sdk-tour 03` | **Live video** or **On-demand images**: it looks every second (on demand: one picture per look) and speaks up when a condition in `03-camera-monitor/app.yaml` holds for three looks; ask what it saw, or to pause |
| 04 | `uv run sdk-tour 04` | "Start taking notes", dictate, ask a question in between, "Read my notes", "Stop taking notes" |
| 05 | `uv run sdk-tour 05` | "Turn the lights on", "Play some music", "stop" |
| 06 | `uv run sdk-tour 06` | open the URL on two devices (two participants); "The valve is closed" on one is told to the other |
| 07 | `uv run sdk-tour 07` | also starts the stand-in A2A agent (`07-delegate-a2a/research_agent`, :8830) before the app; "Find me a quiet cafe near the station" |
| 08 | `uv run sdk-tour 08` | the same request, to the LangGraph graph in the app |
| 09 | `uv run sdk-tour 09` | "What is 20 C in F?"; **Live video** or **On-demand images** for "how long is this?" |
| 10 | `uv run sdk-tour 10` | **Live video** or **On-demand images** of a 3D printer: "start the checklist"; each camera step is checked look by look (on demand: one picture per look); ask "What step am I on?" |
| 11 | `uv run sdk-tour 11` | talk; the trace goes to `11-tracing/traces/` |
| 12 | `uv run sdk-tour 12` | **Live video** (on demand: two pictures a second): "start capture", do and narrate the task, "stop capture"; the packet is in `12-procedure-capture/artifacts/` |

The camera samples work in both camera modes. A question reads
`p.frames.current()` (02, 09): the newest streamed frame, or, in on-demand
camera mode (nothing streamed), one picture asked of the device. A monitor
reads `p.frames.every(interval)` (03, 10's camera steps, 12): a frame per tick, streamed or, on
demand, one picture per tick (after the first tick, which waits 2 s for a
stream). 12 records at two frames a second, so on demand it asks for two
pictures a second; **Live video** is what a recording is for. If no frame
comes at all, 12 says so.

## Run and evaluate without a device

Every sample is its own uv project. Its `pyproject.toml` lists exactly what it
needs: the parts of this repository (by path, as editable installs) and any
PyPI packages. uv builds a private environment for the sample from that file;
there is nothing to `pip install`. Run from the sample's directory, and pass
the repository's `uv.toml` (it pins the qualified dependency versions):

```bash
cd app-samples/sdk-tour/02-vision-qa
# dry run: scripted models, committed placeholder media, no GPU
uv --config-file ../../../uv.toml run xr-ai-dev run app.py scenarios/ --clock virtual --out runs/
uv --config-file ../../../uv.toml run pytest       # 02 only: its own test_app.py
# evaluation: the real models in models.json (servers already running), your own recordings
uv --config-file ../../../uv.toml run xr-ai-dev run app.py scenarios/ --real-models \
    --media-dir ~/recordings/<sample> --out runs/real/ [--relay]
```

`xr-ai-dev` comes with `xr-ai-app`, so every sample has it: `run` plays a
scenario, `serve` exposes the app on a local HTTP endpoint (see
`skills/test-with-local-device/SKILL.md`). Scripted models (`--fakes`) need no
GPU. `--media-dir` is searched first for each media file a scenario names
(same file name), so your recordings replace the placeholders without editing
the scenario. Each run writes `runs/<scenario>/transcript.md` (who said what,
when, and the failed checks), `report.json` and `trace.jsonl` (every model
call with its prompt and answer, every route and tool, with timings); `--relay`
adds ATOF/ATIF for NeMo Compass and Helix. A summary prints one line per
scenario; the exit code is 0 when all passed. Real models word things
differently: checks use `said_any: [...]` where several answers are right;
edit a scenario's expectations to match your recording (it is the contract).

For sample 07 without the launcher, start its agent first, in a second
terminal:

```bash
cd app-samples/sdk-tour/07-delegate-a2a/research_agent
uv --config-file ../../../../uv.toml run research_agent   # A2A agent on :8830
```

| # | Needs from this repo | Needs from PyPI (beyond what those bring) |
|---|---|---|
| 01, 04, 05, 06, 11 | `xr-ai-app`, `xr-ai-kit` | none |
| 02, 03 | `xr-ai-app` (03: with `[video]`, for its scenario), `xr-ai-kit` | `pydantic` (02 also: `pytest`, `pytest-asyncio` as dev tools) |
| 10, 12 | `xr-ai-app` (10: with `[video]`, to exercise its steps on recorded video), `xr-ai-kit` (its procedure library, `xr_ai_kit.sop`; 10 also uses its Router) | 12: `pydantic` |
| 07 | `xr-ai-app`, `xr-ai-kit[a2a]` (the A2A client) | `pydantic`. Its stand-in agent, `research_agent/`, is a separate uv project that needs only `a2a-sdk[http-server]` and `uvicorn`, nothing from this repo |
| 08, 09 | `xr-ai-app`, `xr-ai-kit[langchain]` (the LangChain adapters) | `langchain-core`, `langgraph` |
| the launcher (`pyproject.toml` here) | `xr-ai-launcher`, `xr-ai-logging` | none |
| `model-servers/` | `xr-ai-model-server-stack` | none |

| # | Scenario | Evaluates | Record for a real run |
|---|---|---|---|
| 01 | `remember.yaml` | the dialogue log: a later turn uses an earlier one | nothing |
| 02 | `mug.yaml` | the conversation asks the camera tool, answers from what it sees | `mug.jpg`: a photo of a mug (edit the expected color) |
| 03 | `stove.yaml` | the monitor speaks up on its own when a condition holds | `stove.mp4`: a lit stove, nobody in view |
| 03 | `stove_on_demand.yaml` | the same with the camera on demand (a held picture, nothing streamed) | `stove.jpg`: a photo of the same |
| 04 | `notes.yaml` | routing into a mode that holds focus, a question passed back, leaving the mode | nothing |
| 05 | `lights.yaml` | a two-level menu by the decision model, a code rule with no model | nothing |
| 06 | `two_people.yaml` | two participants; what one does is told to the other | nothing |
| 07 | `research.yaml` | a long request to an A2A agent; progress and result spoken | nothing (start `research_agent/` first) |
| 08 | `cafe.yaml` | a long request to the LangGraph back end | nothing |
| 09 | `math.yaml` | the LangChain agent with our tools (convert, look) | `ruler.jpg`: a photo of a ruler |
| 09 | `ruler_on_demand.yaml` | `look` with the camera on demand | `ruler.jpg` |
| 10 | `checklist.yaml` | the spoken checklist with camera checks; "What step am I on?" answered from its state | `printer.mp4`: clearing the bed, then closing the door |
| 10 | `checklist_on_demand.yaml` | the camera steps with the camera on demand | `printer.jpg`: a clear bed, the door closed |
| 11 | `hello.yaml` | a traced turn (read `trace.jsonl`) | nothing |
| 12 | `demo.yaml` | capture: frames, captions and narration into an evidence packet | frames of a short task (a folder, `media/demo/`) |

One piece of a sample alone on a window of a recording is `xr-ai-dev
exercise` (see `skills/test-with-local-device/SKILL.md`). The tests that
run every sample with scripted models are in `tests/app/surface/` (they use
the `tests/` project, not the samples' own).

## The samples

| # | Sample | Shape | What it shows |
|---|---|---|---|
| 01 | hello-voice | one handler | `Voice`, `Conversation`, `app.on_turn`, the dialogue log |
| 02 | vision-qa | a conversation with a tool | `@tool`, `p.frames`, `perceive.ask`; **test_app.py**: fake models, the trace in tests |
| 03 | camera-monitor | proactive | a loop per participant, a structured read per frame, a yes streak per condition, `p.say(urgent=)`, `ContextValue`, tools that change state |
| 04 | notes-mode | router + sticky mode | `Router`, `OneShotDestination` (the conversation), `StartStopDestination` (a mode started and stopped by asking, holding focus; a question is declined and routed elsewhere), `ContextLog` |
| 05 | manual-dispatch | routing by hand | `decide.decide()` and a dict lookup, with no Router: code rules, a two-level menu, your own fallback |
| 06 | shared-room | multi-party | `app.state`, app-scoped context, `app.participants`, `Persistent` enrollment |
| 07 | delegate-a2a | front end + another agent | `Delegate`, `A2ABackend`: long work runs on any A2A (Agent2Agent) agent; its progress and result become context and speech; cancel sends A2A `tasks/cancel` |
| 08 | delegate-langgraph | front end + LangGraph | `GraphBackend`, `chat_model(role)` (adapter package) |
| 09 | langchain-agent | a LangChain agent as a destination | `chat_model`, `langchain_tools`, `messages`, `stream_text`, `from_langchain` |
| 10 | guided-procedure | a camera-verified checklist | `xr_ai_kit.sop`: `SopRunner` (controls, the run's state for the conversation), `StepDefinitions` (the steps' words, YAML), `WatchStep` (a camera check), `StepWithEscalator` and `ReasoningEscalator` (an agent beside each camera step: tells the user what to show, or asks them), `WaitStep`; each step says its own lines; the procedure is a plain `async def` |
| 11 | tracing | observability | trace events and ids, `app.trace.add(sink)`, `app.trace.span`, `trace:` in app.yaml |
| 12 | procedure-capture | a recorder | `xr_ai_kit.sop.capture.ProcedureCapture`: "start capture", narrate, "stop capture"; frames, captions (activity, phase, change) and narration become an evidence packet, the input for writing a guide that 10 can run; `scenarios/demo.yaml` |

Each app adds `Hub()` (the device connection: without it nobody joins, so no
turn reaches the app) and gets its speech models from the profile's
`speech_in` and `speech_out`; its `app.yaml` names the shared voice gate.

Real-speech model/hardware cases are provided for [guided procedure](10-guided-procedure/eval/speech/README.md) and [procedure capture](12-procedure-capture/eval/speech/README.md). Both use the [devkit real-audio runner](../../app-sdk/xr-ai-app/docs/REAL_AUDIO.md) and the samples' normal profiles and voice configuration.
