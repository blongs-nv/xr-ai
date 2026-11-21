<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Conversation on the xr-ai-app core and kit

Upstream's [conversation example](../../agent-samples/conversation-example/README.md)
rebuilt on the `xr_ai_app` core and its kit, `xr_ai_kit`: a voice-and-vision
assistant whose turns Clef routes to four destinations. A participant can
talk about anything, ask about what the camera shows now, take notes that
the conversation remembers, and ask to be told once when something becomes
visible. When the client streams no video, a question about the view takes
one picture (the web client's on-demand camera mode). Answers stream to
Pocket TTS and to the web client.

| Upstream | Here |
|---|---|
| `xr_ai_routing.AgentRouter` over Clef | the kit's `Router` with `decide=app.role("router")` (`app.py`) |
| `Conversation` + `ContextManager` (dialogue and notes, half the budget each) | the kit's `Conversation` on `llm`, `context=[voice.dialogue, notes]`: the voice dialogue log and a `notes` `ContextLog`, each folded by `llm` when it outgrows its budget; routed as `OneShotDestination("conversation", <upstream's routing text>, conversation.answer)` |
| `VisualObservationAgent` (`CurrentFrameTool` + `StreamingImageQueryTool`) | `current_view.py`: `p.frames.current()` (a fresh frame, else one picture on demand) and `vlm.stream(frame, question)` |
| `NoteTakingAgent`: Clef classifies, holds focus, hands questions off, shares each note with the conversation | `note_taking.py` (a custom destination, a `RouterDestination` the component builds in `__init__` from its own methods: upstream's note prompt names five actions, readback among them, which the kit's `StartStopDestination` does not have): Clef classifies in the destination's admission check (`admit=`; a question is declined, so the router answers it elsewhere); focus is the destination's own state (`focused=`); each note goes to the `notes` log |
| `VisualAlertAgent`: compile the request, then per frame a Clef relevance screen and a race of a Clef image decision against a Cosmos observation judged by Clef; "Alert" (urgent) and a description, once | `visual_alert.py`: the same admission (Clef) and compile (LLM) steps, then one Clef image decision per new frame in a background task (`p.spawn`); "Alert" (urgent) and Cosmos's description of that frame, once. Kept: upstream's prompts, its decision criteria and spoken texts, one alert per participant. Left out: the relevance screen and the race (they save Cosmos calls and add a second opinion; the check stays one call). Changed: the request is compiled with reasoning on (upstream turns it off for speed): without it the omni model answered "unclear" to about half of the clear requests tried on this stack, upstream's exact input included; with it, to none |
| Route-aware acknowledgment, off by default | the kit's `Ack` on `llm`, off by default (`settings.ack_delay_s`) |
| Health probes before ready | the SDK checks every role's server (and speech) before it reports ready |

The alert is a `p.spawn` task, not the kit's `Delegate`: a delegate's back end
gets the request but not the participant, so it cannot read their camera, and
it speaks its result as an ordinary (not urgent) message and keeps it in a
results log. One keyed task per participant gives upstream's rules directly: a
new alert replaces the running one, "cancel" stops it, and it ends when the
participant leaves. The alert checks the frames the client streams
(`p.frames.every`); in on-demand camera mode `every()` takes one picture per
check instead, as upstream does.

## Configure

`yaml/` holds upstream's files, copied unchanged except the
recordings folder in `yaml/media_capture.yaml` (`xr-ai-app-conversation`, so
recordings do not mix with upstream's).
`prompts/` holds upstream's prompt texts. `app.yaml` takes the place of
upstream's `yaml/conversation_example_worker.yaml`, with the same values.

| File | Common changes |
|---|---|
| `app.yaml` | The model profile and voice-gate files, voice activity detection, frame freshness, context and routing-history budgets, the optional acknowledgment, the alert's check interval |
| `yaml/voice_gate.yaml` | Conversation controls, optional wake phrases, listening chime, follow-up window, speaker ASR |
| `yaml/models.json` | Model endpoints; the app's roles are its entry names, `llm`, `router`, `vlm` and `tts` |
| `yaml/device_io_hub.yaml` | Room, ports, web client, and network behavior |
| `yaml/media_capture.yaml` | Opt-in session recording |
| `prompts/` | The conversation, current-view, note and alert instructions (upstream's) |

## Run

Run all commands from `app-samples/conversation/`. Start the sample's
local models first (the same stack as upstream's conversation-example;
servers that already run are reused):

```bash
uv run --project model-servers conversation_model_servers
```

Wait for the launcher to report that all processes are ready and return. Then
start the sample:

```bash
uv sync
uv run conversation
```

Session recording is off; `uv run conversation --session-recording` turns
it on (recordings go to
`~/.local/share/xr-ai/captures/xr-ai-app-conversation`).
The local models keep running after the sample stops; stop them with
`uv run --project model-servers conversation_model_servers --stop`.

The launcher starts DeviceIOHub and the app (`conversation_app`). Open the
authenticated web-client URL that DeviceIOHub prints, choose **Live video**
or **On-demand images** under **Camera Mode**, allow the microphone and
camera, and connect. After the agent reports ready, say "Start talking to
me", then speak without a wake phrase; say "Stop talking to me" to close the
conversation. Typed questions work without opening a conversation. Try "What am I looking at?", "Start taking notes", "Let me know
when the door is open", or any question. Stop with Ctrl+C. An alert needs a
condition one still picture can show: "when the kettle boils" is an event, so
the app asks what to watch for instead.

## Test without a headset

`scenarios/` has five scenarios, each with scripted replies
(`<name>.fakes.yaml`) and small synthetic pictures in `scenarios/media/`
(made by `scenarios/make_images.py`):

| Scenario | Checks |
|---|---|
| `general.yaml` | a general question, answered by the conversation |
| `view.yaml` | a question about a streamed frame (a sign that says CLOSED) |
| `view_on_demand.yaml` | the same with no stream: the app asks for one picture |
| `notes.yaml` | a note taken, a question in between answered from the notes, note mode left, the note recalled afterwards |
| `alert.yaml` | "let me know when the sign says OPEN": no alert while the sign says CLOSED, the conversation goes on, then the picture changes to OPEN and the app says "Alert" and what it sees |

```bash
uv --config-file ../../uv.toml run xr-ai-dev run app.py scenarios/ --out runs/
uv --config-file ../../uv.toml run pytest
uv --config-file ../../uv.toml run xr-ai-dev run app.py scenarios/ \
    --real-models --out runs/real/      # the model servers must be running
```

For speech replay and performance reports, see [`eval/speech/`](eval/speech/).
