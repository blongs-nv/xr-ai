<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Simple VLM on the xr-ai-app core

Upstream's [simple VLM example](../../agent-samples/simple-vlm-example/README.md)
rebuilt on the `xr_ai_app` core alone (no kit): the smallest complete
voice-and-vision application. A participant asks a spoken or typed question
about the current camera view. The app uses the newest video frame when the
client is streaming; otherwise it asks the client for one picture for that
turn (the web client's on-demand camera mode). The answer streams from the
vision model to Pocket TTS and to the web client.

`app.py` is the whole application: one turn handler,
`p.frames.current()` and `vlm.stream(frame, question)`. Everything else is the
SDK's job: it waits until the model servers answer (and warms the vision
model with one image) before it reports ready, shows "processing" while a
question is being answered, and asks for a picture when no frame is streamed.

## Configure

`yaml/` holds upstream's files, copied unchanged except the
recordings folder in `yaml/media_capture.yaml` (`xr-ai-app-simple-vlm`, so
recordings do not mix with upstream's).

| File | Common changes |
|---|---|
| `app.yaml` | The model profile and voice-gate files, voice activity detection |
| `yaml/voice_gate.yaml` | Conversation controls, optional wake phrases, listening chime, follow-up window, speaker ASR |
| `yaml/models.json` | Model endpoints; the app's roles are its entry names, `vlm` and `tts` |
| `yaml/device_io_hub.yaml` | Room, ports, web client, and network behavior |
| `yaml/media_capture.yaml` | Opt-in session recording |
| `prompts/system.txt` | The vision model's instructions (upstream's) |

## Run

Run all commands from `app-samples/simple-vlm/`. Start the sample's
local models first (the same stack as upstream's simple-vlm-example; servers
that already run are reused):

```bash
uv run --project model-servers simple_vlm_model_servers
```

Wait for the launcher to report that all processes are ready and return. Then
start the sample:

```bash
uv sync
uv run simple_vlm
```

Session recording is off; `uv run simple_vlm --session-recording` turns
it on (recordings go to
`~/.local/share/xr-ai/captures/xr-ai-app-simple-vlm`).
The local models keep running after the sample stops; stop them with
`uv run --project model-servers simple_vlm_model_servers --stop`.

The launcher starts DeviceIOHub and the app (`simple_vlm_app`). Open the
authenticated web-client URL that DeviceIOHub prints, allow the microphone and
camera, and connect. After the agent reports ready, say "Start talking to
me", then ask questions without a wake phrase; say "Stop talking to me" to
close the conversation. Typed questions work without
opening a conversation. Choose "On-demand images" in the web client to stream
no video: each question then takes one picture. Stop with Ctrl+C.

## Test without a headset

`scenarios/` has three scenarios, each with scripted replies
(`<name>.fakes.yaml`) and small synthetic pictures in `scenarios/media/`
(made by `scenarios/make_images.py`):

| Scenario | Checks |
|---|---|
| `typed.yaml` | a typed question about a streamed frame (a sign that says EXIT) |
| `spoken.yaml` | two spoken questions about a streamed frame (a red ball); the local device has no microphone, so each is the text the voice pipeline hands the app |
| `on_demand.yaml` | no stream: the app asks for a picture and answers from it |

```bash
uv --config-file ../../uv.toml run xr-ai-dev run app.py scenarios/ --out runs/
uv --config-file ../../uv.toml run pytest
uv --config-file ../../uv.toml run xr-ai-dev run app.py scenarios/ \
    --real-models --out runs/real/      # the model servers must be running
```
