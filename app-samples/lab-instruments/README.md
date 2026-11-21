<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Lab instruments on the xr-ai-app core and kit

Upstream's [lab instrument monitoring sample](../../agent-samples/lab-instrument-monitoring/README.md)
rebuilt on the `xr_ai_app` core and its kit, `xr_ai_kit`: a deterministic
detector, then a model. Each instrument carries a printed QR or ArUco marker.
To read them, the app finds the markers in the camera frame with OpenCV (no
model), names each instrument from `yaml/device_map.yaml`, draws a colored X
on each marker, and asks the vision model once for the display under every
color. A participant can ask for a reading, start a monitor that speaks up
when an instrument appears, changes or is lost, start a silent visual watch,
and ask about the view. Everything the monitors note is written to JSONL
files. When the client streams no video, a read takes one picture (the web
client's on-demand camera mode).

`instruments.py` is the reading, top to bottom:

```python
frame = await p.frames.current(...)           # fresh frame, or one picture
markers = await scan_markers(frame.image)     # QR and ArUco, no model
known = [m for m in markers if (m.marker_type, m.value) in devices]
colored = assign_colors(known)                # magenta, cyan, ... top down
marked = annotate(frame, colored)             # a colored X on each marker
reply = await vlm.ask(marked, reading_query(colors), system=prompt)
readings = parse_readings(str(reply), colors)  # color -> its display
```

| Upstream | Here |
|---|---|
| `ForegroundAgent`: `run_tool_loop` on the omni LLM, nine tools, eight of them `return_direct` | the kit's `Conversation` on `llm` with the same nine `@tool`s (upstream's names, descriptions and argument limits) and `prompts/foreground_prompt.txt`; tools return values and the conversation speaks; the same eight are `@tool(return_direct=True)`, so their result is the answer without another LLM round, and `current_view` returns Cosmos's stream, said as it arrives (`app.py`) |
| `current_view`: `CurrentFrameTool` + `StreamingImageQueryTool`, the user's words asked of Cosmos and streamed | `p.frames.current()` and `vlm.stream(frame, turn.text, system=current_view_prompt)`, returned by the tool and spoken by the conversation as it streams |
| `LabInstrumentAgent`: `MarkerTrackingTool`, the device map, colored X's, one Cosmos call, upstream's parser | `instruments.py`: the same detection, through the foundation's plain `MarkerDetector` (`xr_ai_tools.marker_tracking`, which `MarkerTrackingTool` is built on) called on the frame's image in a thread; the same colors, query, prompt and reply checks |
| `InstrumentMonitorAgent` + `InstrumentAlertAgent`: per participant, scan every 5 s, say "Now tracking", "changed from", "no longer tracking"; a state snapshot every 10 s | `instrument_monitor.py`: one `p.spawn` task per participant, started and stopped by the tools, state in the task and `p.state`; the same messages, `normalize_meter_reading` and timings (`app.yaml`) |
| `MonitorAgent`: a silent visual watch with a focus, every 5 s | `visual_monitor.py`: one `p.spawn` task per participant, upstream's prompt (focus and caption passed as data under the system prompt) and parser; it never speaks |
| `FileOutputAgent`: per participant and session, `monitor.jsonl`, `instrument-monitoring.jsonl`, `foreground.jsonl`, `transcript.jsonl`; the `recent_visual_history` lookup | `records.py`: `p.artifacts.write_jsonl` with upstream's records and fields, `session` and `session_end` lines; `recent_visual_history` reads the visual monitor's last looks (`monitor_history_size`) |
| History only through `recent_visual_history` | also a `ContextLog`, "monitoring history" (readings, changes, losses, the watch's baseline and changes), which the conversation reads |
| `capture_marker_scans` | the same setting: each scanned frame under `marker-scans/` in the participant's artifacts |
| Health probes before ready | the SDK checks every role's server (and speech) before it reports ready |

The monitors are `p.spawn` tasks, not the kit's `Delegate`: a delegate's back
end gets the request but not the participant, so it cannot read their camera
(the same reason as the conversation sample's alert). A keyed task per
participant gives upstream's rules directly: start once, stop on request, end
when the participant leaves.

Left out:

- The live event viewer on :8092 (`web_events.py`) and `--expose-web-events`:
  it is a browser page fed by the foundation's runtime topics, which an
  `xr_ai_app` app does not publish. The JSONL files hold the same records;
  for a live view, add `trace: {jsonl: ...}` to `app.yaml` (every turn, tool
  call, model call and output, as it happens) or run the scenarios with
  `xr-ai-dev`, which writes `trace.jsonl` and `transcript.md`.
- `transcript.jsonl`: the voice pipeline's transcript events are not part of
  the SDK; the trace has each turn's text.
- `relay-events.jsonl`: the SDK's trace replaces it (`trace:` in `app.yaml`
  can also forward to Relay).
- Upstream's `eval/` (routing and visual evals against its worker modules):
  the scenarios below run the same paths on real models.

Two differences: a tool reading earlier notes can be skipped, because the
conversation already reads the monitoring history (asked "what changed
recently?", the omni model answered from it without calling
`recent_visual_history`); and simultaneous monitor messages are joined into
one utterance rather than combined by the LLM (upstream's
`llm_update_combiner`).

## Configure

`yaml/` holds upstream's files, copied unchanged except the
recordings folder in `yaml/media_capture.yaml`
(`xr-ai-app-lab-instruments`, so recordings do not mix with upstream's).
`prompts/` holds upstream's prompt texts. `app.yaml` takes the place of
upstream's `yaml/lab_instrument_monitoring_worker.yaml`, with the same values.

| File | Common changes |
|---|---|
| `app.yaml` | Monitor intervals, lost-instrument timeout, state snapshots, frame freshness, history sizes, marker-scan capture, artifacts directory, voice activity detection |
| `yaml/device_map.yaml` | QR or ArUco identifiers and their instrument names |
| `yaml/voice_gate.yaml` | Conversation controls, optional wake phrases, listening chime, follow-up window, speaker ASR |
| `yaml/models.json` | Model endpoints; the app's roles are its entry names, `llm`, `vlm` and `tts` |
| `yaml/device_io_hub.yaml` | Room, ports, web client, and network behavior |
| `yaml/media_capture.yaml` | Opt-in session recording |
| `prompts/` | The tool-routing, current-view, instrument and visual-monitor instructions (upstream's) |

Print the markers in
[`agent-samples/lab-instrument-monitoring/sample-markers/`](../../agent-samples/lab-instrument-monitoring/sample-markers/README.md)
(they match `yaml/device_map.yaml`) and stick one on each instrument.

## Run

Run all commands from `app-samples/lab-instruments/`. Start the
sample's local models first (the same stack as upstream's
lab-instrument-monitoring; servers that already run are reused):

```bash
uv run --project model-servers lab_instruments_model_servers
```

Wait for the launcher to report that all processes are ready and return. Then
start the sample:

```bash
uv sync
uv run lab_instruments
```

Session recording is off; `uv run lab_instruments --session-recording` turns
it on (recordings go to
`~/.local/share/xr-ai/captures/xr-ai-app-lab-instruments`).
The local models keep running after the sample stops; stop them with
`uv run --project model-servers lab_instruments_model_servers --stop`.

The launcher starts DeviceIOHub and the app (`lab_instruments_app`). Open the
authenticated web-client URL that DeviceIOHub prints, choose **Live video**
or **On-demand images** under **Camera Mode**, allow the microphone and
camera, and connect. After the agent reports ready, say "Start talking to
me", then speak without a wake phrase; typed questions work
without that. Try "What do the instruments read?", "Start monitoring the
instruments", "Keep an eye on the bench", "What changed recently?" or "What
am I looking at?". Durable output is written below `artifacts/`. Stop with
Ctrl+C.

## Test without a headset

`scenarios/` has nine scenarios, each with scripted replies
(`<name>.fakes.yaml`) and synthetic pictures in `scenarios/media/` (made by
`scenarios/make_images.py` from upstream's printable markers). Marker
detection always runs for real.

| Scenario | Checks |
|---|---|
| `read_one.yaml` | one instrument (QR) read from a streamed frame |
| `read_two.yaml` | two instruments (QR and ArUco) in one Cosmos call, from a picture taken on demand |
| `view.yaml` | a question about the view, answered by Cosmos |
| `unknown_marker.yaml` | a marker the device map does not name: no model call |
| `no_marker.yaml` | no marker in view: no model call |
| `unreadable.yaml` | a marked instrument whose display is off |
| `monitor_change.yaml` | the instrument monitor finds Device1, says once that its reading changed, stops |
| `monitor_lost.yaml` | the marker leaves the view; 30 s later the monitor says Device1 is lost, once |
| `watch.yaml` | the silent visual watch notes a change; asked later, the model answers from it |

```bash
uv --config-file ../../uv.toml run xr-ai-dev run app.py scenarios/ --out runs/
uv --config-file ../../uv.toml run pytest
uv --config-file ../../uv.toml run xr-ai-dev run app.py scenarios/ \
    --real-models --out runs/real/      # the model servers must be running
```

`test_app.py` plays the scenarios on the devkit's virtual clock and checks
what they show besides speech (the marked frame Cosmos got, one call for all
instruments, one picture on demand, each change said once, the JSONL
records, the monitoring history in the conversation's context);
`test_helpers.py` covers the detector on the committed pictures, the marked
frame, the reply checks and how readings are compared.

For speech replay and performance reports, see [`eval/speech/`](eval/speech/).
