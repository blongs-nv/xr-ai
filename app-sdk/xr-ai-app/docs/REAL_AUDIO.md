<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Real audio evaluation

`xr-ai-dev run --real-audio` extends the App SDK's scenario runner with microphone-speed speech and camera input. The app uses its normal lifecycle, model roles, voice component and speech gate. An isolated hub and a separate sender process keep the input timeline independent of app/model delays. Model services remain caller-owned.

## Prepare and run

Install `xr-ai-app[audio]` in the app project, or use this repository's `tests` project. Start the sample's model services or select a profile containing hosted endpoints. From the repository root:

```bash
uv --config-file uv.toml run --project tests xr-ai-dev prepare app-samples/conversation/app.py \
    app-samples/conversation/eval/speech/realtime.yaml \
    --out /path/to/fixed-fixtures
uv --config-file uv.toml run --project tests xr-ai-dev run app-samples/conversation/app.py \
    app-samples/conversation/eval/speech/realtime.yaml \
    --real-audio --real-models --clock real \
    --media-dir /path/to/fixed-fixtures --out /path/to/new-run
```

`prepare` synthesizes missing WAVs using the app's effective TTS role and writes `fixtures.json` with file hashes and utterance labels. Existing recordings are reused. Preparation refuses changed pinned files or labels; use a fresh directory when changing inputs. Each generated fixture/run directory ignores its contents in Git. Keep audiovisual fixtures and artifacts local.

Pass `--profile NAME_OR_PATH` to both commands to select another model configuration. Profile paths and role aliases follow the app's configuration rules. Local and hosted endpoints use the same adapters and credential store. Any supported role can change, including speech input/output; the app's effective Voice component and gate still determine which STT path is used. There is no hardcoded model list. New provider protocols belong in `xr_ai_models` adapters.

Reuse exactly the same media and timeline across hardware/model trials. `--metadata hardware.json` attaches an operator-supplied JSON object for hardware, quantization and serving versions. Use a fresh output directory per run. `--drain` adds response time after the source timeline (default 10 seconds); `--ready-timeout` bounds startup. The runner does not start or stop model servers.

## App sample cases

| App | Real-speech scenario | Behavior |
|---|---|---|
| Conversation | `app-samples/conversation/eval/speech/realtime.yaml` | Conversation memory, camera, notes, visual alert and arithmetic |
| Lab instruments | `app-samples/lab-instruments/eval/speech/realtime.yaml` | Marker-based readings, changed readings and monitoring |
| Tea making | `app-samples/tea-making/eval/speech/realtime.yaml` | Package identification, camera-driven water/heating checks, conversation and timer start |
| Procedure capture | `app-samples/sdk-tour/12-procedure-capture/eval/speech/realtime.yaml` | Two spoken start/stop cycles, narration and changing camera scenes |
| Guided procedure | `app-samples/sdk-tour/10-guided-procedure/eval/speech/realtime.yaml` | Printer bed/door checks, warm-up and questions about progress |

Each sample owns its five-minute scenario and assertions. Pass its app file and scenario to the commands above. Simple VLM is covered by its accuracy scenarios; XR has no App SDK equivalent yet.

Conversation and Capture generate simple synthetic camera scenes. Lab uses its own accuracy-scenario instrument renderer and shared printable marker assets. Tea requires caller-supplied images of a legible tea package, water visibly inside a kettle, a readable hot-water display, and tea touching water in a physical cup. The printer checklist generates deterministic occupied-bed, clear-bed/open-door and clear-bed/closed-door scenes, with fixed geometry and no answer labels. Its README lists the optional `--image NAME=PATH` overrides for real printer photographs. Each scenario's adjacent README lists its image bindings. Results with generated scenes apply to those scenes; use recorded photographs for physical-camera qualification. Overrides take precedence over scene factories and their actual bytes are hashed.

Tea's procedure advances automatically on camera evidence. Its five-minute speech test checks timer start and stops the guide; steep completion depends on the package's duration and is covered by the accuracy scenarios.

## Accuracy without real-time cost

`say`, image/video, fake-model, `--relay`, fast-clock and virtual-clock scenarios evaluate accuracy without real-time speech. Use them with `--real-models --profile` for a hosted frontier model when measuring solution accuracy independently of speech throughput:

```bash
uv --config-file uv.toml run --project tests xr-ai-dev run app-samples/conversation/app.py \
    app-samples/conversation/scenarios/general.yaml \
    --real-models --profile /path/to/hosted-profile.json --clock virtual \
    --out /path/to/accuracy-run
```

The CLI loads the normal credential store before importing the app when `xr-ai-launcher` is installed (included in the `audio` extra). Without it, environment credentials work and the CLI prints how to enable saved-store credentials. The real-audio path requires `--real-models --clock real --speed 1`; it rejects virtual execution.

## Scenario contract

```yaml
name: speech under visual load
participant: p1
duration_s: 300
steps:
  - join: p1
  - image: blue.png
  - at: 0
    audio: {path: enroll.wav, text: Start talking to me, handler: false}
  - at: 25
    audio: {path: question.wav, text: What color is the rectangle?}
  - at: 25.1
    expect: {said_any: [blue, Blue], within_s: 20}
```

Audio must be uncompressed PCM16 mono/stereo WAV; preflight converts it to 16 kHz mono, and the sender sends 20 ms chunks, including silence between utterances and during the drain. `text` labels the recording; it is never injected as a query. Optional `audio.sha256` pins a WAV before startup. Clips cannot overlap after chunk alignment.

`handler: false` identifies enrollment/voice controls that should not become app turns. `reply: false` identifies handler inputs such as narration or procedure controls that intentionally have no request-matched conversation reply. These declarations affect coverage expectations, not app behavior.

Real-audio mode supports one participant, join at zero, audio, streamed image, sleep, expect/wait_for, and an optional leave at the end, deferred until the response drain finishes. Images are held and republished at 1 fps with a maximum dimension of 1280 pixels. `at` is a fixed source offset; `+s` is relative to the preceding planned step, independent of responses. Assertions observe completed voice text handoff after the preceding input without pausing the source timeline. Text matching is case-insensitive and respects word boundaries; `56` cannot match `256`, and `12 V` cannot match `12 very`. Observed response text has passed the voice floor checks and been handed to its TTS path; it does not establish completed audible playback. Use `unprompted: true` on an assertion to require an update rather than a conversational reply. Video, frame folders, on-demand images and multiple participants remain supported by the local-device path and are rejected in real-audio mode.

Optional `fixtures: {factory: "fixtures.py:prepare", options: {...}}` calls trusted sample-owned Python code with an output directory. Generated images are staged and cannot overwrite pinned fixture bytes. A scenario file or folder is selected explicitly; there is no SDK registry of samples.

## Results

`report.json`, `transcript.md` and `trace.jsonl` remain. Real-audio reports add `performance`, `audio-events.jsonl`, `speech-events.jsonl` and a canonical `scenario.json`:

- Expected-input handler delivery and completed-reply coverage, linked by source audio timestamps and app turn IDs. Ambiguous timestamps remain unattributed. Matching allows 160 ms lead for speaker encoder hops.
- Median/worst handler delivery and completed reply-text latency from the marker before sending the final audio chunk. Missing replies are not counted as fast responses. Completed text is not completed spoken playback. Runtime streams closed by timeout, eviction or shutdown without a producer terminator are incomplete: they emit no response observation and cannot satisfy output assertions.
- First observed non-silent returned audio. This may be a chime, acknowledgment or previous output; it is not request-matched model time-to-first-token.
- Sender lateness and observed speaker-STT queue depth, drops and resets. Unsupported measurements, including GPU peak memory, are `unmeasured`.
- Profile/settings/gate/fixture hashes, model aliases/hosts and role mappings, source revision/diff/SDK identity and operator metadata. Profile credentials and raw input media are excluded.

For real audio, `passed` requires successful scenario assertions and a timing-comparable run. Individual step results retain the functional checks when timing is invalid. `timing_comparable` requires full expected handler/reply coverage, no unattributed or unexpected control turns, no observed errors or STT drop/reset, and sender lateness at most 100 ms. Incomplete coverage cannot silently become a successful timing comparison. Normal cancellation, startup and teardown are excluded from error counts; errors recovered by the app may still count. Observed error events can include multiple records of the same failure. Semantic errors without a trace/log failure need sample assertions or accuracy evals.

Reports from interrupted runs are retained before cancellation propagates. Only one replay may run per process, and an existing output directory is refused. Native app artifacts can contain media even though the performance report does not; keep the whole run directory out of source control.

## Shared SDK interfaces

The runner uses `xr_ai_hub.isolation.isolated_endpoints` for private hub routing, `xr_ai_voice.diagnostics.observe` for voice health and completed text handoff, and `xr_ai_tools.tool_calling.observe_tool_errors` for raised tool failures. These scopes are opt-in, inherited by child tasks, and restored on exit. Normal app runs do not install observers or isolated routing. The standard hub address pair is exported as `xr_ai_hub.isolation.DEFAULT_ENDPOINTS` so routing and its callers share one default.
