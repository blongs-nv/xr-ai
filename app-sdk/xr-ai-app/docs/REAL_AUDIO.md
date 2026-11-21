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

`prepare` copies the source scenario’s camera media and synthesizes missing WAVs using the app's effective TTS role and writes `fixtures.json` with file hashes and utterance labels. Existing recordings are reused. Preparation refuses changed pinned files or labels; use a fresh directory when changing inputs. Each generated fixture/run directory ignores its contents in Git. Keep audiovisual fixtures and artifacts local.

Pass `--profile NAME_OR_PATH` to both commands to select another model configuration. Profile paths and role aliases follow the app's configuration rules. Local and hosted endpoints use the same adapters and credential store. Any supported role can change, including speech input/output; the app's effective Voice component and gate still determine which STT path is used. There is no hardcoded model list. New provider protocols belong in `xr_ai_models` adapters.

Reuse exactly the same media and timeline across hardware/model trials. `--metadata hardware.json` attaches an operator-supplied JSON object for hardware, quantization and serving versions. Use a fresh output directory per run. `--drain` adds response time after the source timeline (default 10 seconds); `--ready-timeout` bounds startup. The runner does not start or stop model servers.

## App sample cases

Each `app-samples/**/eval/speech/realtime.yaml` references an existing sample scenario:

| App | Existing scenario |
|---|---|
| Conversation | `scenarios/alert.yaml` |
| Lab instruments | `scenarios/monitor_change.yaml` |
| Tea making | `scenarios/guide.yaml` |
| Procedure capture | `scenarios/demo.yaml` |
| Guided procedure | `scenarios/checklist_on_demand.yaml` |

Speech schedules reuse those scenarios' prompts, media and assertions. Preparation copies the source media and synthesizes only the speech. Use the same recorded-media overrides as for the source accuracy case. The source cases' documented synthetic placeholders do not establish accuracy on real scenes. No separate speech-scene generator is needed.

The schedules distribute the source steps over five minutes and add enrollment/listening controls. Source images retain their streamed or on-demand camera behavior. Capture replays its existing frame directory at its declared frame rate. Additional accuracy cases remain available through the normal runner.

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

Real-audio mode supports one participant, join at zero, audio, streamed or on-demand image, sleep, expect/wait_for, and an optional leave at the end, deferred until the response drain finishes. Images are held and republished at 1 fps with a maximum dimension of 1280 pixels. `at` is a fixed source offset; `+s` is relative to the preceding planned step, independent of responses. Assertions observe completed voice text handoff after the preceding input without pausing the source timeline. Text matching is case-insensitive and respects word boundaries; `56` cannot match `256`, and `12 V` cannot match `12 very`. Observed response text has passed the voice floor checks and been handed to its TTS path; it does not establish completed audible playback. Use `unprompted: true` on an assertion to require an update rather than a conversational reply. Video and multiple participants remain supported by the local-device path and are rejected in real-audio mode.

A thin speech schedule selects a source scenario and assigns one absolute time to every source step:

```yaml
scenario: ../../scenarios/alert.yaml
duration_s: 300
schedule: [0, 35, 35.1, 65, 70, 100, 100.1, 170, 170.1, 300]
controls:
  - {at: 8, text: Start talking to me}
  - {at: 20, text: Start talking to me}
  - {at: 280, text: Stop talking to me}
```

Every source step is retained. `say` becomes a pinned audio clip with the same words; checks keep their source assertions and timeout lengths. Their deadlines are relative to the configured check time, capped by the replay and drain window; consecutive source waits need distinct scheduled horizons when a later check follows a timer. A schedule-length mismatch fails, so adding or removing a source step requires a timing update. Optional `unprompted` lists source check indices that require an unsolicited update rather than a conversational acknowledgement. Optional `no_reply` lists zero-based source `say` indices that intentionally do not produce a conversation reply. Controls are excluded from expected handler delivery. Frame folders use their existing images and frame rate; video remains unsupported by real-audio mode. Source media names remain relative to the source scenario, including when using `--media-dir` or `--image`.

A scenario file or folder is selected explicitly; there is no SDK registry of samples.

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

The runner uses `xr_ai_hub.isolation.isolated_endpoints` for private hub routing, `xr_ai_voice.diagnostics.observe` for voice health and completed text handoff, and `xr_ai_tools.tool_calling.observe_tool_errors` for raised tool failures. These scopes are opt-in, inherited by child tasks, and restored on exit. Normal app runs do not install observers or isolated routing. The standard hub address pair is exported as `xr_ai_hub.isolation.DEFAULT_ENDPOINTS` so routing and its callers share one default. On-demand replay answers capture requests through `ConnectorEndpoint.push_image_capture(...)`, the public wrapper for the connector’s existing encoded-image path.
