---
name: test-with-local-device
description: Test an xr-ai app end to end without a headset, hub or LiveKit by driving it like a user - typed turns, images, image folders and videos as timestamped frames - and reading back what it said and its trace. Use when checking or debugging an xr_ai_app app, writing an end-to-end scenario for it, reproducing a bug, or comparing behaviour before and after a prompt or code change.
---

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Test an app with the local device

The local device (`xr_ai_app.devkit`) runs an app in-process with a fake device side in place of the hub. You
act as the user; the app runs unchanged.

## Choose the models first

- **Scripted (default, no GPU):** a YAML of replies per role: `<scenario>.fakes.yaml` beside the scenario is used
  automatically (or `fakes.yaml` there, or `--fakes FILE`). A reply is text, a mapping (a structured answer) or
  `{tool_call: {name: ..., args: {...}}}`; a role called by two components at once answers by prompt
  (`role: {by_prompt: [[text in prompt, reply], ...], otherwise: reply}`; a decision role's context, the
  request it judges, is matched too). With no script every model call fails
  loudly, which shows which roles a scenario needs.
- **Real:** `--real-models` uses the app's `models.json` endpoints. Only use it when the user asked for a real
  run and the model servers are theirs and already running; never start model servers yourself, and never point
  an app at servers you did not confirm.

## Run a scenario (repeatable, gives a report)

```yaml
# scenario.yaml (media paths resolve against this file)
name: mug color
participant: p1
steps:
  - join: p1
  - image: media/mug.jpg                 # or frames: {dir: media/seq/, fps: 1}
                                         # image: {path: x.jpg, stream: false}: held, sent only when asked for
  - at: 0.5                              # or video: {path: clip.mp4, fps: 2, from_s: 0, to_s: 10}
    say: What color is my mug?           #   (add background: true to talk over a playing clip)
  - expect: {said: red, within_s: 10}    # wait_for: same check; said_any: [red, crimson] for alternatives
  - at: +1
    say: thanks
  - leave: p1
```

```bash
cd app-samples/sdk-tour/02-vision-qa   # the app's own uv project (its pyproject.toml)
uv --config-file ../../../../uv.toml run xr-ai-dev run app.py scenarios/mug.yaml --out runs/mug/ [--relay]
uv --config-file ../../../../uv.toml run xr-ai-dev run app.py scenarios/ --out runs/   # every scenario, a summary
uv --config-file ../../../../uv.toml run xr-ai-dev run app.py scenarios/ --real-models \
    --media-dir ~/recordings/02 --out runs/real/   # real models, the user's own recordings (same file names)
```

Every sample in `app-samples/` has such a `scenarios/` folder (see its README for what each evaluates and
what to record). `--clock virtual` skips idle waits inside the app (look intervals, timers) for scripted runs.

An app without its own `pyproject.toml` can run from the `tests/` project instead:
`cd tests && uv --config-file ../uv.toml run xr-ai-dev run path/to/app.py scenario.yaml ...`.

Exit code 0 means every `expect`/`wait_for` passed. Read `runs/mug/transcript.md` first (who said what, when,
and the failed checks), then `report.json`, then `trace.jsonl` for why: each event has `kind` (`turn`, `route`,
`role`, `tool`, `context`, `output`, `job`, `spawn`...) and the `turn`/`task`/`parent` ids that link it to its
cause. `--relay` also writes NeMo Relay `atof/` and `atif/` for NeMo Compass or Helix.

## Drive a live app step by step

```bash
uv --config-file ../../../../uv.toml run xr-ai-dev serve app.py --fakes scenarios/mug.fakes.yaml --port 8765 &
curl -s -XPOST localhost:8765/join -d '{"participant": "p1"}'
curl -s -XPOST 'localhost:8765/frame?participant=p1' --data-binary @scenarios/media/mug.jpg
curl -s -XPOST localhost:8765/say -d '{"text": "What color is my mug?"}'
curl -s 'localhost:8765/output?after=0&wait_s=5'
curl -s 'localhost:8765/trace?kind=tool'
```

Stop the server when done.

## Exercise one piece on a window of a video

Run ONE piece of the app through its real code, the app unchanged: any `async fn(p, probe)` on the part of a
recorded demonstration where it should happen, or a router destination on one turn.

```bash
cd app-samples/sdk-tour/10-guided-procedure
uv --config-file ../../../../uv.toml run xr-ai-dev exercise app.py fn:units.py:look \
    --video demo.mp4 --from 12 --to 31.5 --fakes fakes.yaml \
    --expect Done --between 14 30 --out runs/look/
```

`units.py` is a file of your own next to `app.py`; `look` returns a `Done` when it has seen what it looks for.

| Target | Runs |
|---|---|
| `destination:<name>` | a router destination (a `RouterDestination` attribute of a component, such as a `SopRunner` control, or a `OneShotDestination`/`StartStopDestination` in the app's Router) on one turn: `--text "..."` (registered by `xr-ai-kit`, so the app must depend on it) |
| `fn:<file.py or module>:<attr>` | any `async fn(p, probe)` |

`--setup file.py:fn` runs `fn(p, probe)` first (fill a context channel, set `p.state`); `--say 13.5:text` types a
turn into the app mid-window. `exercise.json` gives the outcome (the result's type name, `returned`,
`unfinished`), the clip time it ended at (check it against an annotated earliest/latest with `--between`), the
unit's report lines with clip times, what was said, and the model calls; `trace.jsonl` has every event. Exit 0
when `--expect`/`--between` hold. Playing a video needs `xr-ai-app[video]`. With scripted roles it runs on a
virtual clock (`--clock virtual`, the default then): idle waits such as look intervals and timers are skipped, so a
long window takes moments; real model calls still take real time. `--real-models` defaults to `--clock real`. In Python,
one step of an `xr_ai_kit.sop` procedure on its own, with a `Run` as its environment (`m` is the loaded app module):
`await exercise(m.app, lambda p, probe: m.bed_clear.run(Run(p, 1, name="x")), video=..., from_s=..., to_s=...,
models=...)`.

## In tests

```python
from xr_ai_app.devkit import LocalDevice, Scenario, run_scenario
from xr_ai_app.testing import FakeModels

report = await run_scenario(app, Scenario.load("scenario.yaml"), models=FakeModels({...}))
assert report.passed, report.transcript()
```

Background work that waits on time (monitors, timers, delegated jobs) needs `wait_for`/`expect` with a
`within_s`, not an immediate check.

## Microphone-speed replay

For speech throughput and overload checks, use `xr-ai-dev prepare APP SCENARIO --out MEDIA` followed by `xr-ai-dev run APP SCENARIO --real-audio --real-models --clock real --media-dir MEDIA --out RUN`. Scenarios use `audio: {path: clip.wav, text: spoken words}`, optional `duration_s`, and speech schedules that reference an existing `scenario` with a `schedule` time for each source step. See [Real audio evaluation](../../app-sdk/xr-ai-app/docs/REAL_AUDIO.md) for coverage, timing eligibility, hosted profiles, and artifacts. Keep accuracy-only `say`/image/video evaluations for model accuracy independent of throughput.
