<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Lifecycle contract

This file is the written contract for the `xr_ai_app` app (the private
`xr_ai_app._base` modules implement it; `xr_ai_app._base.testing.run_lifecycle_contract` checks it). The ordered
steps and timeout defaults also exist as constants in `xr_ai_app._base.scope` and `xr_ai_app._base.testing`;
`tests/app/test_contract_surface.py` fails if this file and those constants disagree. Approved wording is frozen;
change it only through a reviewed contract change.

Vocabulary: the **app** is an `xr_ai_app.App`; a **component** is an `xr_ai_app.Component` added to it with
`app.add`; a **scope** is the per-participant `ParticipantScope`; a **hook** is an `on_*` method of a component.
"Registration order" is the order of `app.add` calls (`@app.on_start` adds its function in the order the decorators
run).

## The eight rules

Each rule has an identifier (used by the contract suite) and numbered statements. Every statement
is a single observable fact a test can assert on `app.test_session()` with fakes.

### `R1_task_ownership`

Every task belongs to the app, a component or a participant scope. App and sample code never calls
`asyncio.create_task`: work that outlives the call goes through `p.spawn`, and concurrency awaited in
place uses `asyncio.TaskGroup` or `asyncio.gather`, whose tasks belong to the task that awaits them (R1.9).

- R1.1 `p.spawn(coro)` runs `coro` in that participant's scope and returns its `asyncio.Task`.
- R1.2 `p.spawn(coro, key=k, supersede=True)` cancels and drains a live task with key `k` before
  starting the new one; with `supersede=False` and a live task under `k` it raises `ValueError`.
- R1.3 `p.cancel(k)` cancels the task under `k`, waits for it to finish and returns `True`; it returns
  `False` when no task is live under `k`.
- R1.4 `p.spawn` after the scope stopped admitting raises `RuntimeError` (app code that
  reaches this has a bug; it is never silently dropped) and never starts the coroutine.
- R1.5 Turns run as tasks in the scope of their participant, so R2 covers them.
- R1.6 Apps and components own app-level tasks the same way: they are started only through the app, and
  they are cancelled and drained in the stop sequence.
- R1.7 Each task runs under its own Relay scope, and a task's events never attribute to another task.
- R1.8 `p.spawn` runs the coroutine in a copy of the caller's context (the Relay context is forked). A
  spawned task is cancelled only by the participant leaving, `p.cancel`, a superseding spawn under the
  same key, or the app stopping; a later turn never cancels it.
- R1.9 While an app runs, every task created on its event loop without an explicit `context` gets a
  forked Relay context too, so app code may use `asyncio.TaskGroup` and `asyncio.gather`: concurrent
  sibling tasks never share a Relay scope stack. Work awaited inside a turn handler, including such
  sub-tasks, is cancelled with the turn (by a newer turn or an interruption).
- R1.10 `Component.spawn(coro, key=None, supersede=False)` runs component-lifetime work: it belongs to the component, not to a
  participant, so it survives every un-enrollment. The app cancels and drains it in step 2 of the stop sequence, after every participant
  is un-enrolled and before the `on_stop` hooks. Keys are per component. It raises `RuntimeError` when the component is not added to an
  app or the app is stopping.

### `R2_unenroll_sequence`

A participant exists from enrollment to un-enrollment (`COMPOSITION_CONTRACT.md` section 5b): the app's enrollment setting
(`PerConnection`, or `Persistent` with `keep_for_s`) decides when a connecting identity becomes a participant and when it stops
being one. A disconnect alone does not end a `Persistent` participant. When a participant is un-enrolled, `UNENROLL_SEQUENCE`
runs, in this order:

1. `stop_admitting_turns`: new queries and `p.spawn` calls for this participant are
   rejected from now on.
2. `cancel_participant_tasks`: every task in the scope is cancelled (turns and spawned tasks).
3. `drain_participant_tasks`: wait for those tasks to finish, at most `drain_s` seconds in total.
4. `run_unenrolled_hooks_reversed`: each component's `on_participant_unenrolled(p)` runs, last registered first,
   each bounded by `hook_s`.
5. `release_frames_state_artifacts`: the artifact sink drains, then `p.frames` and `p.state` are released.

- R2.1 The five steps happen in the order above (the suite records them through a trace).
- R2.2 After step 1 no new turn for that participant starts, and a turn already waiting is dropped.
- R2.3 Hooks see `p.state`, `p.frames` and `p.artifacts` intact; after step 5 they are released.
- R2.4 A participant that is enrolled again with the same id after un-enrollment gets a fresh scope, fresh state and no stale tasks. A second
  connection of an enrolled identity is another connection of the same participant, not a new one.
- R2.5 Un-enrolling participant A never cancels or delays a task of participant B.
- R2.6 A task that ignores cancellation is abandoned after `drain_s`: the un-enroll sequence logs it,
  reports it on the participant's status and continues. It is never left to publish after step 5.
- R2.7 A hook that raises or exceeds `hook_s` is logged and reported; the remaining hooks still run.
  `Component.on_start` is bounded separately by `start_s` (default `hook_s`), so a long warm-up does not widen
  the un-enroll and stop hook budgets; a failing or overrunning `on_start` still aborts startup (R3.3).
- R2.8 `p.say` called in `on_participant_enrolled` is queued until output is ready and is then audible.
- R2.9 A finite `p.say` or `p.reply` is not cut by a partial transcript of the user; only `urgent`
  output or an interrupt cuts it.
- R2.10 A disconnect that does not un-enroll (`Persistent`) cancels nothing: `p.spawn` work lasts from enrollment
  to un-enrollment. Work that must stop on disconnect checks `p.connected` and ends itself.

### `R3_stop_sequence`

When the app stops (`app.stop()`, or `app.run()` after a signal), `STOP_SEQUENCE` runs:

1. `stop_admitting`: no new participants, turns or app-level tasks are accepted. A pending start task
   is cancelled and drained before app-level tasks.
2. `unenroll_every_participant`: the full un-enroll sequence (R2) runs for every participant, concurrently,
   each with its own `drain_s`. Then work started with `Component.spawn` (R1.10) is cancelled and drained.
3. `run_stop_hooks_reversed`: each component's `on_stop()` runs, last registered first, each bounded by `hook_s`.
4. `close_model_clients`: every model client the app created is closed, each bounded by `close_s`.
5. `stop_runtime`: the in-process `AgentRuntime` stops and its delivery tasks drain.

- R3.1 The five steps happen in this order, and no step starts before the previous one finished or timed out.
- R3.2 No task publishes into the runtime after step 5 begins; nothing raises `RuntimeClosedError`
  into app code.
- R3.3 `on_start` hooks run in registration order before any participant is admitted; if one fails,
  `app.start()` runs the stop sequence for what already started and re-raises.
  A caller awaiting `start()` receives `RuntimeError("app stopped during start")` if stop cancelled startup;
  cancellation of the caller itself still propagates as `CancelledError`.
- R3.4 `app.stop()` is idempotent: a second call returns after the first finishes.
- R3.5 Clients are closed only after every task that can use them has drained.

Timeout defaults (`ScopeTimeouts`, configurable under `timeouts:` in `app.yaml`):

- `drain_s` = 5.0, wait for a participant's cancelled tasks.
- `hook_s` = 5.0, per `on_participant_unenrolled` or `on_stop` hook. It is the un-enroll-hook budget: there is
  no per-component budget, so each hook of each component gets `hook_s` on its own.
- `close_s` = 10.0, per model client.
- `start_s` = `None` (meaning `hook_s`), per `Component.on_start` hook; set it to cover a slow warm-up.

### `R4_error_containment`

An exception in a callback or task is logged, reported on the participant's status and contained to
its scope. Nothing calls `os._exit`.

- R4.1 An exception raised by a turn or spawned task is logged with the participant id and task
  key, an `error` status is reported for the participant (a test session's hub records it), and sibling tasks
  of the same participant keep running.
- R4.2 An exception in one participant's scope never affects another participant.
- R4.3 An exception in a hook is handled as in R2.7, and an exception in `on_start` as in R3.3.
- R4.4 An exception in a hub callback (`ProcessorEndpoint`) reaches the owning scope through the seam
  the app provides and never terminates the process.
- R4.5 `CancelledError` is not an error: it is never logged as one and never reported on status.
- R4.6 After R4.1 the failed task ends with its exception: code that awaits the task returned by
  `p.spawn` gets that exception, and a task nobody awaits never logs "exception was never retrieved".

### `R5_cancel_turn_cancels_model_stream`

Cancelling a turn cancels its model stream (fixes issue #697).

- R5.1 When a turn is superseded, interrupted or its participant is un-enrolled, the in-flight `Role.stream`
  or tool-loop model call of that turn is closed within `drain_s`.
- R5.2 A `@tool` call runs in the turn's scope, so cancelling the turn cancels the tool call and any
  model stream the tool started.
- R5.3 A fake model that records stream closure sees exactly one close per cancelled call.

### `R6_no_cancelled_or_closed_errors_in_app_code`

Sample code never catches `CancelledError` or `RuntimeClosedError`.

- R6.1 No SDK call made from app code raises `RuntimeClosedError`: `p.say`, `p.reply`, `p.withdraw`
  and `p.artifacts.*` become silent no-ops after the participant's scope stopped admitting (they never
  raise into a task that is already being cancelled).
- R6.2 SDK APIs that await model or hub work propagate `CancelledError` unchanged, so app code
  needs no `try/except CancelledError` and no `shield`.
- R6.3 A lint rule rejects `asyncio.create_task`, `except CancelledError` and
  `except RuntimeClosedError` in sample code.

### `R7_only_run_installs_signal_handlers`

Only `App.run()` installs signal handlers.

- R7.1 `app.start()`, `app.stop()`, `app.test_session()` and every SDK component install none.
- R7.2 `App.run()` handles `SIGINT` and `SIGTERM` by awaiting `app.stop()` once; a second signal
  during stop forces exit after the `close_s` bound.
- R7.3 The voice pipeline is stopped by the app (through its stop seam), not by its own handler. R8.1 generalizes this.

### `R8_composition`

Components are composed on purpose: hooks run in a defined order, output always names a registered sink, and the app checks the
composition before it starts (`COMPOSITION_CONTRACT.md`, epic C). The suite check of this rule is a pending placeholder until
the `Sink`, `Hub`, `Voice`, `Conversation` and `Router` components exist.

- R8.1 An agent that exposes `request_stop()` (the voice agent) is asked to stop before the un-enroll sequence of any participant, so
  input ends first. This is the documented generalization of R7.3, not a special case for one class.
- R8.2 Hooks follow registration order on start and reverse order on un-enroll and stop. Register sinks first, then
  non-routable components, then conversations, then the router. Stop therefore runs the router first (it drains), then conversations (they
  close their contexts), then sinks last. Pending: `start()` does not check this order yet (the hooks run in the order of the
  `app.add` calls, whatever it is); once implemented, it raises an error if a `Conversation` or `Router` is registered before the
  sink it writes to.
- R8.3 Output addressed to a sink is never published to a topic with no subscriber and never dropped silently: `to=sink` calls the
  sink object directly. Pending: a sink that is not registered is not rejected yet (a `TextSink`, which is not a component,
  receives the output as given); once implemented, the call raises an error naming the sink.
- R8.4 Connection events (`PARTICIPANT_CONNECTED`/`PARTICIPANT_DISCONNECTED`) have one publisher, the `Hub` component; what they mean
  for participants is the app's enrollment setting, seen by components through their enrollment hooks. The voice agent never publishes
  presence.
- R8.5 `start()` validates the composition (`COMPOSITION_CONTRACT.md` section 11) before any task starts and reports every problem at once.
  Pending: `start()` validates no composition today. Only single components are checked: `app.add` takes a `Component`, once,
  before `start()`, and a `Component` class with a foundation `@subscribe` handler fails when it is defined.
- R8.6 An admission decision (`admit`) runs before any output and is given no sink.

## Standard topics

`xr_ai_app._base.topics` (private: the app's own wiring between the hub, the voice pipeline and participants; components do
not subscribe to topics): `USER_QUERY`, `PARTICIPANT_CONNECTED`, `PARTICIPANT_DISCONNECTED`, `INTERRUPTED`, `OUTPUT`. Names are
`xr_ai_app.user_query`, `xr_ai_app.participant_connected`, `xr_ai_app.participant_disconnected`, `xr_ai_app.interrupted`,
`xr_ai_app.output`. Payload types are the `xr_ai_voice` message types on this base.

## Test checklist (one check per rule, see `xr_ai_app._base.testing.LIFECYCLE_RULES`)

`R1_task_ownership`, `R2_unenroll_sequence`, `R3_stop_sequence`, `R4_error_containment`,
`R5_cancel_turn_cancels_model_stream`, `R6_no_cancelled_or_closed_errors_in_app_code`,
`R7_only_run_installs_signal_handlers`, `R8_composition`.
