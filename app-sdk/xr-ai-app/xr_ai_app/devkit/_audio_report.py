# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured source delivery, turn coverage and speech performance."""

from __future__ import annotations

import statistics


def _distribution(values):
    return (
        {"n": len(values), "median": statistics.median(values), "worst": max(values)}
        if values
        else {"n": 0, "median": "unmeasured", "worst": "unmeasured"}
    )


def _performance(plan, source, observed, speech, started, ended, completed):
    finish = next((row for row in source if row["kind"] == "measurement_finished"), None)
    limit = finish["monotonic"] if finish else ended
    events = [(t, event) for t, event in observed if started is not None and started <= t <= limit]
    diagnostics = [row for row in speech if started is not None and started <= row["monotonic"] <= limit]
    inputs = [row for row in source if row["kind"] == "input" and row["action"] == "audio"]
    expected = [event for event in plan["events"] if event["action"] == "audio" and event["handler"]]
    expected_replies = [event for event in expected if event.get("reply", True)]
    turns, unattributed = {}, 0
    for t, event in events:
        if event.kind != "turn" or event.data.get("phase") != "start":
            continue
        pts = event.data.get("timestamp_us")
        candidates = [
            row
            for row in inputs
            if isinstance(pts, int) and row["audio_start_us"] - 160000 <= pts < row["audio_end_us"]
        ]
        if len(candidates) == 1:
            turns[event.turn] = (t, candidates[0])
        else:
            unattributed += 1
    successful = {
        event.turn
        for _, event in events
        if event.kind == "turn" and event.data.get("phase") == "end" and event.data.get("outcome") == "done"
    }
    per_input = []
    for row in inputs:
        linked = [(turn, t) for turn, (t, item) in turns.items() if item["index"] == row["index"]]
        # The marker precedes sending the tail of the last 20 ms audio chunk.
        end = next(
            (
                item["monotonic"]
                for item in source
                if item["kind"] == "speech_end" and item["input_index"] == row["index"]
            ),
            None,
        )
        streams = {}
        for t, event in events:
            if event.kind == "response" and event.data.get("request_id") in {turn for turn, _ in linked}:
                key = (event.data.get("source"), event.data["request_id"], event.data.get("response_id"))
                streams[key] = t if event.data.get("final") else None
        finals = list(streams.values())
        completion = (
            max(finals)
            if finals and all(value is not None for value in finals) and all(turn in successful for turn, _ in linked)
            else None
        )
        audio = [
            item["monotonic"]
            for item in source
            if item["kind"] == "audio_out" and end is not None and item["monotonic"] >= end and item["rms"] > 0.0001
        ]
        next_input = next((item["monotonic"] for item in inputs if item["index"] > row["index"]), limit)
        per_input.append(
            {
                "index": row["index"],
                "text": row["text"],
                "handler": row["handler"],
                "reply": row.get("reply", row["handler"]),
                "turn_ids": [turn for turn, _ in linked],
                "query_delivery_s": max(t for _, t in linked) - end if linked and end is not None else None,
                "completed_reply_text_s": completion - end if completion is not None and end is not None else None,
                "first_observed_audio_s": audio[0] - end if audio and audio[0] < next_input else None,
            }
        )
    trace_errors = sum(
        event.kind == "error"
        or (bool(event.data.get("error")) and event.data["error"] != "CancelledError")
        or (
            event.kind == "turn"
            and event.data.get("phase") == "end"
            and event.data.get("outcome") not in {"done", "CancelledError"}
        )
        for _, event in events
    )
    errors = trace_errors + sum(
        row["kind"] == "error" and row.get("exception_type") != "CancelledError" for row in diagnostics
    )
    received = sum(any(row["index"] == item["index"] and row["turn_ids"] for row in per_input) for item in expected)
    late = finish["max_sender_lateness_s"] if finish else None
    measured_stt = any(row["kind"] in {"queue", "reset", "drop", "speaker"} for row in diagnostics)
    queues = [row["seconds"] for row in diagnostics if row["kind"] == "queue"]
    replies = sum(row["reply"] and row["completed_reply_text_s"] is not None for row in per_input)
    unexpected = sum(len(row["turn_ids"]) for row in per_input if not row["handler"])
    disrupted = any(row["kind"] in {"drop", "reset"} for row in diagnostics)
    source_completed = completed and finish is not None
    return {
        "execution_status": "completed" if source_completed else "failed",
        "handler_coverage": {"received": received, "expected": len(expected)},
        "reply_coverage": {"completed": replies, "expected": len(expected_replies)},
        "unattributed_turns": unattributed,
        "unexpected_control_turns": unexpected,
        "error_events": errors,
        "timing_comparable": source_completed
        and errors == 0
        and late <= 0.1
        and received == len(expected)
        and unattributed == 0
        and unexpected == 0
        and replies == len(expected_replies)
        and not disrupted,
        "max_sender_lateness_s": late if late is not None else "unmeasured",
        "stt": {
            "max_queue_s": max(queues) if queues else "unmeasured",
            "drops": sum(row["kind"] == "drop" for row in diagnostics) if measured_stt else "unmeasured",
            "resets": sum(row["kind"] == "reset" for row in diagnostics) if measured_stt else "unmeasured",
        },
        **{
            name: _distribution([row[name] for row in per_input if row["handler"] and row[name] is not None])
            for name in ("query_delivery_s", "completed_reply_text_s", "first_observed_audio_s")
        },
        "inputs": per_input,
    }
