# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the tea-making sample's hub and app processes.

The app runs the tea guide (a procedure that watches the camera), a
conversation and Clef routing. Its separately managed local model stack lives
in ``model-servers/``. Optional session recording runs as a separate capture
process owned by DeviceIOHub.

Run from ``app-samples/tea-making/`` with ``uv run tea_making``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

_BASE = Path(__file__).resolve().parent

_CAPTURE_PROCESS = Process(
    "capture",
    "../../services/device-io-hub",
    "device_io_capture",
    config="yaml/media_capture.yaml",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tea-making guidance that watches the camera, with "
        "a conversation.",
    )
    parser.add_argument(
        "--session-recording",
        action="store_true",
        help=(
            "enable session recording of participant video, bidirectional "
            "audio, and data-channel traffic"
        ),
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _parser().parse_args(sys.argv[1:] if argv is None else argv)


def _build_processes(*, capture: bool = False) -> list[Process]:
    """Build the hub, app, and explicitly requested capture process."""

    processes = [
        Process(
            "hub",
            "../../services/device-io-hub",
            "device_io_hub",
            config="yaml/device_io_hub.yaml",
        ),
        Process("app", ".", "tea_making_app", config="app.yaml"),
    ]
    if capture:
        processes.insert(1, _CAPTURE_PROCESS)
    return processes


PROCESSES = _build_processes()


def run(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    setup_logging("orchestrator", namespace="tea-making")
    run_stack(_build_processes(capture=args.session_recording), _BASE)


if __name__ == "__main__":
    run()
