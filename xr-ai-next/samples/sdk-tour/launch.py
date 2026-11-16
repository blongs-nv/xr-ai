# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one of the SDK tour samples on DeviceIOHub: the hub, then the app.

  uv run sdk-tour 01-hello-voice      (or just: uv run sdk-tour 01)

The same stack as `../simple-vlm/main.py`, for every sample in this folder:
they share one model stack (`models.json`, started by `model-servers/`) and
one set of hub, voice-gate and session-recording files (`yaml/`, upstream
conversation-example's). The app process is the sample's own uv project,
run as its `<name>_app` script with its `app.yaml` (`01-hello-voice` ->
`hello_voice_app`). Model servers are never started here.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from xr_ai_launcher import Process, run_stack
from xr_ai_logging import setup_logging

HERE = Path(__file__).resolve().parent
SAMPLES = sorted(
    p.name
    for p in HERE.iterdir()
    if p.name[:2].isdigit() and (p / "app.yaml").is_file()
)

# What a sample needs started before its app, beyond the hub.
EXTRA = {
    # 07's stand-in A2A agent, on :8830 where its app expects it.
    "07-delegate-a2a": [
        Process(
            "research_agent", "07-delegate-a2a/research_agent", "research_agent"
        )
    ],
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdk-tour",
        description="Run an SDK tour sample: DeviceIOHub and its app.",
    )
    parser.add_argument(
        "sample", help=f"the sample's folder or number: {', '.join(SAMPLES)}"
    )
    parser.add_argument(
        "--session-recording",
        action="store_true",
        help=(
            "enable session recording of participant video, bidirectional "
            "audio, and data-channel traffic"
        ),
    )
    parser.add_argument(
        "--hub-config",
        type=Path,
        metavar="PATH",
        help="use hub YAML instead of yaml/device_io_hub.yaml "
        "(relative to launch directory)",
    )
    parser.add_argument(
        "--health-port",
        type=int,
        metavar="PORT",
        help="serve stack process health on loopback PORT "
        "(1–65535; disabled by default)",
    )
    return parser


def sample_dir(name: str) -> str | None:
    """The folder `name` means: the folder itself, or its number ("01")."""
    name = name.rstrip("/")
    matches = [s for s in SAMPLES if name in (s, s[:2])]
    return matches[0] if len(matches) == 1 else None


def processes(
    sample: str, *, capture: bool = False, hub_config: Path | None = None
) -> list[Process]:
    """The hub, optional recording, the sample's extras, then its app."""
    hub_project = "../../../services/device-io-hub"
    stack = [
        Process(
            "hub",
            hub_project,
            "device_io_hub",
            config=hub_config or "yaml/device_io_hub.yaml",
        )
    ]
    if capture:
        stack.append(
            Process(
                "capture",
                hub_project,
                "device_io_capture",
                config="yaml/media_capture.yaml",
            )
        )
    app = sample[3:].replace("-", "_") + "_app"
    return [
        *stack,
        *EXTRA.get(sample, []),
        Process("app", sample, app, config=f"{sample}/app.yaml"),
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    sample = sample_dir(args.sample)
    if sample is None:
        parser.error(f"no sample {args.sample!r}; one of {', '.join(SAMPLES)}")
    hub_config = (
        args.hub_config.expanduser().resolve()
        if args.hub_config is not None
        else None
    )
    setup_logging("orchestrator", namespace=sample)
    run_stack(
        processes(
            sample, capture=args.session_recording, hub_config=hub_config
        ),
        HERE,
        health_port=args.health_port,
    )


if __name__ == "__main__":
    main()
