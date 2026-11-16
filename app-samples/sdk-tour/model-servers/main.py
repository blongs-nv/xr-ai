# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the local model servers required by the SDK tour samples.

They are exactly upstream's conversation-example stack: the same services, GPU
profiles and stack name, so either launcher reuses or stops the same servers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from xr_ai_model_server_stack import run_model_stack

_BASE = (
    Path(__file__).resolve().parents[3]
    / "agent-samples" / "conversation-example" / "model-servers"
)
_SERVICES = ("speaker-stt", "tts", "omni", "vlm", "clef")


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop this sample's configured model servers and exit.",
    )
    parser.add_argument(
        "--gpu-profile",
        metavar="NAME",
        help="Select a startup hardware profile; with --stop, apply its port overrides without GPU detection.",
    )
    args = parser.parse_args()
    try:
        run_model_stack(_BASE, _SERVICES, stop=args.stop, gpu_profile=args.gpu_profile)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    run()
