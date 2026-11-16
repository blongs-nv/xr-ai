# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract still frames from a raw capture bundle into ``fixtures/recorded/``.

A raw bundle (``xr_ai_tools.capture.CaptureTools``, see docs/source/components/server-runtime.md) holds
``manifest.json``, ``video/session.264`` (H.264 Annex B) and ``video/frames.jsonl`` (one line per encoded
frame: ``frame_id``, ``pts_us``, ``relative_us``, ``track_id``, ...). This script decodes the stream with
``ffmpeg`` and writes the selected frames as JPEG, each with a ``<name>.json`` provenance file
(bundle path, frame_id, pts_us, relative_us, track_id). It refuses to continue when the decoded frame
count differs from the index, because the stream is joined from several segments and a mismatch would
mislabel frames.

    python extract_frames.py BUNDLE --at 12.5 --name kettle_at_target [--at 40 --name vessel_empty]
    python extract_frames.py BUNDLE --frame-id 310 --name packet_chamomile --track cam0

``--at`` is seconds from session start (the nearest frame is taken). Names must match the fixture file
names in ``cases/recorded.yaml`` (``fixtures/recorded/<name>.jpg``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


class ExtractError(RuntimeError):
    """The bundle or the environment cannot support extraction."""


def read_index(bundle: Path) -> list[dict]:
    """Frame index lines in file order."""
    index = bundle / "video" / "frames.jsonl"
    if not index.is_file():
        raise ExtractError(f"{index} not found; is {bundle} a raw capture bundle?")
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ExtractError(f"{index} is empty")
    return rows


def select(
    rows: list[dict], *, at_s: float | None = None, frame_id: int | None = None, track: str | None = None
) -> int:
    """Position in ``rows`` of the requested frame (by id, or nearest to ``at_s`` seconds)."""
    candidates = [(i, r) for i, r in enumerate(rows) if track is None or r.get("track_id") == track]
    if not candidates:
        raise ExtractError(f"no frames for track {track!r}")
    if frame_id is not None:
        for i, row in candidates:
            if row["frame_id"] == frame_id:
                return i
        raise ExtractError(f"frame_id {frame_id} not in the index")
    if at_s is None:
        raise ExtractError("give --at or --frame-id")
    target = at_s * 1_000_000
    return min(candidates, key=lambda c: abs(c[1]["relative_us"] - target))[0]


def ffmpeg_command(ffmpeg: str, stream: Path, position: int, out: Path) -> list[str]:
    """Command that writes decoded frame number ``position`` (0-based, decode order) as one JPEG."""
    return [
        ffmpeg, "-v", "error", "-y", "-f", "h264", "-i", str(stream),
        "-vf", f"select=eq(n\\,{position})", "-vsync", "0", "-frames:v", "1", "-q:v", "2", str(out),
    ]  # fmt: skip


def count_frames(ffmpeg: str, stream: Path) -> int:
    """Decoded frame count of the stream (``ffmpeg -f null``; progress counter from ``-stats``)."""
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-f", "h264", "-i", str(stream), "-map", "0:v:0"]
        + ["-f", "null", "-", "-progress", "pipe:1"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ExtractError(f"ffmpeg could not decode {stream}: {proc.stderr.strip()[:200]}")
    frames = [int(line.split("=", 1)[1]) for line in proc.stdout.splitlines() if line.startswith("frame=")]
    if not frames:
        raise ExtractError("ffmpeg reported no frame count")
    return frames[-1]


def extract(bundle: Path, picks: list[tuple[str, dict]], out_dir: Path, ffmpeg: str) -> list[Path]:
    """Write each pick (``name``, selector kwargs) as ``<out_dir>/<name>.jpg`` plus ``<name>.json``."""
    rows = read_index(bundle)
    stream = bundle / "video" / "session.264"
    if not stream.is_file():
        raise ExtractError(f"{stream} not found")
    decoded = count_frames(ffmpeg, stream)
    if decoded != len(rows):
        raise ExtractError(
            f"decoded {decoded} frames but frames.jsonl lists {len(rows)}; refusing to label frames "
            "(multi-track or segment gaps). Extract from a single-track bundle or fix the index."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, selector in picks:
        position = select(rows, **selector)
        target = out_dir / f"{name}.jpg"
        subprocess.run(ffmpeg_command(ffmpeg, stream, position, target), check=True)
        if not target.is_file():
            raise ExtractError(f"ffmpeg wrote nothing for {name}")
        row = rows[position]
        provenance = {"bundle": str(bundle), "decode_position": position, **row}
        target.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--at", type=float, action="append", default=[], help="seconds from session start")
    parser.add_argument("--frame-id", type=int, action="append", default=[])
    parser.add_argument("--name", action="append", required=True, help="fixture name; one per --at/--frame-id")
    parser.add_argument("--track", help="track_id to choose frames from")
    parser.add_argument("--out", type=Path, default=_HERE / "fixtures" / "recorded")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args(argv)
    selectors = [{"at_s": a, "track": args.track} for a in args.at] + [
        {"frame_id": f, "track": args.track} for f in args.frame_id
    ]
    if len(selectors) != len(args.name):
        print("error: give exactly one --name per --at/--frame-id", file=sys.stderr)
        return 2
    binary = shutil.which(args.ffmpeg)
    if binary is None:
        print(f"error: {args.ffmpeg!r} not found on PATH; install ffmpeg to extract frames", file=sys.stderr)
        return 2
    try:
        for path in extract(args.bundle, list(zip(args.name, selectors, strict=True)), args.out, binary):
            print(path)
    except (ExtractError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
