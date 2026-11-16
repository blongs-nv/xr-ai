# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read marker-labelled instruments: a deterministic detector, then a model.

    frame   = p.frames.current()          a fresh frame, or one picture
    markers = DETECTOR.detect(frame.image)  QR and ArUco (OpenCV), no model
    names   = device_map[marker]          yaml/device_map.yaml
    marked  = annotate(frame, markers)    a colored X on each marker
    reply   = vlm.ask(marked, query)      one call: "magenta? cyan? ..."
    reading = parse_readings(reply)       each color's display -> instrument

The detector is the foundation's plain `MarkerDetector` (a function of one
image: no hub, no frame source). The colors, the prompt, the query and the
checks on the reply are upstream's
(`lab_instrument_monitoring_worker/instruments.py`). The reply is parsed
rather than asked for as structured output: its checks (a duplicated key,
two colors on one display, a color word as a reading) are not a schema.
"""

import asyncio
import io
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image, ImageChops, ImageDraw
from pydantic import BaseModel, ConfigDict, Field
from xr_ai_next import Frame, Participant, Role, clock
from xr_ai_tools.marker_tracking import (
    MarkerDetector,
    MarkerType,
    TrackedMarker,
)

RGB = tuple[int, int, int]
COLORS: tuple[tuple[str, RGB], ...] = (
    ("magenta", (255, 0, 255)),
    ("cyan", (0, 255, 255)),
    ("red", (255, 0, 0)),
    ("green", (0, 128, 0)),
    ("blue", (0, 0, 255)),
    ("yellow", (255, 255, 0)),
)
DeviceMap = dict[tuple[MarkerType, str], str]

NO_FRAME = "No camera frame available — please try again."
NOT_SCANNED = "The camera frame could not be scanned."
NO_MARKERS = "No readable marker-labelled lab instruments were found."
UNMAPPED = "No configured marker-labelled lab instruments were found."
INVALID = "Markers were found, but the instrument display response was invalid."
NO_MODEL = "The instrument vision model was unavailable."
UNREADABLE = (
    "Markers were found, but their instrument displays could not be read."
)


class Sighting(BaseModel):
    """A configured instrument's marker seen in a frame (upstream's fields)."""

    model_config = ConfigDict(extra="forbid")
    timestamp_us: int = Field(ge=0)
    marker_type: MarkerType
    marker_id: str = Field(min_length=1)
    device_name: str = Field(min_length=1)


class Reading(Sighting):
    """A configured instrument and what its display showed."""

    meter_reading: str = Field(min_length=1)


@dataclass
class ReadResult:
    readings: list[Reading] = field(default_factory=list)
    sightings: list[Sighting] = field(default_factory=list)
    available: bool = True
    message: str = ""

    def spoken(self) -> str:
        """What to say: each reading, or why there is none."""
        if not self.readings:
            return self.message or "No lab instrument readings were available."
        return "; ".join(
            f"{r.device_name}: {r.meter_reading}" for r in self.readings
        )


def load_device_map(path: Path) -> DeviceMap:
    """Marker family and id -> instrument name (yaml/device_map.yaml)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names: DeviceMap = {}
    for family, devices in (raw.get("devices") or {}).items():
        for marker_id, name in devices.items():
            if not str(marker_id).strip() or not str(name).strip():
                raise ValueError(f"{path}: empty marker id or name")
            key = (MarkerType(str(family)), str(marker_id).strip())
            names[key] = str(name).strip()
    return names


class InstrumentReader:
    """Reads every configured instrument in a participant's current view."""

    def __init__(
        self,
        vlm: Role,
        devices: DeviceMap,
        *,
        prompt: str,
        max_age_s: float,
        timeout_s: float,
        keep_scans: bool = False,
    ) -> None:
        self._vlm = vlm
        self._devices = devices
        self._prompt = prompt
        self._max_age_s = max_age_s
        self._timeout_s = timeout_s
        self._keep_scans = keep_scans

    async def read(self, p: Participant) -> ReadResult:
        frame = await p.frames.current(
            max_age_s=self._max_age_s, timeout_s=self._timeout_s
        )
        if frame is None:
            return ReadResult(available=False, message=NO_FRAME)
        if self._keep_scans:  # upstream's capture_marker_scans
            name = f"{clock.wall_us()}-frame-{frame.timestamp_us}"
            name += f"-seq-{frame.sequence}.{frame.mime_type.split('/')[-1]}"
            await p.artifacts.write_blob(f"marker-scans/{name}", frame.image)
        markers = await scan_markers(frame.image)
        if markers is None:
            return ReadResult(available=False, message=NOT_SCANNED)
        if not markers:
            return ReadResult(message=NO_MARKERS)
        known = [
            m for m in markers if (m.marker_type, m.value) in self._devices
        ]
        sightings = [
            Sighting(
                timestamp_us=frame.timestamp_us,
                marker_type=m.marker_type,
                marker_id=m.value,
                device_name=self._devices[m.marker_type, m.value],
            )
            for m in known
        ]
        if not known:
            return ReadResult(available=False, message=UNMAPPED)

        colored = assign_colors(known)
        colors = [color for color, _, _ in colored]
        try:
            with clock.held():  # drawing runs on a thread too
                marked = await asyncio.to_thread(annotate, frame, colored)
            reply = await self._vlm.ask(
                marked, reading_query(colors), system=self._prompt
            )
        except Exception:  # noqa: BLE001 - say so, as upstream
            return ReadResult(
                sightings=sightings, available=False, message=NO_MODEL
            )
        parsed = parse_readings(str(reply), colors)
        if parsed is None:
            return ReadResult(
                sightings=sightings, available=False, message=INVALID
            )
        readings = [
            Reading(
                timestamp_us=frame.timestamp_us,
                marker_type=m.marker_type,
                marker_id=m.value,
                device_name=self._devices[m.marker_type, m.value],
                meter_reading=parsed[color],
            )
            for color, _, m in colored
            if parsed[color].upper() != "UNKNOWN"
        ]
        if not readings:
            return ReadResult(
                sightings=sightings, available=False, message=UNREADABLE
            )
        return ReadResult(readings=readings, sightings=sightings)


# -- the detector ------------------------------------------------------------

DETECTOR = MarkerDetector()  # QR and ArUco (DICT_4X4_50), as upstream's tool
log = logging.getLogger(__name__)


async def scan_markers(image: bytes) -> list[TrackedMarker] | None:
    """Every QR and ArUco marker in `image`; None if it cannot be scanned."""
    try:
        with clock.held():  # OpenCV on a thread: a virtual clock waits
            return await asyncio.to_thread(DETECTOR.detect, image)
    except Exception:  # noqa: BLE001 - a frame that does not decode, as upstream
        log.exception("marker extraction failed")
        return None


# -- marking the frame and asking -------------------------------------------

ColoredMarker = tuple[str, RGB, TrackedMarker]


def assign_colors(markers: list[TrackedMarker]) -> list[ColoredMarker]:
    """A color per marker, top to bottom then left to right; six at most."""

    def position(marker: TrackedMarker) -> tuple[float, float]:
        corners = marker.corners
        return (
            sum(c.y for c in corners) / len(corners),
            sum(c.x for c in corners) / len(corners),
        )

    ordered = sorted(markers, key=position)
    return [
        (name, rgb, marker)
        for (name, rgb), marker in zip(COLORS, ordered, strict=False)
    ]


def annotate(frame: Frame, colored: list[ColoredMarker]) -> Frame:
    """A copy of `frame` (PNG) with each marker crossed out in its color."""
    with Image.open(io.BytesIO(frame.image)) as opened:
        image = opened.convert("RGB")
    for _name, rgb, marker in colored:
        points = [(c.x, c.y) for c in marker.corners]
        left, right = min(x for x, _ in points), max(x for x, _ in points)
        top, bottom = min(y for _, y in points), max(y for _, y in points)
        width = max(3, round(min(right - left, bottom - top) * 0.24))
        inside = Image.new("L", image.size, 0)
        ImageDraw.Draw(inside).polygon(points, fill=255)
        cross = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(cross)
        draw.line((left, top, right, bottom), fill=255, width=width)
        draw.line((right, top, left, bottom), fill=255, width=width)
        image.paste(rgb, mask=ImageChops.multiply(inside, cross))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return Frame(
        image=out.getvalue(),
        timestamp_us=frame.timestamp_us,
        width=image.width,
        height=image.height,
        mime_type="image/png",
        sequence=frame.sequence,
    )


def reading_query(colors: list[str]) -> str:
    """Upstream's question for the colored X's (`_reading_query`)."""
    template = json.dumps(
        {c: {"reading": "UNKNOWN", "display_bbox": None} for c in colors}
    )
    return (
        f"The requested colored-X keys are: {json.dumps(colors)}. "
        "Apply this association rule to every key: a reading belongs to a "
        "color only when that colored X and the display are on the same "
        "visually bounded instrument. For each color, directly return the "
        "associated display reading and its normalized [left, top, right, "
        "bottom] bounding box using coordinates from 0 through 1000. Use "
        "UNKNOWN and null when that color has no unambiguous readable "
        "display. Never use the same physical display for more than one "
        "color; separate displays may show the same reading. Return exactly "
        f"this JSON shape, replacing only its values: {template}. No "
        "explanation or Markdown."
    )


# -- checking the reply (upstream's _parse_joint_readings) -------------------

Box = tuple[float, float, float, float]


def parse_readings(text: str, colors: list[str]) -> dict[str, str] | None:
    """Each color's reading ("UNKNOWN" when unclear); None if not JSON."""
    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = visible.find("{"), visible.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        pairs = json.loads(visible[start : end + 1], object_pairs_hook=list)
    except json.JSONDecodeError:
        return None
    payload = _unique(pairs)
    keys = {color: color.strip().lower() for color in colors}
    if payload is None or len(set(keys.values())) != len(colors):
        return None
    values, repeated = payload
    readings = dict.fromkeys(colors, "UNKNOWN")
    boxes: dict[str, Box] = {}
    for color, key in keys.items():
        entry = None if key in repeated else _unique(values.get(key))
        if entry is None or entry[1] & {"reading", "display_bbox"}:
            continue
        reading = _valid_reading(entry[0].get("reading"))
        box = _valid_box(entry[0].get("display_bbox"))
        if reading is not None and box is not None:
            readings[color], boxes[color] = reading, box
    claimed = list(boxes.items())  # one display claimed by two colors
    for i, (first, first_box) in enumerate(claimed):
        for second, second_box in claimed[i + 1 :]:
            if _same_display(first_box, second_box):
                readings[first] = readings[second] = "UNKNOWN"
    return readings


def _unique(value: object) -> tuple[dict[str, object], set[str]] | None:
    """A JSON object read as pairs: its keys (lowercase) and the repeats."""
    if not isinstance(value, list) or not all(
        isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], str)
        for pair in value
    ):
        return None
    payload: dict[str, object] = {}
    repeated: set[str] = set()
    for key, item in value:
        key = key.strip().lower()
        if key in payload:
            repeated.add(key)
        payload[key] = item
    return payload, repeated


def _valid_reading(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    reading = value.strip()
    names = "|".join(name for name, _ in COLORS)
    if (
        re.search(r"\bUNKNOWN\b", reading, flags=re.IGNORECASE)
        or re.search(r"#[0-9A-Fa-f]{6}\b", reading)
        or re.search(rf"\b({names})\b", reading, flags=re.IGNORECASE)
    ):
        return None
    return reading


def _valid_box(value: object) -> Box | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) for v in value
    ):
        return None
    left, top, right, bottom = (float(v) for v in value)
    if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
        return None
    return left, top, right, bottom


def _same_display(first: Box, second: Box) -> bool:
    """The boxes overlap by at least 80% of the smaller one."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    if left >= right or top >= bottom:
        return False
    overlap = (right - left) * (bottom - top)
    smaller = min(
        (first[2] - first[0]) * (first[3] - first[1]),
        (second[2] - second[0]) * (second[3] - second[1]),
    )
    return overlap / smaller >= 0.8
