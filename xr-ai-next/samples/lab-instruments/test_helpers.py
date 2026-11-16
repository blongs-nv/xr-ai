# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The helpers that hold logic: marker scan and colors on the committed
pictures, the marked frame, the checks on the VLM's replies, and how the
instrument monitor compares readings.

Run from this directory: uv --config-file ../../../uv.toml run pytest
"""

import asyncio
import io
import json
from decimal import Decimal
from pathlib import Path

import pytest
from instrument_monitor import lost, normalize_meter_reading, observe
from instruments import (
    Reading,
    Sighting,
    annotate,
    assign_colors,
    load_device_map,
    parse_readings,
    scan_markers,
)
from PIL import Image
from visual_monitor import parse_monitor_response
from xr_ai_next import Frame

HERE = Path(__file__).resolve().parent
MEDIA = HERE / "scenarios" / "media"
DEVICES = load_device_map(HERE / "yaml" / "device_map.yaml")


def scan(name: str):
    return asyncio.run(scan_markers((MEDIA / name).read_bytes()))


# -- the detector on the committed pictures ----------------------------------


@pytest.mark.parametrize(
    ("name", "found"),
    [
        ("device1_23_5.png", [("qr_code", "device-1")]),
        ("device1_24_1.png", [("qr_code", "device-1")]),
        ("unreadable.png", [("qr_code", "device-1")]),
        ("two_devices.png", [("qr_code", "device-1"), ("aruco", "1")]),
        ("unknown_marker.png", [("aruco", "7")]),
        ("no_marker.png", []),
    ],
)
def test_the_scenario_pictures_have_the_markers_they_claim(
    name: str, found: list
) -> None:
    markers = scan(name)
    assert sorted((m.marker_type.value, m.value) for m in markers) == sorted(
        found
    )


def test_a_frame_that_does_not_decode_is_not_scanned() -> None:
    assert asyncio.run(scan_markers(b"not an image")) is None


def test_colors_go_top_to_bottom_and_device1_is_magenta() -> None:
    colored = assign_colors(scan("two_devices.png"))
    names = [
        (color, DEVICES[m.marker_type, m.value]) for color, _, m in colored
    ]
    assert names == [("magenta", "Device1"), ("cyan", "Device2")]
    assert ("aruco", "7") not in {(t.value, i) for t, i in DEVICES}


def test_annotate_draws_each_x_inside_its_marker_only() -> None:
    source = (MEDIA / "two_devices.png").read_bytes()
    frame = Frame(source, 1, 1280, 720, "image/png", sequence=3)
    marked = annotate(frame, assign_colors(scan("two_devices.png")))
    assert (marked.mime_type, marked.sequence) == ("image/png", 3)
    image = Image.open(io.BytesIO(marked.image)).convert("RGB")
    assert image.getpixel((340, 460)) == (255, 0, 255)  # Device1's QR
    assert image.getpixel((940, 510)) == (0, 255, 255)  # Device2's ArUco
    assert image.getpixel((340, 245)) != (255, 0, 255)  # its display


# -- the VLM's reply ---------------------------------------------------------


def reply(**colors: object) -> str:
    return json.dumps(colors)


def entry(reading: object, box: object = (100, 100, 400, 300)) -> dict:
    return {"reading": reading, "display_bbox": list(box) if box else box}


def test_readings_are_read_per_color_case_and_reasoning_ignored() -> None:
    text = "<think>two displays</think>" + json.dumps(
        {
            "Magenta": entry("23.5 C"),
            "cyan": entry("7.02 pH", (500, 0, 900, 300)),
        }
    )
    assert parse_readings(text, ["magenta", "cyan"]) == {
        "magenta": "23.5 C",
        "cyan": "7.02 pH",
    }


@pytest.mark.parametrize(
    "text",
    [
        "no JSON at all",
        '{"magenta": ',
        '["magenta"]',
    ],
)
def test_a_reply_that_is_not_a_json_object_is_invalid(text: str) -> None:
    assert parse_readings(text, ["magenta"]) is None


@pytest.mark.parametrize(
    ("value", "why"),
    [
        (entry("UNKNOWN", None), "the model could not read it"),
        (entry("23.5 C", None), "no display box"),
        (entry("23.5 C", (400, 100, 100, 300)), "an empty box"),
        (entry("cyan"), "a color word, not a reading"),
        (entry("#FF00FF"), "a color code"),
        (entry(23.5), "not text"),
    ],
)
def test_an_unusable_entry_is_unknown(value: dict, why: str) -> None:
    assert parse_readings(reply(magenta=value), ["magenta"]) == {
        "magenta": "UNKNOWN"
    }, why


def test_a_repeated_color_or_field_is_unknown() -> None:
    repeated_color = (
        '{"magenta": {"reading": "1 V", "display_bbox": [0, 0, 10, 10]},'
        ' "magenta": {"reading": "2 V", "display_bbox": [0, 0, 10, 10]}}'
    )
    repeated_field = (
        '{"magenta": {"reading": "1 V", "reading": "2 V",'
        ' "display_bbox": [0, 0, 10, 10]}}'
    )
    for text in (repeated_color, repeated_field):
        assert parse_readings(text, ["magenta"]) == {"magenta": "UNKNOWN"}


def test_one_display_claimed_by_two_colors_counts_for_neither() -> None:
    text = reply(
        magenta=entry("12.0 V", (100, 100, 400, 300)),
        cyan=entry("12.0 V", (110, 105, 395, 300)),
        red=entry("99.0 A", (600, 100, 900, 300)),  # same value, own display
    )
    assert parse_readings(text, ["magenta", "cyan", "red"]) == {
        "magenta": "UNKNOWN",
        "cyan": "UNKNOWN",
        "red": "99.0 A",
    }


def test_monitor_replies_baseline_and_change() -> None:
    changed = '```json\n{"caption": "Door open.", "changed": true, '
    changed += '"summary": "The door opened."}\n```'
    assert parse_monitor_response(changed, baseline=False).changed
    baseline = parse_monitor_response(changed, baseline=True)
    assert (baseline.changed, baseline.summary) == (False, "")
    quiet = '{"caption": "Door shut.", "changed": false, "summary": "x"}'
    assert parse_monitor_response(quiet, baseline=False).summary == ""
    with pytest.raises(ValueError):
        parse_monitor_response(
            '{"caption": "x", "changed": true}', baseline=False
        )
    with pytest.raises(ValueError):
        parse_monitor_response("nothing", baseline=False)


# -- comparing readings ------------------------------------------------------


@pytest.mark.parametrize(
    ("reading", "previous", "expected"),
    [
        ("23.50 C", "", (Decimal("23.50"), "C", "23.5 C")),
        ("12 volts", "", (Decimal("12"), "V", "12 V")),
        ("1,5 V", "", (Decimal("1.5"), "V", "1.5 V")),
        ("12.0 Volts", "", (Decimal("12.0"), "V", "12 V")),
        ("7.02", "pH", (Decimal("7.02"), "pH", "7.02 pH")),
        ("37 °C", "", (Decimal("37"), "°C", "37 °C")),
        ("ERR", "V", None),
    ],
)
def test_normalize_meter_reading(
    reading: str, previous: str, expected: tuple | None
) -> None:
    assert normalize_meter_reading(reading, previous_unit=previous) == expected


def at(reading: str | None, t: int = 1) -> tuple[list, list]:
    """One read of Device1: its reading (None: marker seen, unreadable)."""
    seen = Sighting(
        timestamp_us=t,
        marker_type="qr_code",
        marker_id="device-1",
        device_name="Device1",
    )
    if reading is None:
        return [], [seen]
    return [Reading(**seen.model_dump(), meter_reading=reading)], [seen]


def test_observe_reports_discovery_and_real_changes_only() -> None:
    tracked: dict = {}
    first = observe(tracked, *at("23.5 C"), now=0.0)
    assert [c.change_type for c in first] == ["discovered"]
    assert observe(tracked, *at("23.50 C"), now=5.0) == []  # same number
    (change,) = observe(tracked, *at("24.1 C"), now=10.0)
    assert (change.previous_reading, change.meter_reading) == (
        "23.5 C",
        "24.1 C",
    )


def test_lost_once_and_a_sighting_keeps_an_instrument_tracked() -> None:
    tracked: dict = {}
    observe(tracked, *at("23.5 C"), now=0.0)
    observe(tracked, *at(None), now=20.0)  # display unreadable, marker seen
    assert lost(tracked, 45.0, after_s=30.0) == []
    (gone,) = lost(tracked, 50.0, after_s=30.0)
    assert (gone.device_name, gone.meter_reading) == ("Device1", "23.5 C")
    assert lost(tracked, 80.0, after_s=30.0) == []  # said once
    observe(tracked, *at("23.5 C"), now=90.0)  # back: tracked again
    assert lost(tracked, 125.0, after_s=30.0) != []
