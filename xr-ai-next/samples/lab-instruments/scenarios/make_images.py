# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the scenarios' synthetic pictures to media/ (committed; rerun to
regenerate): bench instruments with a large display and one of upstream's
printable markers (agent-samples/lab-instrument-monitoring/sample-markers/)
on the housing.

    uv --config-file ../../../uv.toml run python scenarios/make_images.py
"""

from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

MEDIA = Path(__file__).resolve().parent / "media"
MARKERS = (
    Path(__file__).resolve().parents[4]
    / "agent-samples"
    / "lab-instrument-monitoring"
    / "sample-markers"
)
QR_DEVICE1 = MARKERS / "qr" / "Device1_QR_device-1.png"
ARUCO_DEVICE2 = MARKERS / "aruco" / "Device2_ArUco_1.png"
SIZE = (1280, 720)
MARKER_PX = 230


def unknown_aruco() -> Image.Image:
    """ArUco id 7 (DICT_4X4_50), which yaml/device_map.yaml does not name,
    with the same white border as upstream's printed markers."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = Image.fromarray(cv2.aruco.generateImageMarker(dictionary, 7, 896))
    framed = Image.new("L", (1024, 1024), 255)
    framed.paste(marker, (64, 64))
    return framed


def instrument(
    image: Image.Image,
    left: int,
    display: str | None,
    marker: Path | Image.Image | None,
    marker_top: int,
) -> None:
    """One instrument, 500 px wide: a dark housing, a display showing
    `display` (blank when None) and the marker on the housing."""
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (left, 120, left + 500, 640),
        24,
        fill=(70, 74, 80),
        outline="black",
        width=6,
    )
    draw.rectangle((left + 40, 160, left + 460, 330), fill=(205, 225, 200))
    if display is not None:
        font = ImageFont.load_default(size=110)
        draw.text(
            (left + 250, 245), display, fill="black", font=font, anchor="mm"
        )
    if marker is not None:
        if not isinstance(marker, Image.Image):
            marker = Image.open(marker)
        picture = marker.convert("RGB").resize(
            (MARKER_PX, MARKER_PX), Image.Resampling.NEAREST
        )
        image.paste(picture, (left + 135, marker_top))


def bench(*instruments: tuple) -> Image.Image:
    image = Image.new("RGB", SIZE, (225, 222, 215))
    for spec in instruments:
        instrument(image, *spec)
    return image


SCENES = {
    # Device1 (QR "device-1"): a thermometer; later its reading changes.
    "device1_23_5.png": bench((390, "23.5 C", QR_DEVICE1, 370)),
    "device1_24_1.png": bench((390, "24.1 C", QR_DEVICE1, 370)),
    # Device1 and Device2 (ArUco 1), side by side. Device1's marker sits
    # higher, so it is always the first color (magenta).
    "two_devices.png": bench(
        (90, "23.5 C", QR_DEVICE1, 345), (690, "7.02 pH", ARUCO_DEVICE2, 395)
    ),
    # An instrument without a marker.
    "no_marker.png": bench((390, "23.5 C", None, 370)),
    # A marker the device map does not name.
    "unknown_marker.png": bench((390, "12.0 V", unknown_aruco(), 370)),
    # Device1 with its display off: nothing to read.
    "unreadable.png": bench((390, None, QR_DEVICE1, 370)),
}


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    for name, image in SCENES.items():
        image.save(MEDIA / name)
