# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the scenarios' synthetic pictures to media/ (committed; rerun to
regenerate): a tea label, a kettle display, a mug with a tea bag. Scripted
runs only need some frame; with --real-models record your own (README).

    uv --config-file ../../uv.toml run python scenarios/make_images.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MEDIA = Path(__file__).resolve().parent / "media"
SIZE = (640, 480)


def card(lines: list[str], fill: tuple[int, int, int]) -> Image.Image:
    """A card with large centered lines of text."""
    image = Image.new("RGB", SIZE, (235, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 100, 580, 380), fill=fill, outline="black")
    font = ImageFont.load_default(size=72)
    step = 280 // (len(lines) + 1)
    for i, line in enumerate(lines, 1):
        xy = (320, 100 + i * step)
        draw.text(xy, line, fill="white", font=font, anchor="mm")
    return image


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    card(["GREEN TEA"], (40, 120, 60)).save(MEDIA / "tea_label.png")
    card(["176 °F"], (20, 20, 20)).save(MEDIA / "kettle_display.png")
    card(["MUG", "TEA BAG IN"], (120, 90, 40)).save(MEDIA / "mug.png")
