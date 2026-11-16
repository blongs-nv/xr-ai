# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the scenarios' synthetic pictures to media/ (committed; rerun to
regenerate). Scenes a vision model answers reliably: a shop sign with one
large word, before (CLOSED, red) and after (OPEN, green).

    uv --config-file ../../../uv.toml run python scenarios/make_images.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MEDIA = Path(__file__).resolve().parent / "media"
SIZE = (640, 480)


def sign(word: str, color: tuple[int, int, int]) -> Image.Image:
    """A sign that says `word` in large white letters."""
    image = Image.new("RGB", SIZE, (235, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 120, 580, 360), fill=color, outline="black")
    font = ImageFont.load_default(size=120)
    draw.text((320, 240), word, fill="white", font=font, anchor="mm")
    return image


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    sign("CLOSED", (190, 20, 20)).save(MEDIA / "sign_closed.png")
    sign("OPEN", (0, 140, 60)).save(MEDIA / "sign_open.png")
