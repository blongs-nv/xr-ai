# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the scenarios' synthetic pictures to media/ (committed; rerun to
regenerate). Scenes a vision model answers reliably: a sign with one large
word, and a red ball on a plain background.

    uv --config-file ../../../uv.toml run python scenarios/make_images.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MEDIA = Path(__file__).resolve().parent / "media"
SIZE = (640, 480)


def sign() -> Image.Image:
    """A green sign that says EXIT in large white letters."""
    image = Image.new("RGB", SIZE, (235, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 120, 560, 360), fill=(0, 140, 60), outline="black")
    font = ImageFont.load_default(size=150)
    draw.text((320, 240), "EXIT", fill="white", font=font, anchor="mm")
    return image


def red_ball() -> Image.Image:
    """A red ball on a plain white background."""
    image = Image.new("RGB", SIZE, "white")
    ImageDraw.Draw(image).ellipse((200, 120, 440, 360), fill=(220, 20, 20))
    return image


if __name__ == "__main__":
    MEDIA.mkdir(exist_ok=True)
    sign().save(MEDIA / "sign.png")
    red_ball().save(MEDIA / "red_ball.png")
