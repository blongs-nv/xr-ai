# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic camera scenes for this sample's speech replay."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def prepare(directory, *, images):
    for name, scene in images.items():
        path = Path(directory) / name
        if path.exists():
            continue
        image = Image.new("RGB", (960, 540), "white")
        draw = ImageDraw.Draw(image)
        if "rectangle" in scene:
            draw.rectangle((160, 100, 800, 440), fill=scene["rectangle"])
        else:
            draw.text((480, 270), scene["text"], fill="black", anchor="mm", font=ImageFont.load_default(size=70))
        image.save(path)
