# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic camera scenes for this sample's speech replay."""

from pathlib import Path

from PIL import Image, ImageDraw


def prepare(directory, *, images):
    for name, scene in images.items():
        path = Path(directory) / name
        if path.exists():
            continue
        image = Image.new("RGB", (960, 540), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((160, 100, 800, 440), fill=scene["rectangle"])
        image.save(path)
