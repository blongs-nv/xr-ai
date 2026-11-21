# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic printer scenes with geometric bed and door state."""

import math
from pathlib import Path

from PIL import Image, ImageDraw


def render(*, occupied, door_open):
    camera = (4.0, -6.0, 4.0)

    def unit(v):
        length = math.sqrt(sum(x * x for x in v))
        return tuple(x / length for x in v)

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    forward = unit(tuple(b - a for a, b in zip(camera, (0, 0, 1.35), strict=True)))
    right = unit(cross(forward, (0, 0, 1)))
    up = cross(right, forward)
    faces = []

    def face(points, color):
        depth = sum(dot(tuple(x - c for x, c in zip(p, camera, strict=True)), forward) for p in points) / len(points)
        faces.append((depth, points, color))

    def box(lo, hi, color):
        x, y, z = lo
        X, Y, Z = hi
        vertices = [(x, y, z), (X, y, z), (X, Y, z), (x, Y, z), (x, y, Z), (X, y, Z), (X, Y, Z), (x, Y, Z)]
        for indices, shade in [
            ((0, 1, 2, 3), 0.5),
            ((4, 7, 6, 5), 1),
            ((0, 4, 5, 1), 0.85),
            ((1, 5, 6, 2), 0.65),
            ((2, 6, 7, 3), 0.6),
            ((3, 7, 4, 0), 0.75),
        ]:
            face([vertices[i] for i in indices], tuple(round(c * shade) for c in color) + (255,))

    box((-1.1, -0.95, 0), (1.1, 0.95, 0.25), (65, 70, 78))
    box((-0.95, 0.83, 0.25), (0.95, 0.9, 2.75), (155, 160, 166))
    for x in [-1, 0.9]:
        for y in [-0.85, 0.75]:
            box((x, y, 0.25), (x + 0.1, y + 0.1, 2.85), (45, 50, 58))
    box((-1.05, -0.9, 2.75), (1.05, 0.9, 2.9), (65, 70, 78))
    box((-0.83, -0.7, 0.65), (0.83, 0.7, 0.72), (205, 210, 215))
    box((-0.92, 0.3, 2.02), (0.92, 0.44, 2.14), (65, 70, 78))
    box((-0.15, 0.2, 1.72), (0.2, 0.52, 2.18), (35, 105, 170))
    box((-0.03, 0.28, 1.6), (0.06, 0.37, 1.73), (200, 145, 65))
    for x in [-0.8, 0.75]:
        box((x, -0.6, 0.25), (x + 0.05, 0.6, 0.65), (100, 105, 110))
    if occupied:
        box((-0.45, -0.4, 0.73), (0.3, 0.3, 1.42), (240, 115, 30))
        for i in range(12):
            z = 0.78 + i * 0.05
            face(
                [(-0.45, -0.401, z), (0.3, -0.401, z), (0.3, -0.401, z + 0.008), (-0.45, -0.401, z + 0.008)],
                (120, 65, 20, 150),
            )
    angle = math.radians(-105 if door_open else 0)

    def door(u, z):
        return (-1 + u * math.cos(angle), -0.91 + u * math.sin(angle), z)

    def panel(u0, z0, u1, z1, color):
        face([door(u0, z0), door(u1, z0), door(u1, z1), door(u0, z1)], color)

    panel(0, 0.29, 2, 2.73, (100, 185, 220, 35))
    for strip in [(0, 0.29, 0.06, 2.73), (1.94, 0.29, 2, 2.73), (0, 0.29, 2, 0.36), (0, 2.66, 2, 2.73)]:
        panel(*strip, (30, 35, 40, 255))
    panel(1.76, 1.25, 1.84, 1.75, (25, 30, 35, 255))
    image = Image.new("RGB", (960, 720), (235, 237, 240))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.ellipse((210, 530, 800, 690), fill=(40, 45, 50, 35))
    for _, points, color in sorted(faces, key=lambda f: f[0], reverse=True):
        projected = []
        for p in points:
            v = tuple(x - c for x, c in zip(p, camera, strict=True))
            depth = dot(v, forward)
            projected.append((480 + 1400 * dot(v, right) / depth, 360 - 1400 * dot(v, up) / depth))
        draw.polygon(projected, fill=color, outline=(25, 30, 35, 80))
    return image


def prepare(directory, *, images):
    for name, scene in images.items():
        path = Path(directory) / name
        if not path.exists():
            render(**scene).save(path)
