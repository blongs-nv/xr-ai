# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Choose default creation positions near the center of the current view."""

import math

from xr_ai_tools.types import SpatialFrame, Vector3
from xr_render_scene import SceneObject

_DISTANCE = 1.5
_HALF_ANGLE = math.radians(25)
_GAP = math.radians(2)
_STEP = 0.15


def _radius(prim_type: str, size: float) -> float:
    return size * math.sqrt(3) if prim_type == "box" else size


def _footprint(origin: Vector3, position: Vector3, radius: float) -> tuple[tuple[float, ...], float]:
    delta = tuple(getattr(position, axis) - getattr(origin, axis) for axis in ("x", "y", "z"))
    distance = math.sqrt(sum(value * value for value in delta))
    if distance <= radius:
        return (0.0, 0.0, 0.0), math.pi
    return tuple(value / distance for value in delta), math.asin(radius / distance)


def creation_positions(
    frame: SpatialFrame, objects: list[SceneObject], prim_type: str, size: float, count: int,
) -> list[Vector3]:
    """Fit bounding spheres without overlapping their angular footprints.

    The central 50-degree cone is a conservative placement policy, not a
    measurement of the client's field of view. Fail before creating anything
    when the requested objects cannot fit there.
    """
    occupied = [_footprint(frame.origin, obj.position, _radius(obj.type, obj.size)) for obj in objects]
    radius = _radius(prim_type, size)
    offsets = sorted(
        ((x, y) for x in range(-4, 5) for y in range(-4, 5)),
        key=lambda xy: (xy[0] ** 2 + xy[1] ** 2, abs(xy[1]), -xy[0], -xy[1]),
    )
    positions = []
    for x, y in offsets:
        position = Vector3(**{
            axis: round(getattr(frame.origin, axis) + _DISTANCE * getattr(frame.forward, axis)
                        + _STEP * x * getattr(frame.right, axis)
                        + _STEP * y * getattr(frame.up, axis), 3)
            for axis in ("x", "y", "z")
        })
        direction, angle = _footprint(frame.origin, position, radius)
        if math.atan2(_STEP * math.hypot(x, y), _DISTANCE) + angle > _HALF_ANGLE:
            continue
        if any(
            math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(direction, other, strict=True)))))
            < angle + other_angle + _GAP
            for other, other_angle in occupied
        ):
            continue
        positions.append(position)
        occupied.append((direction, angle))
        if len(positions) == count:
            return positions
    raise ValueError("There isn't enough clear space in the current view. Specify a position or remove an object.")
