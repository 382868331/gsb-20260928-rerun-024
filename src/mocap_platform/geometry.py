from __future__ import annotations

import math
from typing import Iterable, Sequence

Vec3 = list[float]


def add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def scale(a: Sequence[float], value: float) -> Vec3:
    return [a[0] * value, a[1] * value, a[2] * value]


def norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(component * component for component in a))


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return norm(sub(a, b))


def normalize(a: Sequence[float]) -> Vec3:
    length = norm(a)
    if length < 1e-12:
        return [0.0, 0.0, 0.0]
    return scale(a, 1.0 / length)


def lerp(a: Sequence[float], b: Sequence[float], t: float) -> Vec3:
    return add(scale(a, 1.0 - t), scale(b, t))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = normalize(a), normalize(b)
    cosine = max(-1.0, min(1.0, dot(na, nb)))
    return math.acos(cosine)


def rotate_to_cone(
    point: Sequence[float],
    vertex: Sequence[float],
    axis: Sequence[float],
    max_angle: float,
) -> Vec3:
    """Project a direction from vertex into a cone around axis."""
    vector = sub(point, vertex)
    current = angle_between(vector, axis)
    if current <= max_angle or norm(vector) < 1e-12:
        return list(point)
    unit_axis = normalize(axis)
    unit_vector = normalize(vector)
    ortho = sub(unit_vector, scale(unit_axis, dot(unit_vector, unit_axis)))
    ortho_length = norm(ortho)
    if ortho_length < 1e-12:
        return list(point)
    ortho = scale(ortho, 1.0 / ortho_length)
    direction = add(
        scale(unit_axis, math.cos(max_angle)),
        scale(ortho, math.sin(max_angle)),
    )
    return add(vertex, scale(direction, norm(vector)))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)
