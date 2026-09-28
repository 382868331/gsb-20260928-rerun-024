from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, sin, sqrt


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Vec3":
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        magnitude = self.length()
        if magnitude == 0.0:
            return Vec3(0.0, 0.0, 0.0)
        return self / magnitude

    def distance(self, other: "Vec3") -> float:
        return (self - other).length()

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_list(cls, values: list[float]) -> "Vec3":
        return cls(float(values[0]), float(values[1]), float(values[2]))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_between(a: Vec3, b: Vec3) -> float:
    aa = a.normalized()
    bb = b.normalized()
    if aa.length() == 0 or bb.length() == 0:
        return 0.0
    return acos(clamp(aa.dot(bb), -1.0, 1.0))


def rotate_to_opening_angle(direction: Vec3, reference: Vec3, target_angle: float) -> Vec3:
    """Rotate direction in the plane shared with reference to a target opening angle."""
    axis = reference.cross(direction)
    axis_length = axis.length()
    if axis_length < 1e-10:
        return direction
    unit_axis = axis / axis_length
    current = angle_between(direction, reference)
    delta = target_angle - current
    c = cos(delta)
    s = sin(delta)
    return (
        direction * c
        + unit_axis.cross(direction) * s
        + unit_axis * (unit_axis.dot(direction) * (1.0 - c))
    )
