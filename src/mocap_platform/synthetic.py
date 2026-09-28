from __future__ import annotations

import copy
import math
from typing import Any

from .geometry import add, scale, sub
from .skeleton import PARENTS

REST_POSE = {
    "pelvis": [0.0, 0.95, 0.0],
    "spine": [0.0, 1.25, 0.0],
    "neck": [0.0, 1.52, 0.0],
    "head": [0.0, 1.72, 0.02],
    "l_shoulder": [0.18, 1.48, 0.0],
    "l_elbow": [0.20, 1.18, 0.0],
    "l_wrist": [0.22, 0.88, 0.02],
    "r_shoulder": [-0.18, 1.48, 0.0],
    "r_elbow": [-0.20, 1.18, 0.0],
    "r_wrist": [-0.22, 0.88, 0.02],
    "l_hip": [0.10, 0.92, 0.0],
    "l_knee": [0.10, 0.55, 0.02],
    "l_ankle": [0.10, 0.10, -0.02],
    "r_hip": [-0.10, 0.92, 0.0],
    "r_knee": [-0.10, 0.55, 0.02],
    "r_ankle": [-0.10, 0.10, -0.02],
}

# Rotations are yaw in the marker's local x/y plane around the parent joint.
MOTION = {
    "spine": (0.04, 0.0),
    "neck": (0.05, 0.4),
    "head": (0.07, 0.8),
    "l_shoulder": (0.28, 0.0),
    "l_elbow": (0.18, 0.7),
    "l_wrist": (0.16, 1.2),
    "r_shoulder": (0.30, math.pi),
    "r_elbow": (0.20, math.pi + 0.6),
    "r_wrist": (0.18, math.pi + 1.1),
    "l_hip": (0.12, 0.3),
    "l_knee": (0.22, 1.0),
    "l_ankle": (0.08, 0.2),
    "r_hip": (0.12, math.pi + 0.3),
    "r_knee": (0.24, math.pi + 0.8),
    "r_ankle": (0.08, math.pi + 0.2),
}


def _rotate_xy(vector: list[float], angle: float) -> list[float]:
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        vector[0] * cosine - vector[1] * sine,
        vector[0] * sine + vector[1] * cosine,
        vector[2],
    ]


def generate_known_action(frame_count: int = 42) -> list[dict[str, list[float]]]:
    frames = []
    for frame_index in range(frame_count):
        phase = 2.0 * math.pi * frame_index / frame_count
        pose: dict[str, list[float]] = {
            "pelvis": [
                0.03 * math.sin(phase),
                REST_POSE["pelvis"][1] + 0.015 * math.sin(2 * phase),
                0.02 * math.cos(phase),
            ]
        }
        for joint in [joint for joint in REST_POSE if joint != "pelvis"]:
            amplitude, offset = MOTION[joint]
            rest_vector = sub(REST_POSE[joint], REST_POSE[PARENTS[joint]])
            rotated = _rotate_xy(rest_vector, amplitude * math.sin(phase + offset))
            pose[joint] = add(pose[PARENTS[joint]], rotated)
        frames.append(pose)
    return frames


def inject_occlusions_and_swaps(
    truth: list[dict[str, list[float]]],
) -> tuple[list[dict[str, list[float] | None]], list[dict[str, Any]]]:
    observed: list[dict[str, list[float] | None]] = copy.deepcopy(truth)
    events = []

    # Two short gaps that interpolation should recover.
    for frame_index in range(8, 12):
        observed[frame_index]["l_wrist"] = None
    events.append({"type": "occlusion", "marker": "l_wrist", "start_frame": 8, "end_frame": 11})

    for frame_index in range(25, 28):
        observed[frame_index]["r_knee"] = None
    events.append({"type": "occlusion", "marker": "r_knee", "start_frame": 25, "end_frame": 27})

    # A gap longer than the default reconstruction limit must remain undetermined.
    for frame_index in range(13, 24):
        observed[frame_index]["head"] = None
    events.append({"type": "long_occlusion", "marker": "head", "start_frame": 13, "end_frame": 23})

    # Manual labels for the wrists are exchanged during a short interval.
    for frame_index in range(30, 36):
        left = observed[frame_index]["l_wrist"]
        right = observed[frame_index]["r_wrist"]
        observed[frame_index]["l_wrist"] = right
        observed[frame_index]["r_wrist"] = left
    events.append({"type": "marker_swap", "markers": ["l_wrist", "r_wrist"], "start_frame": 30, "end_frame": 35})

    return observed, events
