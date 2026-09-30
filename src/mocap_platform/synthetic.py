from __future__ import annotations

import copy
import math
from typing import Any

from .geometry import add, sub
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


def _symmetric_contact_pose() -> dict[str, list[float]]:
    pose: dict[str, list[float]] = {}
    contact = [0.0, 1.12, 0.05]
    for person, facing in (("A", -1.0), ("B", 1.0)):
        spine = [facing * 0.60, REST_POSE["spine"][1], 0.0]
        pelvis = [facing * 0.60, REST_POSE["pelvis"][1], 0.0]
        neck = [facing * 0.60, REST_POSE["neck"][1], 0.0]
        head = [facing * 0.60, REST_POSE["head"][1], 0.02]
        involved = "r_" if person == "A" else "l_"
        other = "l_" if person == "A" else "r_"
        shoulder = [facing * 0.42, REST_POSE[f"{involved}shoulder"][1], 0.0]
        elbow = [facing * 0.21, 1.12, 0.03]
        wrist = [contact[0], contact[1], contact[2] + (0.01 if person == "A" else -0.01)]
        values = {
            "pelvis": pelvis,
            "spine": spine,
            "neck": neck,
            "head": head,
            f"{involved}shoulder": shoulder,
            f"{involved}elbow": elbow,
            f"{involved}wrist": wrist,
            f"{other}shoulder": [facing * 0.78, REST_POSE[f"{other}shoulder"][1], 0.0],
            f"{other}elbow": [facing * 0.80, REST_POSE[f"{other}elbow"][1], 0.0],
            f"{other}wrist": [facing * 0.82, REST_POSE[f"{other}wrist"][1], 0.02],
        }
        for side in ("l", "r"):
            for joint in ("hip", "knee", "ankle"):
                name = f"{side}_{joint}"
                rest = REST_POSE[name]
                values[name] = [facing * 0.60 + rest[0], rest[1], rest[2]]
        for joint, point in values.items():
            pose[f"p{person}_{joint}"] = point
    return pose


TWO_PERSON_FRAME_COUNT = 60


def generate_two_person_action(
    frame_count: int = TWO_PERSON_FRAME_COUNT,
) -> list[dict[str, list[float]]]:
    """Two actors walk toward each other, cross arms, then separate.

    Person A starts on the -x side and person B on the +x side. Around the
    middle of the take they face each other and bring their right arms
    together in a mirror-symmetric pass, which is the identity-crisis region.
    """
    frames: list[dict[str, list[float]]] = []
    single = generate_known_action(frame_count)
    for frame_index, base_pose in enumerate(single):
        phase = 2.0 * math.pi * frame_index / frame_count
        approach = -1.6 + 3.2 * frame_index / (frame_count - 1)
        pose: dict[str, list[float]] = {}
        for person, shift, facing in (("A", approach, 1.0), ("B", -approach, -1.0)):
            for joint, point in base_pose.items():
                moved = list(point)
                moved[0] = moved[0] * facing + shift
                moved[1] += 0.01 * math.sin(phase + (0.0 if person == "A" else math.pi))
                pose[f"p{person}_{joint}"] = moved
        frames.append(pose)
    return frames


def inject_two_person_events(
    truth: list[dict[str, list[float]]],
) -> tuple[list[dict[str, list[float] | None]], list[dict[str, Any]]]:
    """Corrupt one continuous manual take.

    Frames 18-23: the two right arms (mirror groups pA_r_arm and pB_l_arm)
    are labelled with each other's identities during the close approach.
    Frames 24-27: both involved wrists are occluded simultaneously (short).
    Frames 28-31: the actors reappear in a symmetric near-contact pose where
    either assignment is equally plausible without human input.
    A long head gap on person A must remain uncertaired.
    """
    observed: list[dict[str, list[float] | None]] = copy.deepcopy(truth)
    events: list[dict[str, Any]] = []

    group_a = ["pA_r_shoulder", "pA_r_elbow", "pA_r_wrist"]
    group_b = ["pB_l_shoulder", "pB_l_elbow", "pB_l_wrist"]

    for frame_index in range(18, 24):
        for marker_a, marker_b in zip(group_a, group_b):
            observed[frame_index][marker_a], observed[frame_index][marker_b] = (
                observed[frame_index][marker_b],
                observed[frame_index][marker_a],
            )
    events.append(
        {
            "type": "identity_cross",
            "groups": ["pA_r_arm", "pB_l_arm"],
            "start_frame": 18,
            "end_frame": 23,
        }
    )

    for frame_index in range(24, 28):
        observed[frame_index]["pA_r_wrist"] = None
        observed[frame_index]["pB_l_wrist"] = None
    events.append(
        {
            "type": "simultaneous_occlusion",
            "markers": ["pA_r_wrist", "pB_l_wrist"],
            "start_frame": 24,
            "end_frame": 27,
        }
    )

    # Mirror-symmetric reappearance. Both actors face each other with the
    # involved arms extended toward the contact point; the elbow/wrist layout
    # is reflected, so swapping it preserves every internal bone length -- two
    # equally plausible assignments. The upper body is rebuilt to keep all bone
    # lengths valid, and ground truth receives the identical pose.
    for frame_index in range(28, 32):
        pose = _symmetric_contact_pose()
        for marker, point in pose.items():
            observed[frame_index][marker] = point
            truth[frame_index][marker] = point
    events.append(
        {
            "type": "ambiguous_reappearance",
            "groups": ["pA_r_arm", "pB_l_arm"],
            "start_frame": 28,
            "end_frame": 31,
        }
    )

    for frame_index in range(40, 52):
        observed[frame_index]["pA_head"] = None
    events.append(
        {
            "type": "long_occlusion",
            "marker": "pA_head",
            "start_frame": 40,
            "end_frame": 51,
        }
    )

    return observed, events
