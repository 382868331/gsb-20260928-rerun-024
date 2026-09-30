from __future__ import annotations

import math

from .geometry import angle_between, distance, mean
from .identity import split_marker
from .skeleton import BONES, CHAIN_ARMS_LEGS


def calibrate_from_action(frames: list[dict]) -> tuple[dict, dict]:
    bone_samples: dict[str, list[float]] = {f"{a}->{b}": [] for a, b in BONES}
    angle_samples: dict[str, list[float]] = {joint: [] for joint in CHAIN_ARMS_LEGS}

    for frame in frames:
        people = {None}
        for marker in frame:
            people.add(split_marker(marker)[0])
        for person_id in people:
            prefix = "" if person_id is None else f"{person_id}_"
            for start, end in BONES:
                person_start, person_end = f"{prefix}{start}", f"{prefix}{end}"
                if frame.get(person_start) is not None and frame.get(person_end) is not None:
                    bone_samples[f"{start}->{end}"].append(
                        distance(frame[person_start], frame[person_end])
                    )
        for person_id in people:
                prefix = "" if person_id is None else f"{person_id}_"
                for joint, (parent, child) in CHAIN_ARMS_LEGS.items():
                    person_joint, person_parent, person_child = (
                        f"{prefix}{joint}",
                        f"{prefix}{parent}",
                        f"{prefix}{child}",
                    )
                    if frame.get(person_joint) and frame.get(person_parent) and frame.get(person_child):
                        incoming = [
                            frame[person_joint][i] - frame[person_parent][i] for i in range(3)
                        ]
                        outgoing = [
                            frame[person_child][i] - frame[person_joint][i] for i in range(3)
                        ]
                        angle_samples[joint].append(angle_between(incoming, outgoing))

    bone_lengths = {}
    for name, values in bone_samples.items():
        if not values:
            continue
        center = mean(values)
        spread = max(abs(value - center) for value in values)
        bone_lengths[name] = {
            "length": round(center, 6),
            "tolerance": round(max(spread * 1.5, center * 0.03, 1e-5), 6),
        }

    joint_limits = {}
    for joint, values in angle_samples.items():
        if not values:
            continue
        low, high = min(values), max(values)
        margin = max((high - low) * 0.3, math.radians(10))
        joint_limits[joint] = {
            "min_angle": max(0.0, low - margin),
            "max_angle": min(math.pi, high + margin),
        }

    return bone_lengths, joint_limits
