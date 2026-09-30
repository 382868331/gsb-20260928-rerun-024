from __future__ import annotations

import math

from .geometry import angle_between, distance, mean
from .skeleton import BONES, CHAIN_ARMS_LEGS


def calibrate_from_action(frames: list[dict]) -> tuple[dict, dict]:
    bone_samples: dict[str, list[float]] = {f"{a}->{b}": [] for a, b in BONES}
    angle_samples: dict[str, list[float]] = {joint: [] for joint in CHAIN_ARMS_LEGS}

    for frame in frames:
        qualified = any(marker.startswith("pA_") or marker.startswith("pB_") for marker in frame)
        prefixes = sorted({marker[:3] for marker in frame if marker[:2] in ("pA", "pB")}) if qualified else [""]
        for prefix in prefixes:
            def q(name: str) -> str:
                return f"{prefix}{name}" if prefix else name

            for start, end in BONES:
                if frame.get(q(start)) is not None and frame.get(q(end)) is not None:
                    bone_samples[f"{start}->{end}"].append(
                        distance(frame[q(start)], frame[q(end)])
                    )
            for joint, (parent, child) in CHAIN_ARMS_LEGS.items():
                if frame.get(q(joint)) and frame.get(q(parent)) and frame.get(q(child)):
                    incoming = [
                        frame[q(joint)][i] - frame[q(parent)][i] for i in range(3)
                    ]
                    outgoing = [
                        frame[q(child)][i] - frame[q(joint)][i] for i in range(3)
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
