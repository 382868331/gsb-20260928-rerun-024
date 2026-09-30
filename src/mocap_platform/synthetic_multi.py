from __future__ import annotations

import copy
from typing import Any

from .identity import person_marker
from .skeleton import JOINTS
from .synthetic import generate_known_action


PERSON_IDS = ("p0", "p1")
ARM_MARKERS = ("l_shoulder", "l_elbow", "l_wrist")
OPPOSITE_ARM_MARKERS = ("r_shoulder", "r_elbow", "r_wrist")
CROSSING_START = 20
CROSSING_END = 29
REAPPEARANCE_START = 40
REAPPEARANCE_END = 43


def _smoothstep(value: float) -> float:
    clipped = max(0.0, min(1.0, value))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _person_offset(frame_index: int, person_id: str, frames: list[dict[str, list[float]]]) -> list[float]:
    crossing_progress = _smoothstep((frame_index - 8) / 16)
    separation_x = -0.85 + 1.70 * crossing_progress
    if person_id == "p1":
        separation_x = -separation_x

    wrist_joint = "l_wrist" if person_id == "p0" else "r_wrist"
    wrist_x = frames[frame_index][wrist_joint][0]
    if person_id == "p1":
        target_x = -wrist_x
    else:
        target_x = -wrist_x

    contact_weight = _smoothstep(1.0 - abs(frame_index - 24) / 4.0)
    x = wrist_x + (1.0 - contact_weight) * separation_x + contact_weight * (target_x - wrist_x)
    y = 0.02 * (_smoothstep((frame_index - 34) / 10) - 0.5)
    z = 0.08 if person_id == "p1" else 0.0
    return [x, y, z]


def generate_multi_person_action(frame_count: int = 48) -> list[dict[str, list[float]]]:
    base = generate_known_action(frame_count)
    frames: list[dict[str, list[float]]] = []
    for frame_index, base_frame in enumerate(base):
        frame: dict[str, list[float]] = {}
        for person_id in PERSON_IDS:
            offset = _person_offset(frame_index, person_id, base)
            for joint in JOINTS:
                frame[person_marker(person_id, joint)] = [
                    base_frame[joint][axis] + offset[axis] for axis in range(3)
                ]
        frames.append(frame)
    return frames


def inject_multi_person_crossing(
    truth: list[dict[str, list[float]]],
) -> tuple[list[dict[str, list[float] | None]], list[dict[str, Any]], list[dict[str, Any]]]:
    observed: list[dict[str, list[float] | None]] = copy.deepcopy(truth)
    events: list[dict[str, Any]] = []

    for frame_index in range(CROSSING_START, CROSSING_END + 1):
        for left_joint, right_joint in zip(ARM_MARKERS, OPPOSITE_ARM_MARKERS):
            first = person_marker("p0", left_joint)
            second = person_marker("p1", right_joint)
            observed[frame_index][first], observed[frame_index][second] = (
                observed[frame_index][second],
                observed[frame_index][first],
            )
    events.append(
        {
            "type": "identity_swap",
            "markers": [
                *[person_marker("p0", marker) for marker in ARM_MARKERS],
                *[person_marker("p1", marker) for marker in OPPOSITE_ARM_MARKERS],
            ],
            "start_frame": CROSSING_START,
            "end_frame": CROSSING_END,
        }
    )

    for frame_index in range(22, 26):
        observed[frame_index][person_marker("p0", "l_wrist")] = None
        observed[frame_index][person_marker("p1", "r_wrist")] = None
    events.append(
        {
            "type": "simultaneous_short_occlusion",
            "markers": [person_marker("p0", "l_wrist"), person_marker("p1", "r_wrist")],
            "start_frame": 22,
            "end_frame": 25,
        }
    )

    for frame_index in range(8, 19):
        observed[frame_index][person_marker("p0", "head")] = None
    for frame_index in range(30, 41):
        observed[frame_index][person_marker("p1", "head")] = None
    events.extend(
        [
            {
                "type": "long_occlusion",
                "marker": person_marker("p0", "head"),
                "start_frame": 8,
                "end_frame": 18,
            },
            {
                "type": "long_occlusion",
                "marker": person_marker("p1", "head"),
                "start_frame": 30,
                "end_frame": 40,
            },
        ]
    )

    ambiguity_windows = [
        {
            "start_frame": CROSSING_START,
            "end_frame": CROSSING_END,
            "markers": [
                *[person_marker("p0", marker) for marker in ARM_MARKERS],
                *[person_marker("p1", marker) for marker in OPPOSITE_ARM_MARKERS],
            ],
            "reason": "two nearby people make both assignments plausible",
        },
        {
            "start_frame": REAPPEARANCE_START,
            "end_frame": REAPPEARANCE_END,
            "markers": [person_marker("p0", "l_wrist"), person_marker("p1", "r_wrist")],
            "reason": "reappearing wrist tracks have two plausible owners",
        },
    ]
    events.append(
        {
            "type": "equivocal_reappearance",
            "markers": ambiguity_windows[1]["markers"],
            "start_frame": REAPPEARANCE_START,
            "end_frame": REAPPEARANCE_END,
        }
    )
    return observed, events, ambiguity_windows


def multi_person_markers() -> list[str]:
    return [person_marker(person_id, joint) for person_id in PERSON_IDS for joint in JOINTS]


ANKLE_SWAP_START = 25
ANKLE_SWAP_END = 29


def inject_scoped_review_events(
    truth: list[dict[str, list[float]]],
) -> tuple[list[dict[str, list[float] | None]], list[dict[str, Any]], list[dict[str, Any]]]:
    observed: list[dict[str, list[float] | None]] = copy.deepcopy(truth)
    events: list[dict[str, Any]] = []

    for frame_index in range(CROSSING_START, CROSSING_END + 1):
        for left_joint, right_joint in zip(ARM_MARKERS, OPPOSITE_ARM_MARKERS):
            first = person_marker("p0", left_joint)
            second = person_marker("p1", right_joint)
            observed[frame_index][first], observed[frame_index][second] = (
                observed[frame_index][second],
                observed[frame_index][first],
            )
    events.append(
        {
            "type": "identity_swap",
            "markers": [
                *[person_marker("p0", marker) for marker in ARM_MARKERS],
                *[person_marker("p1", marker) for marker in OPPOSITE_ARM_MARKERS],
            ],
            "start_frame": CROSSING_START,
            "end_frame": CROSSING_END,
        }
    )

    for frame_index in range(ANKLE_SWAP_START, ANKLE_SWAP_END + 1):
        first = person_marker("p0", "r_ankle")
        second = person_marker("p1", "r_ankle")
        observed[frame_index][first], observed[frame_index][second] = (
            observed[frame_index][second],
            observed[frame_index][first],
        )
    events.append(
        {
            "type": "identity_swap",
            "markers": [person_marker("p0", "r_ankle"), person_marker("p1", "r_ankle")],
            "start_frame": ANKLE_SWAP_START,
            "end_frame": ANKLE_SWAP_END,
        }
    )

    for frame_index in range(27, 29):
        observed[frame_index][person_marker("p0", "r_ankle")] = None
    events.append(
        {
            "type": "local_occlusion",
            "marker": person_marker("p0", "r_ankle"),
            "start_frame": 27,
            "end_frame": 28,
        }
    )

    ambiguity_windows = [
        {
            "start_frame": CROSSING_START,
            "end_frame": CROSSING_END,
            "markers": [
                *[person_marker("p0", marker) for marker in ARM_MARKERS],
                *[person_marker("p1", marker) for marker in OPPOSITE_ARM_MARKERS],
            ],
            "reason": "two nearby people make both arm assignments plausible",
        },
        {
            "start_frame": ANKLE_SWAP_START,
            "end_frame": ANKLE_SWAP_END,
            "markers": [person_marker("p0", "r_ankle"), person_marker("p1", "r_ankle")],
            "reason": "right ankle tracks cross while people pass each other",
        },
        {
            "start_frame": REAPPEARANCE_START,
            "end_frame": REAPPEARANCE_END,
            "markers": [person_marker("p0", "l_wrist"), person_marker("p1", "r_wrist")],
            "reason": "reappearing wrist tracks have two plausible owners",
        },
    ]
    events.append(
        {
            "type": "equivocal_reappearance",
            "markers": ambiguity_windows[2]["markers"],
            "start_frame": REAPPEARANCE_START,
            "end_frame": REAPPEARANCE_END,
        }
    )
    return observed, events, ambiguity_windows
