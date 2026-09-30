from __future__ import annotations

from dataclasses import dataclass


# Parent-facing and child-facing bones meet at each non-root joint.
JOINTS = [
    "pelvis",
    "spine",
    "neck",
    "head",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "l_hip",
    "l_knee",
    "l_ankle",
    "r_hip",
    "r_knee",
    "r_ankle",
]

BONES: list[tuple[str, str]] = [
    ("pelvis", "spine"),
    ("spine", "neck"),
    ("neck", "head"),
    ("spine", "l_shoulder"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow", "l_wrist"),
    ("spine", "r_shoulder"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow", "r_wrist"),
    ("pelvis", "l_hip"),
    ("l_hip", "l_knee"),
    ("l_knee", "l_ankle"),
    ("pelvis", "r_hip"),
    ("r_hip", "r_knee"),
    ("r_knee", "r_ankle"),
]

PARENTS = {
    "spine": "pelvis",
    "neck": "spine",
    "head": "neck",
    "l_shoulder": "spine",
    "l_elbow": "l_shoulder",
    "l_wrist": "l_elbow",
    "r_shoulder": "spine",
    "r_elbow": "r_shoulder",
    "r_wrist": "r_elbow",
    "l_hip": "pelvis",
    "l_knee": "l_hip",
    "l_ankle": "l_knee",
    "r_hip": "pelvis",
    "r_knee": "r_hip",
    "r_ankle": "r_knee",
}


@dataclass(frozen=True)
class BoneConstraint:
    start: str
    end: str
    length: float
    tolerance: float

    @property
    def minimum(self) -> float:
        return max(0.0, self.length - self.tolerance)

    @property
    def maximum(self) -> float:
        return self.length + self.tolerance


@dataclass(frozen=True)
class JointLimit:
    joint: str
    parent_joint: str
    child_joint: str
    min_angle: float
    max_angle: float


CHAIN_ARMS_LEGS = {
    "l_elbow": ("l_shoulder", "l_wrist"),
    "l_shoulder": ("spine", "l_elbow"),
    "r_elbow": ("r_shoulder", "r_wrist"),
    "r_shoulder": ("spine", "r_elbow"),
    "l_knee": ("l_hip", "l_ankle"),
    "l_hip": ("pelvis", "l_knee"),
    "r_knee": ("r_hip", "r_ankle"),
    "r_hip": ("pelvis", "r_knee"),
    "neck": ("spine", "head"),
    "spine": ("pelvis", "neck"),
}

PERSONS = ("A", "B")

# Limb groups whose markers move together when a tracked identity is
# reassigned. A confirmation/denial on one marker propagates to the group.
LIMB_GROUPS: dict[str, list[str]] = {
    "r_arm": ["r_shoulder", "r_elbow", "r_wrist"],
    "l_arm": ["l_shoulder", "l_elbow", "l_wrist"],
    "r_leg": ["r_hip", "r_knee", "r_ankle"],
    "l_leg": ["l_hip", "l_knee", "l_ankle"],
}

# Mirror group on the other person used when one side is denied without an
# explicit target (left/right are exchanged during a close pass).
MIRROR_GROUP = {
    ("A", "r_arm"): ("B", "l_arm"),
    ("A", "l_arm"): ("B", "r_arm"),
    ("B", "r_arm"): ("A", "l_arm"),
    ("B", "l_arm"): ("A", "r_arm"),
    ("A", "r_leg"): ("B", "l_leg"),
    ("A", "l_leg"): ("B", "r_leg"),
    ("B", "r_leg"): ("A", "l_leg"),
    ("B", "l_leg"): ("A", "r_leg"),
}


def qualified(person: str, joint: str) -> str:
    return f"p{person}_{joint}"


def split_marker(name: str) -> tuple[str, str]:
    """Split ``pA_r_wrist`` into ``('A', 'r_wrist')``."""
    return name[1], name[3:]


def group_markers(person: str, group: str) -> list[str]:
    return [qualified(person, joint) for joint in LIMB_GROUPS[group]]


def marker_group(joint: str) -> str | None:
    for group_name, joints in LIMB_GROUPS.items():
        if joint in joints:
            return group_name
    return None
