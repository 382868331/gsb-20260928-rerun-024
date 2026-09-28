from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import Vec3


@dataclass(frozen=True)
class Observation:
    frame: int
    marker: str
    position: Vec3
    confidence: float = 1.0


@dataclass(frozen=True)
class IdentityRevision:
    revision_id: str
    frame_start: int
    frame_end: int
    mapping: dict[str, str]
    reason: str = ""

    def applies(self, frame: int) -> bool:
        return self.frame_start <= frame <= self.frame_end


@dataclass(frozen=True)
class SkeletonDefinition:
    joints: tuple[str, ...]
    root: str
    bones: tuple[tuple[str, str], ...]
    bone_lengths: dict[tuple[str, str], float]
    joint_limits: dict[tuple[str, str, str], tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.root not in self.joints:
            raise ValueError("root must be one of the joints")
        for parent, child in self.bones:
            if parent not in self.joints or child not in self.joints:
                raise ValueError(f"unknown joint in bone {(parent, child)}")
            if (parent, child) not in self.bone_lengths:
                raise ValueError(f"missing calibrated bone length for {(parent, child)}")

    def parent_of(self) -> dict[str, str | None]:
        parents: dict[str, str | None] = {joint: None for joint in self.joints}
        parents[self.root] = None
        for parent, child in self.bones:
            parents[child] = parent
        return parents


@dataclass(frozen=True)
class CaptureRecord:
    capture_id: str
    fps: float
    observations: tuple[Observation, ...]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def frame_count(self) -> int:
        if not self.observations:
            return 0
        return max(item.frame for item in self.observations) + 1


@dataclass(frozen=True)
class CalibrationRecord:
    calibration_id: str
    capture_version: str
    skeleton: SkeletonDefinition
    frame_start: int
    frame_end: int


@dataclass(frozen=True)
class JointState:
    position: Vec3 | None
    status: str
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessingRecord:
    processing_id: str
    capture_version: str
    calibration_version: str
    parent_processing_version: str | None
    revisions: tuple[IdentityRevision, ...]
    frames: tuple[dict[str, JointState], ...]
    unknown_frames: tuple[int, ...]
    warnings: tuple[str, ...]
