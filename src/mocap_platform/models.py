from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Point3 = list[float] | None
Frame = dict[str, Point3]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CaptureRecord:
    capture_id: str
    action_name: str
    frame_count: int
    frame_rate: float
    markers: list[str]
    observed_frames: list[Frame]
    ground_truth_frames: list[Frame] = field(default_factory=list)
    source: str = "manual"
    notes: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "capture",
            "version": 1,
            "capture_id": self.capture_id,
            "action_name": self.action_name,
            "frame_count": self.frame_count,
            "frame_rate": self.frame_rate,
            "markers": self.markers,
            "source": self.source,
            "notes": self.notes,
            "created_at": self.created_at,
            "observations": self.observed_frames,
            "ground_truth": self.ground_truth_frames,
        }


@dataclass
class Calibration:
    calibration_id: str
    capture_id: str
    bone_lengths: dict[str, dict[str, float]]
    joint_limits: dict[str, dict[str, float]]
    source_version: str = "v001"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "calibration",
            "version": self.source_version,
            "calibration_id": self.calibration_id,
            "capture_id": self.capture_id,
            "created_at": self.created_at,
            "bone_lengths": self.bone_lengths,
            "joint_limits": self.joint_limits,
        }


@dataclass
class IdentityRevision:
    revision_id: str = ""
    marker: str = ""
    start_frame: int = 0
    end_frame: int = 0
    corrected_identity: str = ""
    reason: str = ""
    decision: str = "confirm"
    group: str = ""
    propagated: bool = False
    supersedes: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "marker": self.marker,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "corrected_identity": self.corrected_identity,
            "reason": self.reason,
            "decision": self.decision,
            "group": self.group,
            "propagated": self.propagated,
            "supersedes": self.supersedes,
            "created_at": self.created_at,
        }


MarkerStatus = Literal["observed", "recovered", "uncertain", "missing"]


@dataclass
class ProcessingVersion:
    action_name: str
    pipeline_name: str
    version: str
    capture_id: str
    calibration_version: str
    parameters: dict[str, Any]
    frames: list[Frame]
    statuses: list[dict[str, MarkerStatus]]
    revisions: list[IdentityRevision] = field(default_factory=list)
    based_on: str | None = None
    ambiguities: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "processing",
            "version": self.version,
            "action_name": self.action_name,
            "pipeline_name": self.pipeline_name,
            "capture_id": self.capture_id,
            "calibration_version": self.calibration_version,
            "based_on": self.based_on,
            "created_at": self.created_at,
            "parameters": self.parameters,
            "identity_revisions": [revision.to_dict() for revision in self.revisions],
            "ambiguities": self.ambiguities,
            "frames": self.frames,
            "statuses": self.statuses,
        }
