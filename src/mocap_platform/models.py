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
    marker: str
    start_frame: int
    end_frame: int
    corrected_identity: str
    reason: str
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "corrected_identity": self.corrected_identity,
            "reason": self.reason,
            "created_at": self.created_at,
        }


@dataclass
class IdentityDecision:
    decision_id: str
    person_id: str
    marker: str
    start_frame: int
    end_frame: int
    decision: Literal["confirm", "deny"]
    alternative_person_id: str
    reason: str
    related_markers: list[str] = field(default_factory=list)
    alternative_related_markers: list[str] = field(default_factory=list)
    status: Literal["active", "revoked"] = "active"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "person_id": self.person_id,
            "marker": self.marker,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "decision": self.decision,
            "alternative_person_id": self.alternative_person_id,
            "reason": self.reason,
            "related_markers": self.related_markers,
            "alternative_related_markers": self.alternative_related_markers,
            "status": self.status,
            "created_at": self.created_at,
        }


MarkerStatus = Literal["observed", "recovered", "uncertain", "missing"]
IdentityStatus = Literal[
    "certain",
    "confirmed",
    "ambiguous",
    "pending",
    "disputed",
    "reassigned",
]


@dataclass
class TrackReassignment:
    reassignment_id: str
    source_person_id: str
    target_person_id: str
    marker: str
    start_frame: int
    end_frame: int
    related_markers: list[str] = field(default_factory=list)
    target_related_markers: list[str] = field(default_factory=list)
    resolution: Literal["paired_adjustment", "disputed"] = "paired_adjustment"
    paired_reassignment_id: str | None = None
    reason: str = ""
    status: Literal["active", "revoked"] = "active"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "track_reassignment",
            "reassignment_id": self.reassignment_id,
            "source_person_id": self.source_person_id,
            "target_person_id": self.target_person_id,
            "marker": self.marker,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "related_markers": self.related_markers,
            "target_related_markers": self.target_related_markers,
            "resolution": self.resolution,
            "paired_reassignment_id": self.paired_reassignment_id,
            "reason": self.reason,
            "status": self.status,
            "created_at": self.created_at,
        }


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
    decisions: list[IdentityDecision] = field(default_factory=list)
    reassignments: list[TrackReassignment] = field(default_factory=list)
    identity_statuses: list[dict[str, IdentityStatus]] = field(default_factory=list)
    based_on: str | None = None
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
            "identity_decisions": [decision.to_dict() for decision in self.decisions],
            "track_reassignments": [item.to_dict() for item in self.reassignments],
            "frames": self.frames,
            "statuses": self.statuses,
            "identity_statuses": self.identity_statuses,
        }
