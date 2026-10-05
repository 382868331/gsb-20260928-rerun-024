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
    depends_on: list[str] = field(default_factory=list)
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
            "depends_on": self.depends_on,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass
class TrajectoryReassignment:
    reassignment_id: str
    source_person_id: str
    target_person_id: str
    joints: list[str]
    start_frame: int
    end_frame: int
    reason: str
    target_joints: list[str] = field(default_factory=list)
    kept_decision_ids: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    status: Literal["proposed", "applied", "disputed", "revoked"] = "proposed"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reassignment_id": self.reassignment_id,
            "source_person_id": self.source_person_id,
            "target_person_id": self.target_person_id,
            "joints": self.joints,
            "target_joints": self.target_joints,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "reason": self.reason,
            "kept_decision_ids": self.kept_decision_ids,
            "depends_on": self.depends_on,
            "status": self.status,
            "created_at": self.created_at,
        }


MarkerStatus = Literal["observed", "recovered", "uncertain", "missing"]
IdentityStatus = Literal["certain", "confirmed", "ambiguous"]


@dataclass
class ObjectTrack:
    """Observed trajectory of a contactable object; missing frames stay null."""

    object_id: str
    positions: list[Point3]
    source: str = "manual"
    notes: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "object_track",
            "version": 1,
            "object_id": self.object_id,
            "positions": self.positions,
            "source": self.source,
            "notes": self.notes,
            "created_at": self.created_at,
        }


@dataclass
class ContactObservation:
    """Sourced manual statement that a person's body part touched an object.

    Contact is evidence only: it never confirms an identity by itself and a
    later automatic recommendation must never overwrite a manual statement.
    """

    contact_id: str
    person_id: str
    body_part: str
    object_id: str
    start_frame: int
    end_frame: int
    source: str
    reason: str = ""
    simultaneous_with: list[str] = field(default_factory=list)
    status: Literal["active", "revoked"] = "active"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contact_id": self.contact_id,
            "person_id": self.person_id,
            "body_part": self.body_part,
            "object_id": self.object_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "source": self.source,
            "reason": self.reason,
            "simultaneous_with": self.simultaneous_with,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass
class IdentityCandidate:
    """One identity interpretation of a body trajectory inside a crossing.

    ``joint_mapping`` maps candidate-person joint names to the observed marker
    (person-qualified) actually carrying that joint under this interpretation.
    Two candidates may map different identities, but the same observed marker
    can never be claimed by two identities on the same frame.
    """

    candidate_id: str
    person_id: str
    joint_mapping: dict[str, str]
    start_frame: int
    end_frame: int
    reason: str = ""
    decision: Literal["adopted", "excluded", "open"] = "open"
    decision_reason: str = ""
    depends_on: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "person_id": self.person_id,
            "joint_mapping": self.joint_mapping,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "reason": self.reason,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "depends_on": self.depends_on,
            "created_at": self.created_at,
        }


@dataclass
class IdentityScheme:
    """Independent saved interpretation of the same crossing event."""

    scheme_id: str
    label: str
    crossing_start_frame: int
    crossing_end_frame: int
    candidate_ids: list[str] = field(default_factory=list)
    adopted_reassignment_ids: list[str] = field(default_factory=list)
    notes: str = ""
    status: Literal["open", "resolved", "superseded"] = "open"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme_id": self.scheme_id,
            "label": self.label,
            "crossing_start_frame": self.crossing_start_frame,
            "crossing_end_frame": self.crossing_end_frame,
            "candidate_ids": self.candidate_ids,
            "adopted_reassignment_ids": self.adopted_reassignment_ids,
            "notes": self.notes,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass
class SupplementaryBatch:
    """One batch of supplementary observations submitted for contact gaps.

    ``tracks`` maps a target name ("object" for the object itself or a
    person-qualified marker such as "p0_l_wrist") to
    {"start_frame": int, "positions": [Point3, ...]}.  Tracks in one batch
    may have different lengths and only partially overlap.  Batches are saved
    as ``pending``; reviewers explicitly adopt chosen missing intervals, and
    adoption/withdrawal are appended as separate records so the batch itself
    is never rewritten.
    """

    batch_id: str
    object_id: str
    tracks: dict[str, dict[str, Any]]
    source: str
    reason: str = ""
    adopted_intervals: list[dict[str, int]] = field(default_factory=list)
    status: Literal["pending", "partially_adopted", "adopted", "withdrawn"] = "pending"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "object_id": self.object_id,
            "tracks": self.tracks,
            "source": self.source,
            "reason": self.reason,
            "adopted_intervals": self.adopted_intervals,
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
    reassignments: list["TrajectoryReassignment"] = field(default_factory=list)
    contact_observations: list["ContactObservation"] = field(default_factory=list)
    identity_schemes: list["IdentityScheme"] = field(default_factory=list)
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
            "trajectory_reassignments": [
                reassignment.to_dict() for reassignment in self.reassignments
            ],
            "contact_observations": [
                contact.to_dict() for contact in self.contact_observations
            ],
            "identity_schemes": [scheme.to_dict() for scheme in self.identity_schemes],
            "frames": self.frames,
            "statuses": self.statuses,
            "identity_statuses": self.identity_statuses,
        }
