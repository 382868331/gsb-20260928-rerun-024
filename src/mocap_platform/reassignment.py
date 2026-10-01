from __future__ import annotations

from typing import Any

from .identity import decision_affected_markers, person_marker
from .models import IdentityDecision, IdentityRevision, TrajectoryReassignment


def reassignment_target_joints(reassignment: TrajectoryReassignment) -> list[str]:
    return reassignment.target_joints or reassignment.joints


def reassignment_source_markers(reassignment: TrajectoryReassignment) -> set[str]:
    return {
        person_marker(reassignment.source_person_id, joint)
        for joint in reassignment.joints
    }


def reassignment_markers(reassignment: TrajectoryReassignment) -> set[str]:
    return reassignment_source_markers(reassignment) | {
        person_marker(reassignment.target_person_id, joint)
        for joint in reassignment_target_joints(reassignment)
    }


def reassignment_trajectories(reassignment: TrajectoryReassignment) -> list[dict[str, Any]]:
    return [
        {
            "marker": marker,
            "start_frame": reassignment.start_frame,
            "end_frame": reassignment.end_frame,
        }
        for marker in sorted(reassignment_markers(reassignment))
    ]


def _frames_overlap(
    first_start: int, first_end: int, second_start: int, second_end: int
) -> bool:
    return first_start <= second_end and second_start <= first_end


def reassignments_contend(
    first: TrajectoryReassignment, second: TrajectoryReassignment
) -> bool:
    if not _frames_overlap(
        first.start_frame, first.end_frame, second.start_frame, second.end_frame
    ):
        return False
    return bool(
        reassignment_source_markers(first) & reassignment_source_markers(second)
    )


def resolve_reassignments(
    reassignments: list[TrajectoryReassignment],
) -> tuple[list[TrajectoryReassignment], list[dict[str, Any]], set[str]]:
    active = sorted(
        (
            reassignment
            for reassignment in reassignments
            if reassignment.status in {"proposed", "applied"}
        ),
        key=lambda reassignment: (
            reassignment.created_at,
            reassignment.reassignment_id,
        ),
    )
    accepted: list[TrajectoryReassignment] = []
    disputes: list[dict[str, Any]] = []
    disputed_ids: set[str] = set()
    for reassignment in active:
        blocker = next(
            (
                other
                for other in accepted
                if reassignments_contend(reassignment, other)
            ),
            None,
        )
        if blocker is None:
            accepted.append(reassignment)
            continue
        disputed_ids.add(reassignment.reassignment_id)
        shared = reassignment_source_markers(reassignment) & reassignment_source_markers(
            blocker
        )
        disputes.append(
            {
                "reassignment_id": reassignment.reassignment_id,
                "disputes_with": blocker.reassignment_id,
                "start_frame": max(reassignment.start_frame, blocker.start_frame),
                "end_frame": min(reassignment.end_frame, blocker.end_frame),
                "markers": sorted(shared),
                "reassignment_trajectories": reassignment_trajectories(reassignment),
                "disputes_with_trajectories": reassignment_trajectories(blocker),
                "reason": (
                    "the same trajectory segment is already adopted by another "
                    "reassignment; both are kept visible and neither is shared "
                    "silently"
                ),
            }
        )
    applied = [
        reassignment for reassignment in accepted if reassignment.status == "applied"
    ]
    return applied, disputes, disputed_ids


def pending_confirmations(
    decisions: list[IdentityDecision],
    reassignments: list[TrajectoryReassignment],
    applied: list[TrajectoryReassignment],
) -> list[dict[str, Any]]:
    revoked_ids = {
        reassignment.reassignment_id
        for reassignment in reassignments
        if reassignment.status == "revoked"
    }
    pending: list[dict[str, Any]] = []
    for decision in decisions:
        if decision.status != "active":
            continue
        revoked_dependencies = sorted(set(decision.depends_on) & revoked_ids)
        if revoked_dependencies:
            pending.append(
                {
                    "decision_id": decision.decision_id,
                    "reason": "dependency_revoked",
                    "dependencies": revoked_dependencies,
                    "start_frame": decision.start_frame,
                    "end_frame": decision.end_frame,
                    "markers": sorted(decision_affected_markers(decision)),
                }
            )
            continue
        for reassignment in applied:
            if decision.decision_id in reassignment.kept_decision_ids:
                continue
            shared = decision_affected_markers(decision) & reassignment_markers(
                reassignment
            )
            if not shared or not _frames_overlap(
                decision.start_frame,
                decision.end_frame,
                reassignment.start_frame,
                reassignment.end_frame,
            ):
                continue
            pending.append(
                {
                    "decision_id": decision.decision_id,
                    "reason": "trajectory_reassigned",
                    "reassignment_id": reassignment.reassignment_id,
                    "start_frame": max(
                        decision.start_frame, reassignment.start_frame
                    ),
                    "end_frame": min(decision.end_frame, reassignment.end_frame),
                    "markers": sorted(shared),
                }
            )
            break
    return pending


def reassignments_to_revisions(
    reassignments: list[TrajectoryReassignment],
) -> list[IdentityRevision]:
    revisions: list[IdentityRevision] = []
    for reassignment in reassignments:
        target_joints = reassignment_target_joints(reassignment)
        for source_joint, target_joint in zip(reassignment.joints, target_joints):
            source = person_marker(reassignment.source_person_id, source_joint)
            target = person_marker(reassignment.target_person_id, target_joint)
            if source == target:
                continue
            reason = (
                "manual trajectory reassignment "
                f"{reassignment.reassignment_id}: {reassignment.reason}"
            )
            revisions.append(
                IdentityRevision(
                    marker=source,
                    start_frame=reassignment.start_frame,
                    end_frame=reassignment.end_frame,
                    corrected_identity=target,
                    reason=reason,
                    created_at=reassignment.created_at,
                )
            )
            revisions.append(
                IdentityRevision(
                    marker=target,
                    start_frame=reassignment.start_frame,
                    end_frame=reassignment.end_frame,
                    corrected_identity=source,
                    reason=reason,
                    created_at=reassignment.created_at,
                )
            )
    return revisions


def reassignment_impact_preview(
    reassignment: TrajectoryReassignment,
    decisions: list[IdentityDecision],
    ambiguity_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    markers = reassignment_markers(reassignment)
    affected_decisions: list[dict[str, Any]] = []
    for decision in decisions:
        if decision.status != "active":
            continue
        shared = decision_affected_markers(decision) & markers
        if not shared or not _frames_overlap(
            decision.start_frame,
            decision.end_frame,
            reassignment.start_frame,
            reassignment.end_frame,
        ):
            continue
        affected_decisions.append(
            {
                "decision_id": decision.decision_id,
                "start_frame": max(decision.start_frame, reassignment.start_frame),
                "end_frame": min(decision.end_frame, reassignment.end_frame),
                "markers": sorted(shared),
                "kept": decision.decision_id in reassignment.kept_decision_ids,
            }
        )
    affected_windows = [
        {
            "start_frame": max(window["start_frame"], reassignment.start_frame),
            "end_frame": min(window["end_frame"], reassignment.end_frame),
            "markers": sorted(set(window.get("markers", [])) & markers),
            "reason": window.get("reason", ""),
        }
        for window in ambiguity_windows
        if _frames_overlap(
            window["start_frame"],
            window["end_frame"],
            reassignment.start_frame,
            reassignment.end_frame,
        )
        and set(window.get("markers", [])) & markers
    ]
    return {
        "affected_decisions": affected_decisions,
        "affected_ambiguity_windows": affected_windows,
        "skeleton_markers": sorted(markers),
    }
