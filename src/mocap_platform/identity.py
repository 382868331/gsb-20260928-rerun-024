from __future__ import annotations

import re
from typing import Any

from .models import IdentityDecision, IdentityRevision, TrackReassignment


PERSON_MARKER = re.compile(r"^(p\d+)_(.+)$")


def split_marker(marker: str) -> tuple[str | None, str]:
    match = PERSON_MARKER.match(marker)
    if match is None:
        return None, marker
    return match.group(1), match.group(2)


def person_marker(person_id: str, joint_name: str) -> str:
    return f"{person_id}_{joint_name}"


def kinematic_group(joint_name: str) -> list[str]:
    side = joint_name[0]
    if "wrist" in joint_name or "elbow" in joint_name or "shoulder" in joint_name:
        return [
            f"{side}_shoulder",
            f"{side}_elbow",
            f"{side}_wrist",
        ]
    if "ankle" in joint_name or "knee" in joint_name or "hip" in joint_name:
        return [
            f"{side}_hip",
            f"{side}_knee",
            f"{side}_ankle",
        ]
    if joint_name in {"head", "neck", "spine", "pelvis"}:
        return ["pelvis", "spine", "neck", "head"]
    return [joint_name]


def decision_joint_groups(decision: IdentityDecision) -> tuple[list[str], list[str]]:
    group = decision.related_markers or kinematic_group(decision.marker)
    alternative_group = (
        decision.alternative_related_markers
        or decision.related_markers
        or kinematic_group(decision.marker)
    )
    return group, alternative_group


def decision_affected_markers(decision: IdentityDecision) -> set[str]:
    group, alternative_group = decision_joint_groups(decision)
    return {
        person_marker(decision.person_id, joint_name) for joint_name in group
    } | {
        person_marker(decision.alternative_person_id, joint_name)
        for joint_name in alternative_group
    }


def decision_trajectories(decision: IdentityDecision) -> list[dict[str, Any]]:
    return [
        {
            "marker": marker,
            "start_frame": decision.start_frame,
            "end_frame": decision.end_frame,
        }
        for marker in sorted(decision_affected_markers(decision))
    ]


def _frames_overlap(first: IdentityDecision, second: IdentityDecision) -> bool:
    return (
        first.start_frame <= second.end_frame
        and second.start_frame <= first.end_frame
    )


def decisions_conflict(first: IdentityDecision, second: IdentityDecision) -> bool:
    if not _frames_overlap(first, second):
        return False
    if not decision_affected_markers(first) & decision_affected_markers(second):
        return False
    return (
        first.decision != second.decision
        or first.person_id != second.person_id
        or first.alternative_person_id != second.alternative_person_id
    )


def resolve_decisions(
    decisions: list[IdentityDecision],
) -> tuple[list[IdentityDecision], list[dict[str, Any]]]:
    active = sorted(
        (decision for decision in decisions if decision.status == "active"),
        key=lambda decision: (decision.created_at, decision.decision_id),
    )
    accepted: list[IdentityDecision] = []
    conflicts: list[dict[str, Any]] = []
    for decision in active:
        blocker = next(
            (other for other in accepted if decisions_conflict(decision, other)),
            None,
        )
        if blocker is None:
            accepted.append(decision)
            continue
        shared = decision_affected_markers(decision) & decision_affected_markers(blocker)
        conflicts.append(
            {
                "decision_id": decision.decision_id,
                "conflicts_with": blocker.decision_id,
                "start_frame": max(decision.start_frame, blocker.start_frame),
                "end_frame": min(decision.end_frame, blocker.end_frame),
                "markers": sorted(shared),
                "decision_trajectories": decision_trajectories(decision),
                "conflicts_with_trajectories": decision_trajectories(blocker),
                "reason": (
                    "overlapping manual decisions disagree; the earlier "
                    "decision is kept and this one is not applied"
                ),
            }
        )
    return accepted, conflicts


def decisions_to_revisions(
    decisions: list[IdentityDecision],
) -> list[IdentityRevision]:
    active, _conflicts = resolve_decisions(decisions)
    pairs: set[tuple[str, str, int, int]] = set()
    revisions: list[IdentityRevision] = []

    for decision in active:
        group, alternative_group = decision_joint_groups(decision)
        for joint_name, alternative_joint_name in zip(group, alternative_group):
            source = person_marker(decision.person_id, joint_name)
            target = person_marker(decision.alternative_person_id, alternative_joint_name)
            key = tuple(sorted((source, target)) + [decision.start_frame, decision.end_frame])
            if key in pairs:
                continue
            pairs.add(key)
            action = "confirmed" if decision.decision == "confirm" else "denied"
            reason = f"manual {action} identity propagated to related joint: {decision.reason}"
            revisions.append(
                IdentityRevision(
                    marker=source,
                    start_frame=decision.start_frame,
                    end_frame=decision.end_frame,
                    corrected_identity=target,
                    reason=reason,
                    created_at=decision.created_at,
                )
            )
            revisions.append(
                IdentityRevision(
                    marker=target,
                    start_frame=decision.start_frame,
                    end_frame=decision.end_frame,
                    corrected_identity=source,
                    reason=reason,
                    created_at=decision.created_at,
                )
            )
    return revisions


def decision_window(decisions: list[IdentityDecision], frame_count: int) -> tuple[int, int] | None:
    active = [decision for decision in decisions if decision.status == "active"]
    if not active:
        return None
    return (
        max(0, min(decision.start_frame for decision in active) - 2),
        min(frame_count - 1, max(decision.end_frame for decision in active) + 2),
    )


def cleared_markers_at(decisions: list[IdentityDecision], frame_index: int) -> set[str]:
    cleared: set[str] = set()
    for decision in decisions:
        if decision.start_frame <= frame_index <= decision.end_frame:
            cleared |= decision_affected_markers(decision)
    return cleared


def initial_identity_statuses(
    frames: list[dict[str, Any]],
    ambiguity_windows: list[dict[str, Any]],
    decisions: list[IdentityDecision] | None = None,
) -> list[dict[str, str]]:
    markers = sorted({marker for frame in frames for marker in frame})
    statuses = [{marker: "certain" for marker in markers} for _ in frames]
    decisions = decisions or []
    accepted, _conflicts = resolve_decisions(decisions)
    for window in ambiguity_windows:
        for frame_index in range(window["start_frame"], window["end_frame"] + 1):
            if not 0 <= frame_index < len(statuses):
                continue
            cleared = cleared_markers_at(accepted, frame_index)
            for marker in window.get("markers", []):
                if marker in cleared:
                    continue
                if marker in statuses[frame_index]:
                    statuses[frame_index][marker] = "ambiguous"
    confirmed = {
        frame_index: set()
        for frame_index in range(len(frames))
    }
    for decision in accepted:
        confirmed_markers = decision_affected_markers(decision)
        for frame_index in range(decision.start_frame, decision.end_frame + 1):
            if 0 <= frame_index < len(frames):
                confirmed[frame_index].update(confirmed_markers)
    for frame_index, frame_status in enumerate(statuses):
        for marker in confirmed[frame_index]:
            if marker in frame_status and frame_status[marker] != "ambiguous":
                frame_status[marker] = "confirmed"
    return statuses


def reassignment_joint_groups(reassignment: TrackReassignment) -> tuple[list[str], list[str]]:
    source_group = reassignment.related_markers or kinematic_group(reassignment.marker)
    target_group = reassignment.target_related_markers or source_group
    return source_group, target_group


def reassignment_affected_markers(reassignment: TrackReassignment) -> set[str]:
    source_group, target_group = reassignment_joint_groups(reassignment)
    return {
        person_marker(reassignment.source_person_id, joint) for joint in source_group
    } | {
        person_marker(reassignment.target_person_id, joint) for joint in target_group
    }


def reassignment_trajectories(reassignment: TrackReassignment) -> list[dict[str, Any]]:
    source_group, target_group = reassignment_joint_groups(reassignment)
    return [
        {
            "from_person": reassignment.source_person_id,
            "to_person": reassignment.target_person_id,
            "marker": joint,
            "start_frame": reassignment.start_frame,
            "end_frame": reassignment.end_frame,
        }
        for joint in sorted(source_group)
    ] + [
        {
            "from_person": reassignment.target_person_id,
            "to_person": reassignment.source_person_id,
            "marker": joint,
            "start_frame": reassignment.start_frame,
            "end_frame": reassignment.end_frame,
        }
        for joint in sorted(target_group)
    ]


def _overlaps(start: int, end: int, other_start: int, other_end: int) -> bool:
    return start <= other_end and other_start <= end


def _active_reassignments(
    reassignments: list[TrackReassignment],
) -> list[TrackReassignment]:
    return sorted(
        (item for item in reassignments if item.status == "active"),
        key=lambda item: (item.created_at, item.reassignment_id),
    )


def resolve_reassignments(
    reassignments: list[TrackReassignment],
) -> tuple[list[TrackReassignment], list[TrackReassignment], list[dict[str, Any]]]:
    active = _active_reassignments(reassignments)
    by_id = {item.reassignment_id: item for item in active}
    paired: list[TrackReassignment] = []
    disputed: list[TrackReassignment] = []
    blockers: list[dict[str, Any]] = []
    consumed: set[str] = set()

    for item in active:
        if item.reassignment_id in consumed:
            continue
        partner_id = item.paired_reassignment_id
        partner = by_id.get(partner_id) if partner_id else None
        if item.resolution == "disputed" and partner is not None:
            disputed.append(item)
            consumed.add(item.reassignment_id)
            blockers.append(
                {
                    "reassignment_id": item.reassignment_id,
                    "blocked_by": partner.reassignment_id,
                    "start_frame": max(item.start_frame, partner.start_frame),
                    "end_frame": min(item.end_frame, partner.end_frame),
                    "markers": sorted(
                        reassignment_affected_markers(item)
                        & reassignment_affected_markers(partner)
                    ),
                    "reason": "the same track segment already belongs to another person",
                }
            )
            continue
        if item.resolution == "paired_adjustment" and partner is not None:
            if partner.paired_reassignment_id == item.reassignment_id:
                paired.extend([item, partner])
                consumed.update({item.reassignment_id, partner.reassignment_id})
                continue
        if item.resolution == "disputed":
            disputed.append(item)
            consumed.add(item.reassignment_id)
            continue

        occupier = next(
            (
                other
                for other in active
                if other.reassignment_id != item.reassignment_id
                and other.resolution != "disputed"
                and _overlaps(
                    item.start_frame,
                    item.end_frame,
                    other.start_frame,
                    other.end_frame,
                )
                and reassignment_affected_markers(item)
                & reassignment_affected_markers(other)
            ),
            None,
        )
        if occupier is not None:
            disputed.append(item)
            blockers.append(
                {
                    "reassignment_id": item.reassignment_id,
                    "blocked_by": occupier.reassignment_id,
                    "start_frame": max(item.start_frame, occupier.start_frame),
                    "end_frame": min(item.end_frame, occupier.end_frame),
                    "markers": sorted(
                        reassignment_affected_markers(item)
                        & reassignment_affected_markers(occupier)
                    ),
                    "reason": "the same track segment already belongs to another person",
                }
            )
        else:
            paired.append(item)
        consumed.add(item.reassignment_id)

    return paired, disputed, blockers


def decisions_effective_for_reconstruction(
    decisions: list[IdentityDecision],
    reassignments: list[TrackReassignment],
) -> list[IdentityDecision]:
    accepted, _conflicts = resolve_decisions(decisions)
    blocked_ids: set[str] = set()
    for reassignment in reassignments:
        markers = reassignment_affected_markers(reassignment)
        for decision in accepted:
            if not _overlaps(
                reassignment.start_frame,
                reassignment.end_frame,
                decision.start_frame,
                decision.end_frame,
            ):
                continue
            if markers & decision_affected_markers(decision):
                blocked_ids.add(decision.decision_id)
    return [decision for decision in accepted if decision.decision_id not in blocked_ids]


def reassignments_to_revisions(
    reassignments: list[TrackReassignment],
) -> list[IdentityRevision]:
    paired, _disputed, _blockers = resolve_reassignments(reassignments)
    revisions: list[IdentityRevision] = []
    seen_pairs: set[tuple[str, str, int, int]] = set()

    for item in paired:
        source_group, target_group = reassignment_joint_groups(item)
        for source_joint, target_joint in zip(source_group, target_group):
            source = person_marker(item.source_person_id, source_joint)
            target = person_marker(item.target_person_id, target_joint)
            key = tuple(sorted((source, target)) + [item.start_frame, item.end_frame])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            reason = f"manual scoped track reassignment: {item.reason}"
            revisions.append(
                IdentityRevision(
                    marker=source,
                    start_frame=item.start_frame,
                    end_frame=item.end_frame,
                    corrected_identity=target,
                    reason=reason,
                    created_at=item.created_at,
                )
            )
            revisions.append(
                IdentityRevision(
                    marker=target,
                    start_frame=item.start_frame,
                    end_frame=item.end_frame,
                    corrected_identity=source,
                    reason=reason,
                    created_at=item.created_at,
                )
            )
    return revisions


def preview_reassignment(
    reassignment: TrackReassignment,
    reassignments: list[TrackReassignment],
    decisions: list[IdentityDecision],
    statuses: list[dict[str, str]],
) -> dict[str, Any]:
    active_decisions, _decision_conflicts = resolve_decisions(decisions)
    markers = reassignment_affected_markers(reassignment)
    source_group, target_group = reassignment_joint_groups(reassignment)
    preview_markers = (
        {
            person_marker(reassignment.source_person_id, joint)
            for joint in [*source_group, *target_group]
        }
        | {
            person_marker(reassignment.target_person_id, joint)
            for joint in [*source_group, *target_group]
        }
    )
    affected_decisions = []
    for decision in active_decisions:
        if not _overlaps(
            reassignment.start_frame,
            reassignment.end_frame,
            decision.start_frame,
            decision.end_frame,
        ):
            continue
        shared = markers & decision_affected_markers(decision)
        if not shared:
            continue
        affected_decisions.append(
            {
                "decision_id": decision.decision_id,
                "state": "pending_recheck",
                "start_frame": max(reassignment.start_frame, decision.start_frame),
                "end_frame": min(reassignment.end_frame, decision.end_frame),
                "markers": sorted(shared),
                "dependency": reassignment.reassignment_id,
            }
        )

    occlusion_markers = {}
    for frame_index in range(reassignment.start_frame, reassignment.end_frame + 1):
        if not 0 <= frame_index < len(statuses):
            continue
        for marker in preview_markers:
            state = statuses[frame_index].get(marker)
            if state == "recovered":
                occlusion_markers[marker] = "occluded_and_rebuilt"
            elif state in {"uncertain", "missing"} and marker not in occlusion_markers:
                occlusion_markers[marker] = "occluded_unresolved"
    recovered_markers = sorted(
        {
            marker
            for frame_index in range(reassignment.start_frame, reassignment.end_frame + 1)
            if 0 <= frame_index < len(statuses)
            for marker in preview_markers
            if statuses[frame_index].get(marker) == "recovered"
        }
    )
    _paired, disputed, blockers = resolve_reassignments([*reassignments, reassignment])
    unresolved = preview_markers
    unresolved_windows = _marker_windows(
        statuses,
        unresolved,
        reassignment.start_frame,
        reassignment.end_frame,
        {"uncertain", "missing"},
    )

    return {
        "reassignment_id": reassignment.reassignment_id,
        "resolution": "disputed" if any(
            item.reassignment_id == reassignment.reassignment_id for item in disputed
        ) else reassignment.resolution,
        "trajectories": reassignment_trajectories(reassignment),
        "related_changes": {
            "marker_confirmations": affected_decisions,
            "occlusion_judgments": [
                {"marker": marker, "state": state}
                for marker, state in sorted(occlusion_markers.items())
            ],
            "skeleton_correspondences": [
                {"source_marker": marker, "state": "rebuild_within_window"}
                for marker in sorted(markers)
            ],
            "recovered_markers": recovered_markers,
        },
        "occupancy_blockers": [
            blocker
            for blocker in blockers
            if blocker["reassignment_id"] == reassignment.reassignment_id
        ],
        "dependencies": [
            {
                "kind": "paired_reassignment",
                "requires": reassignment.paired_reassignment_id,
                "state": "required" if reassignment.paired_reassignment_id else "none",
            }
        ],
        "unresolved_intervals": unresolved_windows,
    }


def _marker_windows(
    statuses: list[dict[str, str]],
    markers: set[str],
    start_frame: int,
    end_frame: int,
    states: set[str],
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for marker in sorted(markers):
        start = None
        for frame_index in range(start_frame, min(end_frame, len(statuses) - 1) + 1):
            active = statuses[frame_index].get(marker) in states
            if active and start is None:
                start = frame_index
            elif not active and start is not None:
                windows.append(
                    {"marker": marker, "start_frame": start, "end_frame": frame_index - 1}
                )
                start = None
        if start is not None:
            windows.append({"marker": marker, "start_frame": start, "end_frame": end_frame})
    return windows


def apply_reassignment_identity_statuses(
    statuses: list[dict[str, str]],
    reassignments: list[TrackReassignment],
    decisions: list[IdentityDecision],
) -> list[dict[str, str]]:
    result = [dict(frame) for frame in statuses]
    paired, disputed, _blockers = resolve_reassignments(reassignments)
    accepted_decisions, _conflicts = resolve_decisions(decisions)

    dependent: dict[int, set[str]] = {index: set() for index in range(len(result))}
    for reassignment in reassignments:
        markers = reassignment_affected_markers(reassignment)
        for decision in accepted_decisions:
            if not _overlaps(
                reassignment.start_frame,
                reassignment.end_frame,
                decision.start_frame,
                decision.end_frame,
            ):
                continue
            shared = markers & decision_affected_markers(decision)
            for frame_index in range(
                max(reassignment.start_frame, decision.start_frame),
                min(reassignment.end_frame, decision.end_frame) + 1,
            ):
                if 0 <= frame_index < len(result):
                    dependent[frame_index].update(shared)

    for frame_index, markers in dependent.items():
        for marker in markers:
            if marker in result[frame_index]:
                result[frame_index][marker] = "pending"

    for reassignment in paired:
        markers = reassignment_affected_markers(reassignment)
        for frame_index in range(reassignment.start_frame, reassignment.end_frame + 1):
            if not 0 <= frame_index < len(result):
                continue
            for marker in markers:
                if marker in result[frame_index]:
                    if marker not in dependent[frame_index]:
                        result[frame_index][marker] = "reassigned"
    for reassignment in disputed:
        markers = reassignment_affected_markers(reassignment)
        for frame_index in range(reassignment.start_frame, reassignment.end_frame + 1):
            if not 0 <= frame_index < len(result):
                continue
            for marker in markers:
                if marker in result[frame_index]:
                    result[frame_index][marker] = "disputed"
    return result
