from __future__ import annotations

import re
from typing import Any

from .models import IdentityDecision, IdentityRevision


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
