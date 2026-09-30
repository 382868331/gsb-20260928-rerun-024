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


def decisions_to_revisions(
    decisions: list[IdentityDecision],
) -> list[IdentityRevision]:
    active = [decision for decision in decisions if decision.status == "active"]
    pairs: set[tuple[str, str, int, int]] = set()
    revisions: list[IdentityRevision] = []

    for decision in active:
        group = decision.related_markers or kinematic_group(decision.marker)
        alternative_group = (
            decision.alternative_related_markers
            if decision.alternative_related_markers
            else group
        )
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


def ambiguity_markers(window: dict[str, Any], decisions: list[IdentityDecision]) -> set[str]:
    active = [
        decision
        for decision in decisions
        if decision.status == "active"
        and decision.start_frame <= window["end_frame"]
        and decision.end_frame >= window["start_frame"]
    ]
    if active:
        return set()
    return set(window.get("markers", []))


def initial_identity_statuses(
    frames: list[dict[str, Any]],
    ambiguity_windows: list[dict[str, Any]],
    decisions: list[IdentityDecision] | None = None,
) -> list[dict[str, str]]:
    markers = sorted({marker for frame in frames for marker in frame})
    statuses = [{marker: "certain" for marker in markers} for _ in frames]
    decisions = decisions or []
    for window in ambiguity_windows:
        affected = ambiguity_markers(window, decisions)
        for frame_index in range(window["start_frame"], window["end_frame"] + 1):
            for marker in affected:
                if marker in statuses[frame_index]:
                    statuses[frame_index][marker] = "ambiguous"
    confirmed = {
        frame_index: set()
        for frame_index in range(len(frames))
    }
    for decision in decisions:
        if decision.status != "active":
            continue
        confirmed_markers = {
            person_marker(decision.person_id, decision.marker),
        } | {
            person_marker(decision.person_id, joint)
            for joint in (
                decision.related_markers or kinematic_group(decision.marker)
            )
        } | {
            person_marker(decision.alternative_person_id, joint)
            for joint in (
                decision.alternative_related_markers
                or decision.related_markers
                or kinematic_group(decision.marker)
            )
        }
        for frame_index in range(decision.start_frame, decision.end_frame + 1):
            if 0 <= frame_index < len(frames):
                confirmed[frame_index].update(confirmed_markers)
    for frame_index, frame_status in enumerate(statuses):
        for marker in confirmed[frame_index]:
            if marker in frame_status and frame_status[marker] != "ambiguous":
                frame_status[marker] = "confirmed"
    return statuses
