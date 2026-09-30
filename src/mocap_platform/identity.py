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


def normalize_frames(start_frame: int, end_frame: int, frame_count: int | None = None) -> range:
    if end_frame < start_frame:
        raise ValueError("decision end_frame cannot be before start_frame")
    if frame_count is not None:
        start_frame = max(0, start_frame)
        end_frame = min(frame_count - 1, end_frame)
    return range(start_frame, end_frame + 1)


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


def decision_trajectories(decision: IdentityDecision) -> list[tuple[str, str]]:
    group = decision.related_markers or kinematic_group(decision.marker)
    alternative_group = decision.alternative_related_markers or group
    trajectories: list[tuple[str, str]] = []
    for joint_name, alternative_joint_name in zip(group, alternative_group):
        trajectories.append((person_marker(decision.person_id, joint_name), decision.person_id))
        trajectories.append(
            (
                person_marker(decision.alternative_person_id, alternative_joint_name),
                decision.alternative_person_id,
            )
        )
    return sorted(set(trajectories))


def merge_decisions(
    existing: list[IdentityDecision], updates: list[IdentityDecision]
) -> list[IdentityDecision]:
    merged: dict[str, IdentityDecision] = {
        decision.decision_id: decision for decision in existing
    }
    for update in updates:
        previous = merged.get(update.decision_id)
        if previous is not None and update.created_at == previous.created_at:
            merged[update.decision_id] = IdentityDecision.from_dict(
                {**previous.to_dict(), **update.to_dict(), "created_at": previous.created_at}
            )
        else:
            merged[update.decision_id] = update
    return list(merged.values())


def _claim_key(decision: IdentityDecision, marker: str, owner: str) -> tuple[str, str]:
    return owner, decision.decision


def conflicted_cells(
    decisions: list[IdentityDecision],
) -> dict[tuple[int, str], set[str]]:
    claims: dict[tuple[int, str], dict[str, str]] = {}
    for decision in decisions:
        if decision.status != "active":
            continue
        group = decision.related_markers or kinematic_group(decision.marker)
        alternative_group = decision.alternative_related_markers or group
        for frame_index in normalize_frames(decision.start_frame, decision.end_frame):
            for joint_name, alternative_joint_name in zip(group, alternative_group):
                source = person_marker(decision.person_id, joint_name)
                target = person_marker(decision.alternative_person_id, alternative_joint_name)
                if decision.decision == "confirm":
                    claims.setdefault((frame_index, source), {})[target] = decision.decision_id
                    claims.setdefault((frame_index, target), {})[source] = decision.decision_id
                else:
                    claims.setdefault((frame_index, source), {})[source] = decision.decision_id
                    claims.setdefault((frame_index, target), {})[target] = decision.decision_id
    return {
        cell: set(claim.values())
        for cell, claim in claims.items()
        if len(claim) > 1
    }


def active_conflicts(decisions: list[IdentityDecision]) -> set[str]:
    return {
        decision_id
        for ids in conflicted_cells([d for d in decisions if d.status == "active"]).values()
        for decision_id in ids
    }


def _frame_ranges(frames: list[int]) -> list[dict[str, int]]:
    if not frames:
        return []
    ordered = sorted(set(frames))
    ranges: list[dict[str, int]] = []
    start = previous = ordered[0]
    for frame_index in ordered[1:]:
        if frame_index == previous + 1:
            previous = frame_index
            continue
        ranges.append({"start_frame": start, "end_frame": previous})
        start = previous = frame_index
    ranges.append({"start_frame": start, "end_frame": previous})
    return ranges


def ambiguity_key(window: dict[str, Any], frame_index: int, marker: str) -> str:
    marker_text = ",".join(window.get("markers", []))
    return f"{window.get('id', f'{window["start_frame"]}-{window["end_frame"]}-{marker_text}')}:{frame_index}:{marker}"


def ambiguity_claims(
    ambiguity_windows: list[dict[str, Any]],
    decisions: list[IdentityDecision],
) -> dict[str, list[str]]:
    claims: dict[str, list[str]] = {}
    active = [decision for decision in decisions if decision.status == "active"]
    for decision in active:
        trajectories = {marker for marker, _owner in decision_trajectories(decision)}
        for window in ambiguity_windows:
            window_markers = set(window.get("markers", []))
            overlap_start = max(decision.start_frame, window["start_frame"])
            overlap_end = min(decision.end_frame, window["end_frame"])
            if overlap_start > overlap_end:
                continue
            for frame_index in range(overlap_start, overlap_end + 1):
                for marker in trajectories & window_markers:
                    claims.setdefault(ambiguity_key(window, frame_index, marker), []).append(
                        decision.decision_id
                    )
    return claims


def decision_effects(
    decisions: list[IdentityDecision],
    ambiguity_windows: list[dict[str, Any]],
    frame_count: int,
) -> list[dict[str, Any]]:
    conflicts = conflicted_cells([d for d in decisions if d.status == "active"])
    active_decisions = [decision for decision in decisions if decision.status == "active"]
    claims = ambiguity_claims(ambiguity_windows, active_decisions)
    resolved_by: dict[str, list[str]] = {}
    for key, decision_ids in claims.items():
        unique_ids = set(decision_ids)
        parts = key.split(":", 2)
        frame_index = int(parts[1])
        marker = parts[2]
        if unique_ids and not (unique_ids & conflicts.get((frame_index, marker), set())):
            resolved_by[key] = sorted(unique_ids)

    effects: list[dict[str, Any]] = []
    for decision in decisions:
        trajectories = [
            {"marker": marker, "owner_person_id": owner}
            for marker, owner in decision_trajectories(decision)
        ]
        frames = list(normalize_frames(decision.start_frame, decision.end_frame, frame_count))
        trajectory_set = {marker for marker, _owner in decision_trajectories(decision)}
        conflict_frames = [
            frame_index
            for frame_index in frames
            if any(
                decision.decision_id in conflicts.get((frame_index, marker), set())
                for marker in trajectory_set
            )
        ]
        resolved = sorted(
            key
            for key, decision_ids in resolved_by.items()
            if decision.decision_id in decision_ids
        )
        reopened: list[str] = []
        if decision.status == "revoked":
            active_after = [item for item in decisions if item is not decision and item.status == "active"]
            active_claims = ambiguity_claims(ambiguity_windows, active_after)
            active_conflict_cells = conflicted_cells(active_after)
            active_resolved = {
                key
                for key, ids in active_claims.items()
                if not (
                    set(ids)
                    & active_conflict_cells.get(
                        (int(key.split(":", 2)[1]), key.split(":", 2)[2]), set()
                    )
                )
            }
            trajectories_set = {item["marker"] for item in trajectories}
            for window in ambiguity_windows:
                for frame_index in frames:
                    if not window["start_frame"] <= frame_index <= window["end_frame"]:
                        continue
                    for marker in trajectories_set & set(window.get("markers", [])):
                        key = ambiguity_key(window, frame_index, marker)
                        if key not in active_resolved:
                            reopened.append(key)

        effects.append(
            {
                "decision_id": decision.decision_id,
                "status": decision.status,
                "decision": decision.decision,
                "affected_trajectories": trajectories,
                "frame_ranges": _frame_ranges(frames),
                "resolved_ambiguity_keys": resolved,
                "reopened_ambiguity_keys": sorted(set(reopened)),
                "conflicted": bool(conflict_frames) and decision.status == "active",
                "conflicted_frame_ranges": _frame_ranges(conflict_frames),
                "conflicting_with": sorted(
                    other_id
                    for frame_index in frames
                    for marker in trajectory_set
                    for other_id in conflicts.get((frame_index, marker), set())
                    if other_id != decision.decision_id
                ),
            }
        )
    return effects


def decisions_to_revisions(
    decisions: list[IdentityDecision],
) -> list[IdentityRevision]:
    conflicted = conflicted_cells([d for d in decisions if d.status == "active"])
    active = [
        decision
        for decision in decisions
        if decision.status == "active"
    ]
    pairs: set[tuple[str, str, int, int]] = set()
    revisions: list[IdentityRevision] = []

    for decision in active:
        group = decision.related_markers or kinematic_group(decision.marker)
        alternative_group = (
            decision.alternative_related_markers
            if decision.alternative_related_markers
            else group
        )
        if decision.decision == "deny":
            continue
        for joint_name, alternative_joint_name in zip(group, alternative_group):
            for frame_index in normalize_frames(decision.start_frame, decision.end_frame):
                source = person_marker(decision.person_id, joint_name)
                target = person_marker(decision.alternative_person_id, alternative_joint_name)
                if decision.decision_id in conflicted.get((frame_index, source), set()) or decision.decision_id in conflicted.get((frame_index, target), set()):
                    continue
                key = tuple(sorted((source, target)) + [frame_index])
                if key in pairs:
                    continue
                pairs.add(key)
                reason = f"manual confirmed identity propagated to related joint: {decision.reason}"
                revisions.append(
                    IdentityRevision(
                        marker=source,
                        start_frame=frame_index,
                        end_frame=frame_index,
                        corrected_identity=target,
                        reason=reason,
                        created_at=decision.created_at,
                    )
                )
                revisions.append(
                    IdentityRevision(
                        marker=target,
                        start_frame=frame_index,
                        end_frame=frame_index,
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
        affected = set(window.get("markers", []))
        for frame_index in range(window["start_frame"], window["end_frame"] + 1):
            for marker in affected:
                if marker in statuses[frame_index]:
                    statuses[frame_index][marker] = "ambiguous"
    resolved = {
        frame_index: set()
        for frame_index in range(len(frames))
    }
    conflicted = conflicted_cells([d for d in decisions if d.status == "active"])
    for decision in decisions:
        if decision.status != "active":
            continue
        resolved_markers = {marker for marker, _owner in decision_trajectories(decision)}
        for frame_index in range(decision.start_frame, decision.end_frame + 1):
            if 0 <= frame_index < len(frames):
                resolved[frame_index].update(resolved_markers)
    for decision in decisions:
        if decision.status != "active":
            continue
        for frame_index in range(decision.start_frame, decision.end_frame + 1):
            if not 0 <= frame_index < len(frames):
                continue
            for marker, _owner in decision_trajectories(decision):
                if marker not in statuses[frame_index]:
                    continue
                if decision.decision_id in conflicted.get((frame_index, marker), set()):
                    statuses[frame_index][marker] = "conflicted"
                elif decision.decision == "confirm":
                    statuses[frame_index][marker] = "confirmed"
                else:
                    statuses[frame_index][marker] = "denied"
    return statuses
