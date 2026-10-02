from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from .contact import (
    candidates_dependent_on,
    disputed_contact_intervals,
    evaluate_contact_compatibility,
    object_missing_intervals,
    schemes_dependent_on,
    unresolved_pose_intervals,
)
from .geometry import angle_between, cross, distance, dot, lerp, norm, normalize
from .identity import (
    decision_affected_markers,
    decision_trajectories,
    decisions_to_revisions,
    initial_identity_statuses,
    resolve_decisions,
)
from .identity import split_marker
from .models import Frame, IdentityRevision, MarkerStatus, ProcessingVersion
from .reassignment import (
    pending_confirmations,
    reassignment_impact_preview,
    reassignment_markers,
    reassignment_trajectories,
    reassignments_to_revisions,
    resolve_reassignments,
)
from .skeleton import BONES, CHAIN_ARMS_LEGS, PARENTS


def _bone_parts(name: str) -> tuple[str, str]:
    start, end = name.split("->")
    return start, end


def remap_identities(
    frames: list[Frame], revisions: list[IdentityRevision]
) -> tuple[list[Frame], list[bool]]:
    corrected: list[Frame] = [dict(frame) for frame in frames]
    changed = [False] * len(frames)
    for frame_index, original in enumerate(frames):
        swaps: dict[str, str] = {}
        for revision in revisions:
            if not revision.start_frame <= frame_index <= revision.end_frame:
                continue
            swaps[revision.marker] = revision.corrected_identity
        handled: set[str] = set()
        for source, target in swaps.items():
            pair = tuple(sorted((source, target)))
            if pair in handled:
                continue
            handled.add(pair)
            reciprocal = swaps.get(target)
            if reciprocal == source:
                corrected[frame_index][target] = original.get(source)
                corrected[frame_index][source] = original.get(target)
            else:
                corrected[frame_index][target] = original.get(source)
            changed[frame_index] = True
    return corrected, changed


def interpolate_gaps(
    observations: list[Frame], max_gap_frames: int
) -> tuple[list[Frame], list[dict[str, MarkerStatus]]]:
    markers = sorted({marker for frame in observations for marker in frame})
    frames: list[Frame] = [dict(frame) for frame in observations]
    statuses: list[dict[str, MarkerStatus]] = [
        {marker: "missing" for marker in markers} for _ in observations
    ]

    for marker in markers:
        index = 0
        while index < len(frames):
            if observations[index].get(marker) is not None:
                frames[index][marker] = observations[index][marker]
                statuses[index][marker] = "observed"
                index += 1
                continue

            start = index
            while index < len(frames) and observations[index].get(marker) is None:
                index += 1
            end = index
            before = start - 1
            after = end if end < len(frames) else None
            gap_length = end - start

            if before >= 0 and after is not None and gap_length <= max_gap_frames:
                for missing_index in range(start, end):
                    t = (missing_index - before) / (after - before)
                    frames[missing_index][marker] = lerp(
                        observations[before][marker], observations[after][marker], t
                    )
                    statuses[missing_index][marker] = "recovered"
            else:
                for missing_index in range(start, end):
                    frames[missing_index][marker] = None
                    statuses[missing_index][marker] = "uncertain"

    return frames, statuses


def _set_vector_length(point: list[float], anchor: list[float], length: float) -> list[float]:
    direction = normalize([point[i] - anchor[i] for i in range(3)])
    return [anchor[i] + direction[i] * length for i in range(3)]


def _place_between(
    candidate: list[float],
    first: list[float],
    second: list[float],
    first_distance: float,
    second_distance: float,
) -> list[float]:
    axis = [second[i] - first[i] for i in range(3)]
    anchor_distance = norm(axis)
    if anchor_distance < 1e-12:
        return _set_vector_length(candidate, first, first_distance)
    unit = [value / anchor_distance for value in axis]
    if anchor_distance >= first_distance + second_distance:
        base_distance = first_distance
        height = 0.0
    elif anchor_distance <= abs(first_distance - second_distance):
        base_distance = min(first_distance, anchor_distance)
        height = 0.0
    else:
        base_distance = (
            first_distance * first_distance
            - second_distance * second_distance
            + anchor_distance * anchor_distance
        ) / (2 * anchor_distance)
        height = math.sqrt(max(0.0, first_distance * first_distance - base_distance * base_distance))
    base = [first[i] + unit[i] * base_distance for i in range(3)]
    if height == 0.0:
        return base
    to_candidate = [candidate[i] - base[i] for i in range(3)]
    perpendicular = [
        to_candidate[i] - unit[i] * dot(to_candidate, unit) for i in range(3)
    ]
    if norm(perpendicular) < 1e-12:
        fallback = normalize(cross(unit, [0.0, 0.0, 1.0]))
        if norm(fallback) < 1e-12:
            fallback = normalize(cross(unit, [0.0, 1.0, 0.0]))
        perpendicular = fallback
    perpendicular = normalize(perpendicular)
    return [base[i] + perpendicular[i] * height for i in range(3)]


def _enforce_bone_lengths(
    frames: list[Frame],
    statuses: list[dict[str, MarkerStatus]],
    bone_lengths: dict[str, dict[str, float]],
) -> None:
    for _ in range(4):
        for bone_name, constraint in bone_lengths.items():
            start_joint, end_joint = _bone_parts(bone_name)
            target = constraint["length"]
            for frame_index, frame in enumerate(frames):
                people = {None}
                for marker in frame:
                    people.add(split_marker(marker)[0])
                for person_id in people:
                    prefix = "" if person_id is None else f"{person_id}_"
                    start_name = f"{prefix}{start_joint}"
                    end_name = f"{prefix}{end_joint}"
                    start = frame.get(start_name)
                    end = frame.get(end_name)
                    if start is None or end is None:
                        continue
                    actual = distance(start, end)
                    start_observed = statuses[frame_index].get(start_name) == "observed"
                    end_observed = statuses[frame_index].get(end_name) == "observed"
                    if start_observed and end_observed:
                        continue
                    if actual < 1e-12:
                        continue
                    if start_observed:
                        frame[end_name] = _set_vector_length(end, start, target)
                    elif end_observed:
                        parent_name = PARENTS.get(start_joint)
                        parent = frame.get(f"{prefix}{parent_name}") if parent_name else None
                        parent_is_observed = (
                            parent is not None
                            and statuses[frame_index].get(f"{prefix}{parent_name}") == "observed"
                        )
                        previous_bone = (
                            bone_lengths.get(f"{parent_name}->{start_joint}") if parent_name else None
                        )
                        if parent_is_observed and previous_bone:
                            frame[start_name] = _place_between(
                                start,
                                parent,
                                end,
                                previous_bone["length"],
                                target,
                            )
                        else:
                            frame[start_name] = _set_vector_length(start, end, target)
                    else:
                        frame[end_name] = _set_vector_length(end, start, target)


def _set_angle(
    parent: list[float], joint: list[float], child: list[float], angle: float
) -> None:
    incoming = [joint[i] - parent[i] for i in range(3)]
    outgoing = [child[i] - joint[i] for i in range(3)]
    axis = normalize(incoming)
    length = norm(outgoing)
    direction = normalize(outgoing)
    projected = [direction[i] - axis[i] * dot(direction, axis) for i in range(3)]
    if norm(projected) < 1e-9:
        projected = normalize(cross(axis, [0.0, 0.0, 1.0]))
        if norm(projected) < 1e-9:
            projected = normalize(cross(axis, [0.0, 1.0, 0.0]))
    projected = normalize(projected)
    new_direction = [
        axis[i] * math.cos(angle) + projected[i] * math.sin(angle) for i in range(3)
    ]
    for i in range(3):
        child[i] = joint[i] + new_direction[i] * length


def _enforce_joint_limits(
    frames: list[Frame],
    statuses: list[dict[str, MarkerStatus]],
    joint_limits: dict[str, dict[str, float]],
) -> None:
    for frame_index, frame in enumerate(frames):
        for joint_name, limits in joint_limits.items():
            parent_joint, child_joint = CHAIN_ARMS_LEGS[joint_name]
            people = {None}
            for marker in frame:
                people.add(split_marker(marker)[0])
            for person_id in people:
                prefix = "" if person_id is None else f"{person_id}_"
                resolved_joint = f"{prefix}{joint_name}"
                parent_name = f"{prefix}{parent_joint}"
                child_name = f"{prefix}{child_joint}"
                parent = frame.get(parent_name)
                joint = frame.get(resolved_joint)
                child = frame.get(child_name)
                if not parent or not joint or not child:
                    continue
                incoming = [joint[i] - parent[i] for i in range(3)]
                outgoing = [child[i] - joint[i] for i in range(3)]
                if norm(incoming) < 1e-9 or norm(outgoing) < 1e-9:
                    continue
                current = angle_between(incoming, outgoing)
                target_angle = None
                if current < limits["min_angle"]:
                    target_angle = limits["min_angle"]
                elif current > limits["max_angle"]:
                    target_angle = limits["max_angle"]
                if target_angle is None:
                    continue
                if statuses[frame_index].get(child_name) == "recovered":
                    _set_angle(parent, joint, child, target_angle)
                elif statuses[frame_index].get(resolved_joint) == "recovered":
                    frame[resolved_joint] = _place_between(
                        joint, parent, child, norm(incoming), norm(outgoing)
                    )


def apply_skeleton_constraints(
    frames: list[Frame],
    statuses: list[dict[str, MarkerStatus]],
    calibration: dict[str, Any],
) -> tuple[list[Frame], list[dict[str, MarkerStatus]]]:
    constrained = [dict(frame) for frame in frames]
    for _ in range(3):
        _enforce_bone_lengths(constrained, statuses, calibration["bone_lengths"])
        _enforce_joint_limits(constrained, statuses, calibration["joint_limits"])

    for frame_index, frame in enumerate(constrained):
        for marker, value in list(frame.items()):
            if value is None:
                continue
            if statuses[frame_index].get(marker) != "recovered":
                continue
            violations = constraint_violations(frame, calibration, marker)
            if violations > 0:
                frame[marker] = None
                statuses[frame_index][marker] = "uncertain"
    return constrained, statuses


def constraint_violations(frame: Frame, calibration: dict[str, Any], marker: str) -> int:
    violations = 0
    person_id, _joint_name = split_marker(marker)
    prefix = "" if person_id is None else f"{person_id}_"
    for bone_name, constraint in calibration["bone_lengths"].items():
        start_name, end_name = _bone_parts(bone_name)
        start_name = f"{prefix}{start_name}"
        end_name = f"{prefix}{end_name}"
        if marker not in (start_name, end_name):
            continue
        start, end = frame.get(start_name), frame.get(end_name)
        if start is None or end is None:
            continue
        actual = distance(start, end)
        if actual < constraint["length"] - constraint["tolerance"] or actual > (
            constraint["length"] + constraint["tolerance"]
        ):
            violations += 1
    return violations


def _review_window(
    decisions: list[Any],
    revisions: list[IdentityRevision],
    reassignments: list[Any],
    frame_count: int,
) -> tuple[int, int] | None:
    ranges = [(decision.start_frame, decision.end_frame) for decision in decisions]
    ranges += [
        (reassignment.start_frame, reassignment.end_frame)
        for reassignment in reassignments
    ]
    ranges += [(revision.start_frame, revision.end_frame) for revision in revisions]
    if not ranges:
        return None
    start = max(0, min(start for start, _end in ranges) - 2)
    end = min(frame_count - 1, max(end for _start, end in ranges) + 2)
    return start, end


def _decision_impacts(
    decisions: list[Any],
    conflicts: list[dict[str, Any]],
    pending_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    conflicted = {conflict["decision_id"] for conflict in conflicts}
    pending_ids = pending_ids or set()
    impacts = []
    for decision in decisions:
        if decision.status == "revoked":
            state = "revoked"
        elif decision.decision_id in pending_ids:
            state = "pending"
        elif decision.decision_id in conflicted:
            state = "conflict"
        else:
            state = "applied"
        impacts.append(
            {
                "decision_id": decision.decision_id,
                "decision": decision.decision,
                "status": state,
                "trajectories": decision_trajectories(decision),
            }
        )
    return impacts


def _reassignment_impacts(
    reassignments: list[Any],
    decisions: list[Any],
    ambiguity_windows: list[dict[str, Any]],
    disputes: list[dict[str, Any]],
    disputed_ids: set[str],
) -> list[dict[str, Any]]:
    impacts = []
    for reassignment in reassignments:
        if reassignment.status == "revoked":
            state = "revoked"
        elif reassignment.reassignment_id in disputed_ids:
            state = "disputed"
        else:
            state = reassignment.status
        impacts.append(
            {
                "reassignment_id": reassignment.reassignment_id,
                "status": state,
                "source_person_id": reassignment.source_person_id,
                "target_person_id": reassignment.target_person_id,
                "start_frame": reassignment.start_frame,
                "end_frame": reassignment.end_frame,
                "trajectories": reassignment_trajectories(reassignment),
                "kept_decision_ids": reassignment.kept_decision_ids,
                "depends_on": reassignment.depends_on,
                "preview": reassignment_impact_preview(
                    reassignment, decisions, ambiguity_windows
                ),
            }
        )
    return impacts


def _reassignment_dependencies(
    reassignments: list[Any], decisions: list[Any]
) -> list[dict[str, Any]]:
    return [
        {
            "reassignment_id": reassignment.reassignment_id,
            "depends_on": reassignment.depends_on,
            "depended_by": sorted(
                decision.decision_id
                for decision in decisions
                if reassignment.reassignment_id in decision.depends_on
            ),
        }
        for reassignment in reassignments
    ]


def _bone_length_deviation(
    frames: list[Frame],
    bone_lengths: dict[str, dict[str, float]],
    start_frame: int,
    end_frame: int,
) -> float | None:
    deviations: list[float] = []
    for frame_index in range(start_frame, end_frame + 1):
        if not 0 <= frame_index < len(frames):
            continue
        frame = frames[frame_index]
        people = {split_marker(marker)[0] for marker in frame}
        for bone_name, constraint in bone_lengths.items():
            start_joint, end_joint = _bone_parts(bone_name)
            for person_id in people:
                prefix = "" if person_id is None else f"{person_id}_"
                start = frame.get(f"{prefix}{start_joint}")
                end = frame.get(f"{prefix}{end_joint}")
                if start is None or end is None:
                    continue
                deviations.append(abs(distance(start, end) - constraint["length"]))
    if not deviations:
        return None
    return sum(deviations) / len(deviations)


def _unresolved_intervals(
    statuses: list[dict[str, MarkerStatus]],
    identity_statuses: list[dict[str, str]],
) -> list[dict[str, int]]:
    intervals: list[dict[str, int]] = []
    start: int | None = None
    for frame_index, frame_statuses in enumerate(statuses):
        unresolved = any(status == "uncertain" for status in frame_statuses.values())
        if not unresolved and frame_index < len(identity_statuses):
            unresolved = any(
                status == "ambiguous"
                for status in identity_statuses[frame_index].values()
            )
        if unresolved and start is None:
            start = frame_index
        elif not unresolved and start is not None:
            intervals.append({"start_frame": start, "end_frame": frame_index - 1})
            start = None
    if start is not None:
        intervals.append({"start_frame": start, "end_frame": len(statuses) - 1})
    return intervals


def _pose_comparisons(
    reassignments: list[Any],
    based_on: ProcessingVersion,
    frames: list[Frame],
    statuses: list[dict[str, MarkerStatus]],
    identity_statuses: list[dict[str, str]],
    calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    comparisons = []
    for reassignment in reassignments:
        if reassignment.status != "applied":
            continue
        start, end = reassignment.start_frame, reassignment.end_frame
        comparisons.append(
            {
                "reassignment_id": reassignment.reassignment_id,
                "start_frame": start,
                "end_frame": end,
                "bone_deviation_before": _bone_length_deviation(
                    based_on.frames, calibration["bone_lengths"], start, end
                ),
                "bone_deviation_after": _bone_length_deviation(
                    frames, calibration["bone_lengths"], start, end
                ),
                "unresolved_intervals_before": _unresolved_intervals(
                    based_on.statuses, based_on.identity_statuses
                ),
                "unresolved_intervals_after": _unresolved_intervals(
                    statuses, identity_statuses
                ),
            }
        )
    return comparisons


def _decision_windows(decisions: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "decision_id": decision.decision_id,
            "start_frame": decision.start_frame,
            "end_frame": decision.end_frame,
            "markers": sorted(decision_affected_markers(decision)),
        }
        for decision in decisions
        if decision.status == "active"
    ]


def _restored_ambiguities(
    previous: list[dict[str, str]], current: list[dict[str, str]]
) -> list[dict[str, Any]]:
    if not previous:
        return []
    markers = sorted({marker for frame in current for marker in frame})
    restored: list[dict[str, Any]] = []
    for marker in markers:
        start: int | None = None
        for frame_index in range(len(current)):
            before = previous[frame_index].get(marker) if frame_index < len(previous) else None
            is_restored = (
                current[frame_index].get(marker) == "ambiguous"
                and before is not None
                and before != "ambiguous"
            )
            if is_restored and start is None:
                start = frame_index
            elif not is_restored and start is not None:
                restored.append(
                    {"marker": marker, "start_frame": start, "end_frame": frame_index - 1}
                )
                start = None
        if start is not None:
            restored.append(
                {"marker": marker, "start_frame": start, "end_frame": len(current) - 1}
            )
    return restored


def reconstruct(
    action_name: str,
    pipeline_name: str,
    capture_id: str,
    observations: list[Frame],
    calibration: dict[str, Any],
    *,
    max_gap_frames: int = 5,
    revisions: list[IdentityRevision] | None = None,
    decisions: list[Any] | None = None,
    reassignments: list[Any] | None = None,
    contact_observations: list[Any] | None = None,
    identity_candidates: list[Any] | None = None,
    contact_objects: dict[str, list[list[float] | None]] | None = None,
    identity_schemes: list[Any] | None = None,
    ambiguity_windows: list[dict[str, Any]] | None = None,
    based_on: ProcessingVersion | None = None,
) -> ProcessingVersion:
    decisions = decisions or []
    reassignments = reassignments or []
    contact_observations = contact_observations or []
    identity_candidates = identity_candidates or []
    contact_objects = contact_objects or {}
    identity_schemes = identity_schemes or []
    applied_reassignments, reassignment_disputes, disputed_ids = resolve_reassignments(
        reassignments
    )
    pending = pending_confirmations(decisions, reassignments, applied_reassignments)
    pending_ids = {item["decision_id"] for item in pending}
    effective_decisions = [
        decision
        for decision in decisions
        if decision.decision_id not in pending_ids
    ]
    if revisions is None:
        revisions = decisions_to_revisions(
            effective_decisions
        ) + reassignments_to_revisions(applied_reassignments)
    ambiguity_windows = ambiguity_windows or []
    _accepted, conflicts = resolve_decisions(effective_decisions)
    corrected, changed = remap_identities(observations, revisions)
    confirmed_ranges = [
        (reassignment_markers(reassignment), reassignment.start_frame, reassignment.end_frame)
        for reassignment in applied_reassignments
    ]
    identity_statuses = initial_identity_statuses(
        observations, ambiguity_windows, effective_decisions, confirmed_ranges
    )
    decision_parameters = {
        "decision_impacts": _decision_impacts(decisions, conflicts, pending_ids),
        "decision_conflicts": conflicts,
        "decision_windows": _decision_windows(effective_decisions),
        "pending_confirmations": pending,
        "reassignment_impacts": _reassignment_impacts(
            reassignments, decisions, ambiguity_windows, reassignment_disputes, disputed_ids
        ),
        "reassignment_disputes": reassignment_disputes,
        "reassignment_dependencies": _reassignment_dependencies(
            reassignments, decisions
        ),
    }

    active_contacts = [
        contact
        for contact in contact_observations
        if getattr(contact, "status", "active") == "active"
    ]
    revoked_contact_ids = {
        contact.contact_id
        for contact in contact_observations
        if getattr(contact, "status", "active") == "revoked"
    }
    contact_review = evaluate_contact_compatibility(
        active_contacts, identity_candidates, observations, contact_objects
    )
    contact_intervals = disputed_contact_intervals(contact_review)
    dependent_candidate_ids = candidates_dependent_on(
        identity_candidates,
        revoked_contact_ids,
        {
            reassignment.reassignment_id
            for reassignment in reassignments
            if reassignment.status == "revoked"
        },
        {
            decision.decision_id
            for decision in decisions
            if decision.status == "revoked"
        },
    )
    dependent_scheme_ids = schemes_dependent_on(
        identity_schemes, set(dependent_candidate_ids)
    )
    def _scheme_review(scheme: Any) -> dict[str, Any]:
        scheme_candidates = [
            replace(candidate, decision="open")
            for candidate in identity_candidates
            if candidate.candidate_id in set(scheme.candidate_ids)
        ]
        scheme_contact_review = evaluate_contact_compatibility(
            active_contacts, scheme_candidates, observations, contact_objects
        )
        scheme_contact_intervals = disputed_contact_intervals(scheme_contact_review)
        return {
            "scheme_id": scheme.scheme_id,
            "label": scheme.label,
            "status": scheme.status,
            "crossing_start_frame": scheme.crossing_start_frame,
            "crossing_end_frame": scheme.crossing_end_frame,
            "candidate_ids": list(scheme.candidate_ids),
            "disputed_pose_intervals": unresolved_pose_intervals(
                identity_statuses,
                scheme.crossing_start_frame,
                scheme.crossing_end_frame,
            ),
            "disputed_contact_intervals": [
                interval
                for interval in scheme_contact_intervals["contact_conflict_intervals"]
                if interval["end_frame"] >= scheme.crossing_start_frame
                and interval["start_frame"] <= scheme.crossing_end_frame
            ],
            "unobserved_contact_intervals": [
                interval
                for interval in scheme_contact_intervals["unobserved_contact_intervals"]
                if interval["end_frame"] >= scheme.crossing_start_frame
                and interval["start_frame"] <= scheme.crossing_end_frame
            ],
            "claim_conflict_intervals": scheme_contact_intervals[
                "identity_claim_conflict_intervals"
            ],
            "contact_review": scheme_contact_review,
            "needs_recheck": scheme.scheme_id in set(dependent_scheme_ids),
        }

    scheme_reviews = [_scheme_review(scheme) for scheme in identity_schemes]
    contact_parameters = {
        "contact_review": contact_review,
        "object_missing_intervals": object_missing_intervals(contact_objects),
        "disputed_contact_intervals": contact_intervals,
        "contact_dependent_candidates": dependent_candidate_ids,
        "contact_dependent_schemes": dependent_scheme_ids,
        "scheme_reviews": scheme_reviews,
    }

    contact_ranges = [
        (contact.start_frame, contact.end_frame) for contact in active_contacts
    ] + [
        (scheme.crossing_start_frame, scheme.crossing_end_frame)
        for scheme in identity_schemes
    ]
    if contact_ranges:
        contact_window = (
            max(0, min(start for start, _end in contact_ranges) - 2),
            min(
                len(observations) - 1,
                max(end for _start, end in contact_ranges) + 2,
            ),
        )
    else:
        contact_window = None
    review_window = _review_window(decisions, revisions, reassignments, len(observations))
    if review_window is None:
        review_window = contact_window
    elif contact_window is not None:
        review_window = (
            min(review_window[0], contact_window[0]),
            max(review_window[1], contact_window[1]),
        )
    if based_on is not None and review_window is not None:
        start, end = review_window
        frames = [dict(frame) for frame in based_on.frames]
        statuses = [dict(status) for status in based_on.statuses]
        window_observations = corrected[start : end + 1]
        local_frames, local_statuses = interpolate_gaps(window_observations, max_gap_frames)
        local_frames, local_statuses = apply_skeleton_constraints(
            local_frames, local_statuses, calibration
        )
        for offset, frame in enumerate(local_frames):
            frames[start + offset] = frame
            statuses[start + offset] = local_statuses[offset]
        for frame_index in range(start, end + 1):
            previous_statuses = (
                based_on.identity_statuses[frame_index]
                if based_on.identity_statuses
                else {}
            )
            identity_statuses[frame_index] = {
                marker: identity_statuses[frame_index].get(marker, previous)
                for marker, previous in previous_statuses.items()
            }
        parameters = {
            "max_gap_frames": max_gap_frames,
            "recompute_window": {"start_frame": start, "end_frame": end},
            "changed_frames": [i for i, value in enumerate(changed) if value],
            "ambiguity_windows": ambiguity_windows,
            "restored_ambiguities": _restored_ambiguities(
                based_on.identity_statuses, identity_statuses
            ),
            "pose_comparison": _pose_comparisons(
                applied_reassignments,
                based_on,
                frames,
                statuses,
                identity_statuses,
                calibration,
            ),
            **decision_parameters,
            **contact_parameters,
        }
        version = _next_version(based_on.version)
    else:
        frames, statuses = interpolate_gaps(corrected, max_gap_frames)
        frames, statuses = apply_skeleton_constraints(frames, statuses, calibration)
        parameters = {
            "max_gap_frames": max_gap_frames,
            "recompute_window": None,
            "ambiguity_windows": ambiguity_windows,
            "restored_ambiguities": [],
            "pose_comparison": [],
            **decision_parameters,
            **contact_parameters,
        }
        version = "v001"

    return ProcessingVersion(
        action_name=action_name,
        pipeline_name=pipeline_name,
        version=version,
        capture_id=capture_id,
        calibration_version=calibration.get("version", "v001"),
        parameters=parameters,
        frames=frames,
        statuses=statuses,
        revisions=revisions,
        decisions=decisions,
        reassignments=reassignments,
        contact_observations=contact_observations,
        identity_schemes=identity_schemes,
        identity_statuses=identity_statuses,
        based_on=based_on.version if based_on else None,
    )


def _next_version(version: str) -> str:
    return f"v{int(version.removeprefix('v')) + 1:03d}"
