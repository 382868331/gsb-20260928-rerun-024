from __future__ import annotations

import math
from typing import Any

from .geometry import angle_between, cross, distance, dot, lerp, norm, normalize
from .models import Frame, IdentityRevision, MarkerStatus, ProcessingVersion
from .skeleton import BONES, CHAIN_ARMS_LEGS, PARENTS


def _bone_parts(name: str) -> tuple[str, str]:
    start, end = name.split("->")
    return start, end


def remap_identities(
    frames: list[Frame], revisions: list[IdentityRevision]
) -> tuple[list[Frame], list[bool]]:
    corrected: list[Frame] = [dict(frame) for frame in frames]
    changed = [False] * len(frames)
    for revision in revisions:
        start = max(0, revision.start_frame)
        end = min(len(frames) - 1, revision.end_frame)
        for frame_index in range(start, end + 1):
            source = revision.marker
            target = revision.corrected_identity
            value = frames[frame_index].get(source)
            previous_target = frames[frame_index].get(target)
            corrected[frame_index][target] = value
            corrected[frame_index][source] = previous_target
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
                start = frame.get(start_joint)
                end = frame.get(end_joint)
                if start is None or end is None:
                    continue
                actual = distance(start, end)
                start_observed = statuses[frame_index].get(start_joint) == "observed"
                end_observed = statuses[frame_index].get(end_joint) == "observed"
                if start_observed and end_observed:
                    continue
                if actual < 1e-12:
                    continue
                if start_observed:
                    frame[end_joint] = _set_vector_length(end, start, target)
                elif end_observed:
                    parent_name = PARENTS.get(start_joint)
                    parent = frame.get(parent_name) if parent_name else None
                    parent_is_observed = (
                        parent is not None
                        and statuses[frame_index].get(parent_name) == "observed"
                    )
                    previous_bone = bone_lengths.get(f"{parent_name}->{start_joint}") if parent_name else None
                    if parent_is_observed and previous_bone:
                        frame[start_joint] = _place_between(
                            start,
                            parent,
                            end,
                            previous_bone["length"],
                            target,
                        )
                    else:
                        frame[start_joint] = _set_vector_length(start, end, target)
                else:
                    frame[end_joint] = _set_vector_length(end, start, target)


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
            parent_name, child_name = CHAIN_ARMS_LEGS[joint_name]
            parent = frame.get(parent_name)
            joint = frame.get(joint_name)
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
            elif statuses[frame_index].get(joint_name) == "recovered":
                frame[joint_name] = _place_between(joint, parent, child, norm(incoming), norm(outgoing))


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
    for bone_name, constraint in calibration["bone_lengths"].items():
        start_name, end_name = _bone_parts(bone_name)
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


def _affected_window(revisions: list[IdentityRevision], frame_count: int) -> tuple[int, int]:
    if not revisions:
        return 0, frame_count - 1
    start = max(0, min(revision.start_frame for revision in revisions) - 2)
    end = min(frame_count - 1, max(revision.end_frame for revision in revisions) + 2)
    return start, end


def reconstruct(
    action_name: str,
    pipeline_name: str,
    capture_id: str,
    observations: list[Frame],
    calibration: dict[str, Any],
    *,
    max_gap_frames: int = 5,
    revisions: list[IdentityRevision] | None = None,
    based_on: ProcessingVersion | None = None,
) -> ProcessingVersion:
    revisions = revisions or []
    corrected, changed = remap_identities(observations, revisions)

    if based_on is not None and revisions:
        start, end = _affected_window(revisions, len(observations))
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
        parameters = {
            "max_gap_frames": max_gap_frames,
            "recompute_window": {"start_frame": start, "end_frame": end},
            "changed_frames": [i for i, value in enumerate(changed) if value],
        }
        version = _next_version(based_on.version)
    else:
        frames, statuses = interpolate_gaps(corrected, max_gap_frames)
        frames, statuses = apply_skeleton_constraints(frames, statuses, calibration)
        parameters = {"max_gap_frames": max_gap_frames, "recompute_window": None}
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
        based_on=based_on.version if based_on else None,
    )


def _next_version(version: str) -> str:
    return f"v{int(version.removeprefix('v')) + 1:03d}"
