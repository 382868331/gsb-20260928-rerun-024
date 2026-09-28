from __future__ import annotations

from dataclasses import dataclass
from math import inf

from .geometry import Vec3, angle_between, rotate_to_opening_angle
from .model import IdentityRevision, JointState, Observation, SkeletonDefinition


@dataclass(frozen=True)
class ProcessingConfig:
    max_short_gap: int = 5
    bone_tolerance: float = 0.02
    angle_tolerance: float = 0.02


RecoveryResult = tuple[list[dict[str, JointState]], list[int], list[str]]


def apply_identity_revisions(
    observations: list[Observation], revisions: tuple[IdentityRevision, ...]
) -> list[Observation]:
    revised: list[Observation] = []
    for observation in observations:
        marker = observation.marker
        sources = ("raw",)
        for revision in revisions:
            if revision.applies(observation.frame) and marker in revision.mapping:
                marker = revision.mapping[marker]
                sources = ("raw", revision.revision_id)
        revised.append(
            Observation(
                frame=observation.frame,
                marker=marker,
                position=observation.position,
                confidence=observation.confidence,
            )
            if sources == ("raw",)
            else RevisedObservation(
                frame=observation.frame,
                marker=marker,
                position=observation.position,
                confidence=observation.confidence,
                sources=sources,
            )
        )
    return revised


@dataclass(frozen=True)
class RevisedObservation(Observation):
    sources: tuple[str, ...] = ("raw",)


def _index_observations(
    frame_count: int, joints: tuple[str, ...], observations: list[Observation]
) -> list[dict[str, Observation]]:
    indexed = [dict() for _ in range(frame_count)]
    for observation in observations:
        if 0 <= observation.frame < frame_count and observation.marker in joints:
            indexed[observation.frame][observation.marker] = observation
    return indexed


def _interpolate_short_gaps(
    indexed: list[dict[str, Observation]],
    joints: tuple[str, ...],
    config: ProcessingConfig,
) -> tuple[list[dict[str, JointState]], list[str]]:
    states: list[dict[str, JointState]] = [dict() for _ in indexed]
    warnings: list[str] = []

    for joint in joints:
        present_frames = [frame for frame, row in enumerate(indexed) if joint in row]
        for frame, row in enumerate(indexed):
            if joint in row:
                observation = row[joint]
                sources = getattr(observation, "sources", ("raw",))
                states[frame][joint] = JointState(observation.position, "observed", sources)

        for left_frame, right_frame in zip(present_frames, present_frames[1:]):
            missing = right_frame - left_frame - 1
            if missing == 0:
                continue
            if missing > config.max_short_gap:
                warnings.append(
                    f"{joint}: frames {left_frame + 1}-{right_frame - 1} left undetermined "
                    f"(gap length {missing} > {config.max_short_gap})"
                )
                continue
            left = indexed[left_frame][joint].position
            right = indexed[right_frame][joint].position
            for frame in range(left_frame + 1, right_frame):
                alpha = (frame - left_frame) / (right_frame - left_frame)
                position = left + (right - left) * alpha
                states[frame][joint] = JointState(position, "interpolated", ("gap-fill",))

        if present_frames:
            first = present_frames[0]
            last = present_frames[-1]
            if first > 0:
                warnings.append(f"{joint}: frames 0-{first - 1} left undetermined (leading gap)")
            if last < len(indexed) - 1:
                warnings.append(
                    f"{joint}: frames {last + 1}-{len(indexed) - 1} left undetermined (trailing gap)"
                )
    return states, warnings


def _enforce_constraints(
    states: list[dict[str, JointState]],
    skeleton: SkeletonDefinition,
    config: ProcessingConfig,
) -> None:
    parents = skeleton.parent_of()
    ordered_joints: list[str] = [skeleton.root]
    changed = True
    while changed:
        changed = False
        for parent, child in skeleton.bones:
            if parent in ordered_joints and child not in ordered_joints:
                ordered_joints.append(child)
                changed = True

    children: dict[str, list[str]] = {joint: [] for joint in skeleton.joints}
    for parent, child in skeleton.bones:
        children[parent].append(child)

    for frame, row in enumerate(states):
        for child in ordered_joints[1:]:
            parent = parents[child]
            if parent is None:
                continue
            parent_state = row.get(parent)
            child_state = row.get(child)
            if (
                parent_state is None
                or child_state is None
                or parent_state.position is None
                or child_state.position is None
            ):
                continue

            bone = (parent, child)
            target_length = skeleton.bone_lengths[bone]
            offset = child_state.position - parent_state.position
            current_length = offset.length()
            position = child_state.position
            if current_length > 1e-9 and abs(current_length - target_length) > target_length * config.bone_tolerance:
                position = parent_state.position + offset.normalized() * target_length

            for grandparent in children:
                limit = skeleton.joint_limits.get((grandparent, parent, child))
                if limit is None:
                    continue
                gp_state = row.get(grandparent)
                if gp_state is None or gp_state.position is None:
                    continue
                incoming = parent_state.position - gp_state.position
                outgoing = position - parent_state.position
                if incoming.length() < 1e-9 or outgoing.length() < 1e-9:
                    continue
                current_angle = angle_between(incoming, outgoing)
                low, high = limit
                target_angle = min(max(current_angle, low), high)
                if abs(current_angle - target_angle) > config.angle_tolerance:
                    corrected = rotate_to_opening_angle(outgoing, incoming, target_angle)
                    position = parent_state.position + corrected.normalized() * target_length

            if position != child_state.position:
                status = "constrained" if child_state.status == "interpolated" else "observed_constrained"
                row[child] = JointState(position, status, child_state.sources)


def recover_motion(
    frame_count: int,
    observations: list[Observation],
    skeleton: SkeletonDefinition,
    revisions: tuple[IdentityRevision, ...] = (),
    config: ProcessingConfig | None = None,
) -> RecoveryResult:
    config = config or ProcessingConfig()
    revised_observations = apply_identity_revisions(observations, revisions)
    indexed = _index_observations(frame_count, skeleton.joints, revised_observations)
    states, warnings = _interpolate_short_gaps(indexed, skeleton.joints, config)
    _enforce_constraints(states, skeleton, config)

    unknown_frames: list[int] = []
    for frame, row in enumerate(states):
        for joint in skeleton.joints:
            state = row.get(joint)
            if state is None or state.position is None:
                row[joint] = JointState(None, "unknown")
        if any(row[joint].status == "unknown" for joint in skeleton.joints):
            unknown_frames.append(frame)

    return states, unknown_frames, warnings


def rmse(recovered: list[dict[str, JointState]], truth: list[dict[str, Vec3]]) -> float:
    total = 0.0
    count = 0
    for frame, row in enumerate(recovered):
        for joint, state in row.items():
            if state.position is not None and joint in truth[frame]:
                total += state.position.distance(truth[frame][joint]) ** 2
                count += 1
    if count == 0:
        return inf
    return (total / count) ** 0.5
