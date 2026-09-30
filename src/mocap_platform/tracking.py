from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .geometry import distance, mean
from .models import IdentityRevision
from .skeleton import MIRROR_GROUP, PERSONS, group_markers, split_marker


@dataclass(frozen=True)
class GroupPair:
    person: str
    group: str

    @property
    def markers(self) -> list[str]:
        return group_markers(self.person, self.group)

    @property
    def key(self) -> tuple[str, str]:
        return self.person, self.group

    @property
    def mirror(self) -> "GroupPair":
        other_person, other_group = MIRROR_GROUP[self.key]
        return GroupPair(other_person, other_group)

    def __str__(self) -> str:
        return f"p{self.person}_{self.group}"


def limb_group_pairs() -> list[GroupPair]:
    from .skeleton import LIMB_GROUPS

    return [
        GroupPair(person, group)
        for person in PERSONS
        for group in LIMB_GROUPS
    ]


def _bone_table(calibration: dict[str, Any]) -> dict[tuple[str, str], dict[str, float]]:
    table: dict[tuple[str, str], dict[str, float]] = {}
    for bone_name, constraint in calibration["bone_lengths"].items():
        start, end = bone_name.split("->")
        table[(start, end)] = constraint
    return table


def _residual(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return distance(a, b)


def hypothesis_score(
    frame: dict[str, Any],
    pair: GroupPair,
    calibration: dict[str, Any],
) -> float | None:
    """Mean normalised residual for the *internal* bones of a limb group.

    Only bones whose both endpoints move with the group are considered:
    shoulders/hips stay anchored to the reliably tracked torso, while elbow/
    wrist (or knee/ankle) labels can be exchanged during a close pass. When
    two mirrored groups reappear symmetrically the internal-bone score is
    identical for both assignments.
    """
    bones = _bone_table(calibration)
    residuals: list[float] = []

    if pair.group.endswith("arm"):
        side = "l_" if pair.group.startswith("l_") else "r_"
        links = [
            (
                f"p{pair.person}_{side}shoulder",
                f"p{pair.person}_{side}elbow",
                (f"{side}shoulder", f"{side}elbow"),
            ),
            (
                f"p{pair.person}_{side}elbow",
                f"p{pair.person}_{side}wrist",
                (f"{side}elbow", f"{side}wrist"),
            ),
        ]
    else:
        side = "l_" if pair.group.startswith("l_") else "r_"
        links = [
            (
                f"p{pair.person}_{side}hip",
                f"p{pair.person}_{side}knee",
                (f"{side}hip", f"{side}knee"),
            ),
            (
                f"p{pair.person}_{side}knee",
                f"p{pair.person}_{side}ankle",
                (f"{side}knee", f"{side}ankle"),
            ),
        ]

    for proximal_name, distal_name, key in links:
        proximal, distal = frame.get(proximal_name), frame.get(distal_name)
        residual = _residual(proximal, distal)
        constraint = bones.get(key)
        if residual is None or constraint is None:
            continue
        residuals.append(abs(residual - constraint["length"]) / constraint["tolerance"])

    return mean(residuals) if residuals else None


def _had_bilateral_occlusion(
    observations: list[dict[str, Any]],
    pair: GroupPair,
    frame_index: int,
    lookback: int,
) -> bool:
    mirror = pair.mirror
    distal = [pair.markers[-1], mirror.markers[-1]]
    start = max(0, frame_index - lookback)
    return any(
        all(observations[index].get(marker) is None for marker in distal)
        for index in range(start, frame_index)
    )


def detect_identity_ambiguities(
    observations: list[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    tie_tolerance: float = 0.25,
    occlusion_lookback: int = 8,
) -> list[dict[str, Any]]:
    """Flag windows where two assignments are equally plausible.

    For every pair of mirrored limb groups, compare the evidence for the
    current labelling against the swapped labelling. When the two scores are
    statistically indistinguishable (symmetric pass, simultaneous occlusion
    exit) the window is reported as ambiguous instead of being silently
    committed either way.
    """
    ambiguities: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for pair in limb_group_pairs():
        if pair.key in seen_pairs:
            continue
        mirror = pair.mirror
        seen_pairs.add(pair.key)
        seen_pairs.add(mirror.key)

        ambiguous_frames: list[int] = []
        current_scores: list[float] = []
        swapped_scores: list[float] = []
        for frame_index, frame in enumerate(observations):
            current = hypothesis_score(frame, pair, calibration)
            swapped_frame = dict(frame)
            for marker, other in zip(pair.markers, mirror.markers):
                swapped_frame[marker], swapped_frame[other] = (
                    frame.get(other),
                    frame.get(marker),
                )
            swapped = hypothesis_score(swapped_frame, pair, calibration)
            if current is None or swapped is None:
                continue
            bilateral_gap = _had_bilateral_occlusion(
                observations, pair, frame_index, occlusion_lookback
            )
            if (
                bilateral_gap
                and abs(current - swapped) <= tie_tolerance
                and _groups_close(frame, pair, mirror, limit=0.12)
            ):
                ambiguous_frames.append(frame_index)
                current_scores.append(current)
                swapped_scores.append(swapped)

        for start, end in _runs(ambiguous_frames):
            ambiguities.append(
                {
                    "groups": [str(pair), str(mirror)],
                    "markers": sorted(pair.markers + mirror.markers),
                    "group_marker_sets": [pair.markers, mirror.markers],
                    "start_frame": start,
                    "end_frame": end,
                    "current_score": round(mean(current_scores), 6),
                    "swapped_score": round(mean(swapped_scores), 6),
                    "resolution": "unresolved",
                }
            )
    return ambiguities


def _groups_close(
    frame: dict[str, Any], pair: GroupPair, mirror: GroupPair, *, limit: float = 0.15
) -> bool:
    distances = []
    for marker, other in zip(pair.markers, mirror.markers):
        a, b = frame.get(marker), frame.get(other)
        if a is not None and b is not None:
            distances.append(distance(a, b))
    return bool(distances) and min(distances) <= 0.15


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        runs.append((start, previous))
        start = previous = index
    runs.append((start, previous))
    return runs


def parse_group(group_name: str) -> GroupPair:
    person, group = group_name.removeprefix("p").split("_", 1)
    return GroupPair(person, group)


def swap_groups(
    frames: list[dict[str, Any]],
    pair: GroupPair,
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    """Return frames with two mirrored groups exchanged in a window.

    Raw observations are never mutated: callers pass copies.
    """
    mirror = pair.mirror
    result = [dict(frame) for frame in frames]
    for frame_index in range(max(0, start), min(len(frames) - 1, end) + 1):
        for marker, other in zip(pair.markers, mirror.markers):
            result[frame_index][marker], result[frame_index][other] = (
                frames[frame_index].get(other),
                frames[frame_index].get(marker),
            )
    return result


def active_revisions(
    revisions: list[IdentityRevision], revoked_ids: set[str] | None = None
) -> list[IdentityRevision]:
    revoked_ids = revoked_ids or set()
    superseded = {revision.supersedes for revision in revisions if revision.supersedes}
    return [
        revision
        for revision in revisions
        if revision.revision_id not in revoked_ids
        and revision.revision_id not in superseded
    ]


def apply_revisions(
    observations: list[dict[str, Any]], revisions: list[IdentityRevision]
) -> tuple[list[dict[str, Any]], list[int]]:
    """Apply confirmations/denials by swapping whole mirrored limb groups.

    Each revision names one marker in a group; the swap propagates to every
    joint in the group (shoulder/elbow/wrist or hip/knee/ankle). A ``deny``
    sends the group to its mirrored counterpart; a ``confirm`` pins the
    current labelling (a no-op swap) and resolves ambiguity.
    """
    corrected: list[dict[str, Any]] = [dict(frame) for frame in observations]
    changed = [False] * len(observations)
    for revision in revisions:
        person, joint = split_marker(revision.marker)
        if revision.group:
            group = revision.group
        else:
            from .skeleton import marker_group

            group = marker_group(joint) or ""
        if not group:
            continue
        pair = GroupPair(person, group)
        window_start = revision.start_frame
        window_end = revision.end_frame
        if revision.decision == "deny":
            corrected = swap_groups(corrected, pair, window_start, window_end)
        for frame_index in range(
            max(0, window_start), min(len(observations) - 1, window_end) + 1
        ):
            changed[frame_index] = True
    return corrected, changed


def confirmed_windows(revisions: list[IdentityRevision]) -> dict[str, tuple[int, int, str]]:
    """Map marker -> union window with the latest decision covering it."""
    windows: dict[str, tuple[int, int, str]] = {}
    for revision in revisions:
        person, joint = split_marker(revision.marker)
        pair = GroupPair(person, revision.group) if revision.group else None
        markers = pair.markers if pair else [revision.marker]
        for marker in markers:
            previous = windows.get(marker)
            if previous is None:
                windows[marker] = (revision.start_frame, revision.end_frame, revision.decision)
            else:
                windows[marker] = (
                    min(previous[0], revision.start_frame),
                    max(previous[1], revision.end_frame),
                    revision.decision,
                )
    return windows
