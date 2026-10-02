from __future__ import annotations

from typing import Any

from .identity import person_marker


CONTACT_DISTANCE_TOLERANCE = 0.15
AUTO_SOURCE_PREFIX = "automatic"


def contact_marker(contact: Any) -> str:
    return person_marker(contact.person_id, contact.body_part)


def _frames_overlap(
    first_start: int, first_end: int, second_start: int, second_end: int
) -> bool:
    return first_start <= second_end and second_start <= first_end


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[dict[str, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: list[list[int]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [{"start_frame": start, "end_frame": end} for start, end in merged]


def candidate_for_contact(candidates: list[Any], contact: Any) -> Any | None:
    for candidate in candidates:
        if candidate.decision == "excluded":
            continue
        if candidate.person_id != contact.person_id:
            continue
        if not _frames_overlap(
            candidate.start_frame,
            candidate.end_frame,
            contact.start_frame,
            contact.end_frame,
        ):
            continue
        if contact.body_part in candidate.joint_mapping:
            return candidate
    return None


def interpreted_body_marker(candidate: Any | None, contact: Any) -> str:
    if candidate is not None:
        return candidate.joint_mapping[contact.body_part]
    return contact_marker(contact)


def _distance(first: list[float], second: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(first, second)) ** 0.5


def evaluate_contact_compatibility(
    contacts: list[Any],
    candidates: list[Any],
    raw_observations: list[dict[str, Any]],
    objects: dict[str, list[list[float] | None]],
    *,
    tolerance: float = CONTACT_DISTANCE_TOLERANCE,
) -> dict[str, Any]:
    """Compare each identity candidate against sourced contact evidence.

    Only raw observed positions are used; interpolated/recovered positions are
    never passed in here, so missing observations cannot manufacture certainty.
    Verdicts are ``geometrically_incompatible``, ``unobserved`` and
    ``still_plausible``.
    """
    results: list[dict[str, Any]] = []
    simultaneous: list[dict[str, Any]] = []
    active_contacts = [
        contact
        for contact in contacts
        if getattr(contact, "status", "active") == "active"
    ]
    for contact in active_contacts:
        candidate = candidate_for_contact(candidates, contact)
        body_marker = interpreted_body_marker(candidate, contact)
        object_track = objects.get(contact.object_id, [])
        supported: list[int] = []
        incompatible: list[dict[str, Any]] = []
        missing: list[int] = []
        observations_used: list[dict[str, Any]] = []
        start = max(0, contact.start_frame)
        end = min(len(raw_observations) - 1, contact.end_frame)
        for frame_index in range(start, end + 1):
            body = raw_observations[frame_index].get(body_marker)
            obj = (
                object_track[frame_index]
                if frame_index < len(object_track)
                else None
            )
            if body is None or obj is None:
                missing.append(frame_index)
                continue
            gap = _distance(body, obj)
            observations_used.append(
                {
                    "frame": frame_index,
                    "object_id": contact.object_id,
                    "observed_marker": body_marker,
                    "distance": gap,
                }
            )
            if gap <= tolerance:
                supported.append(frame_index)
            else:
                incompatible.append({"frame": frame_index, "distance": gap})
        if incompatible:
            verdict = "geometrically_incompatible"
        elif supported:
            verdict = "still_plausible"
        else:
            verdict = "unobserved"
        results.append(
            {
                "contact_id": contact.contact_id,
                "candidate_id": candidate.candidate_id if candidate else None,
                "person_id": contact.person_id,
                "body_part": contact.body_part,
                "object_id": contact.object_id,
                "start_frame": start,
                "end_frame": end,
                "observed_marker": body_marker,
                "verdict": verdict,
                "supported_frames": supported,
                "incompatible_frames": incompatible,
                "missing_observation_frames": missing,
                "conflict_intervals": _merge_intervals(
                    [(item["frame"], item["frame"]) for item in incompatible]
                ),
                "missing_intervals": _merge_intervals(
                    [(frame, frame) for frame in missing]
                ),
                "observations_used": observations_used,
                "source": contact.source,
                "reason": contact.reason,
            }
        )

    # Two people may touch the same object at the same time; that is allowed.
    for index, first in enumerate(active_contacts):
        for second in active_contacts[index + 1 :]:
            if first.object_id != second.object_id:
                continue
            start = max(first.start_frame, second.start_frame)
            end = min(first.end_frame, second.end_frame)
            if start > end:
                continue
            first_candidate = candidate_for_contact(candidates, first)
            second_candidate = candidate_for_contact(candidates, second)
            first_marker = interpreted_body_marker(first_candidate, first)
            second_marker = interpreted_body_marker(second_candidate, second)
            for frame_index in range(start, end + 1):
                obj = objects.get(first.object_id, [])
                obj_point = obj[frame_index] if frame_index < len(obj) else None
                first_point = raw_observations[frame_index].get(first_marker)
                second_point = raw_observations[frame_index].get(second_marker)
                if obj_point is None or first_point is None or second_point is None:
                    continue
                if (
                    _distance(first_point, obj_point) <= tolerance
                    and _distance(second_point, obj_point) <= tolerance
                ):
                    simultaneous.append(
                        {
                            "frame": frame_index,
                            "object_id": first.object_id,
                            "contacts": [first.contact_id, second.contact_id],
                            "markers": [first_marker, second_marker],
                        }
                    )

    return {
        "tolerance": tolerance,
        "contact_results": results,
        "simultaneous_contacts": simultaneous,
        "claim_conflicts": candidate_claim_conflicts(candidates),
    }


def candidate_claim_conflicts(candidates: list[Any]) -> list[dict[str, Any]]:
    """One observed body trajectory cannot be claimed by two identities."""
    claims: dict[tuple[str, int], list[Any]] = {}
    for candidate in candidates:
        if candidate.decision == "excluded":
            continue
        for marker in candidate.joint_mapping.values():
            for frame_index in range(candidate.start_frame, candidate.end_frame + 1):
                claims.setdefault((marker, frame_index), []).append(candidate)
    conflicts: list[dict[str, Any]] = []
    recorded: set[tuple[str, ...]] = set()
    for (marker, frame_index), owners in claims.items():
        identities = {candidate.person_id for candidate in owners}
        if len(owners) < 2 or len(identities) < 2:
            continue
        owner_ids = tuple(sorted(candidate.candidate_id for candidate in owners))
        key = owner_ids + (marker,)
        if key in recorded:
            continue
        recorded.add(key)
        shared_frames = sorted(
            other_frame
            for (other_marker, other_frame), other_owners in claims.items()
            if other_marker == marker
            and tuple(sorted(candidate.candidate_id for candidate in other_owners))
            == owner_ids
        )
        conflicts.append(
            {
                "marker": marker,
                "candidate_ids": list(owner_ids),
                "person_ids": sorted(identities),
                "intervals": _merge_intervals(
                    [(shared_frames[0], shared_frames[-1])]
                ),
                "frames": shared_frames,
                "reason": (
                    "the same observed body trajectory is claimed by two "
                    "identities; simultaneous object contact does not allow this"
                ),
            }
        )
    return conflicts


def candidates_dependent_on(
    candidates: list[Any],
    revoked_contact_ids: set[str],
    revoked_reassignment_ids: set[str],
    revoked_decision_ids: set[str] | None = None,
) -> list[str]:
    revoked_decision_ids = revoked_decision_ids or set()
    changed = revoked_contact_ids | revoked_reassignment_ids | revoked_decision_ids
    return [
        candidate.candidate_id
        for candidate in candidates
        if set(candidate.depends_on) & changed
    ]


def schemes_dependent_on(schemes: list[Any], candidate_ids: set[str]) -> list[str]:
    return [
        scheme.scheme_id
        for scheme in schemes
        if set(scheme.candidate_ids) & candidate_ids
    ]


def is_manual_source(source: str) -> bool:
    return not source.startswith(AUTO_SOURCE_PREFIX)


def disputed_contact_intervals(contact_review: dict[str, Any]) -> dict[str, Any]:
    geometric: list[tuple[int, int]] = []
    unobserved: list[tuple[int, int]] = []
    for result in contact_review.get("contact_results", []):
        for interval in result["conflict_intervals"]:
            geometric.append((interval["start_frame"], interval["end_frame"]))
        if result["verdict"] == "unobserved":
            unobserved.append((result["start_frame"], result["end_frame"]))
    claim: list[tuple[int, int]] = []
    for conflict in contact_review.get("claim_conflicts", []):
        for interval in conflict["intervals"]:
            claim.append((interval["start_frame"], interval["end_frame"]))
    return {
        "contact_conflict_intervals": _merge_intervals(geometric),
        "unobserved_contact_intervals": _merge_intervals(unobserved),
        "identity_claim_conflict_intervals": _merge_intervals(claim),
    }


def unresolved_pose_intervals(
    identity_statuses: list[dict[str, str]],
    start_frame: int,
    end_frame: int,
) -> list[dict[str, int]]:
    frames: list[int] = []
    upper = min(end_frame, len(identity_statuses) - 1)
    for frame_index in range(max(0, start_frame), upper + 1):
        if any(status == "ambiguous" for status in identity_statuses[frame_index].values()):
            frames.append(frame_index)
    return _merge_intervals([(frame, frame) for frame in frames])


def object_missing_intervals(
    objects: dict[str, list[list[float] | None]],
) -> dict[str, list[dict[str, int]]]:
    return {
        object_id: _merge_intervals(
            [(frame, frame) for frame, point in enumerate(positions) if point is None]
        )
        for object_id, positions in objects.items()
    }
