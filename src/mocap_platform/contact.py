from __future__ import annotations

from typing import Any

from .identity import person_marker


CONTACT_DISTANCE_TOLERANCE = 0.15
AUTO_SOURCE_PREFIX = "automatic"
SUPPLEMENT_SOURCE_PREFIX = "supplement"
DUPLICATE_DISTANCE_TOLERANCE = 1e-6


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


def _subject_point_options(
    supplement_index: dict[tuple[str, str], list[dict[str, Any]]],
    subject_kind: str,
    subject_id: str,
    frame_index: int,
    raw_point: list[float] | None,
) -> dict[str, Any]:
    """Raw observation always wins; supplements only cover its missing frames."""
    entries = [
        entry
        for entry in supplement_index.get((subject_kind, subject_id), [])
        if entry["frame"] == frame_index
    ]
    duplicates: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    if raw_point is not None:
        # New material on a frame already visible to the old analysis never
        # adds evidence; adopted supplements are recorded as duplicates.
        for entry in entries:
            if entry["adopted"] and _distance(raw_point, entry["point"]) <= DUPLICATE_DISTANCE_TOLERANCE:
                duplicates.append(
                    {
                        "subject": subject_kind,
                        "subject_id": subject_id,
                        "batch_id": entry["batch_id"],
                        "reason": "frame already visible in the original analysis",
                    }
                )
        return {
            "points": [
                {"point": list(raw_point), "source": "raw_observation", "batch_id": None}
            ],
            "duplicates": duplicates,
            "pending": pending,
        }
    adopted: list[dict[str, Any]] = []
    for entry in entries:
        if entry["adopted"]:
            candidate = {
                "point": list(entry["point"]),
                "source": f"{SUPPLEMENT_SOURCE_PREFIX}:{entry['source']}",
                "batch_id": entry["batch_id"],
                "duplicate_batch_ids": [],
            }
            adopted.append(candidate)
        else:
            pending.append(
                {
                    "subject": subject_kind,
                    "subject_id": subject_id,
                    "batch_id": entry["batch_id"],
                    "reason": "supplemental material available but not adopted",
                }
            )
    unique: list[dict[str, Any]] = []
    for entry in adopted:
        if any(
            _distance(entry["point"], other["point"]) <= DUPLICATE_DISTANCE_TOLERANCE
            for other in unique
        ):
            duplicates.append(
                {
                    "subject": subject_kind,
                    "subject_id": subject_id,
                    "batch_id": entry["batch_id"],
                    "reason": "identical supplemental point repeated in another batch",
                }
            )
            for other in unique:
                if _distance(entry["point"], other["point"]) <= DUPLICATE_DISTANCE_TOLERANCE:
                    other["duplicate_batch_ids"].append(entry["batch_id"])
                    # Keep the first adopted batch as the geometry source so
                    # repeated adoption does not add a second evidence point.
                    break
        else:
            unique.append(entry)
    return {"points": unique, "duplicates": duplicates, "pending": pending}


def _same_subject_conflicts(
    frame_index: int,
    options: list[dict[str, Any]],
    subject_id: str,
    subject_kind: str,
) -> list[dict[str, Any]]:
    """Side-by-side listing when adopted batches disagree on one subject."""
    if len(options) < 2:
        return []
    clashes: list[dict[str, Any]] = []
    for index, first in enumerate(options):
        for second in options[index + 1 :]:
            if first["batch_id"] == second["batch_id"]:
                continue
            if _distance(first["point"], second["point"]) <= DUPLICATE_DISTANCE_TOLERANCE:
                continue
            clashes.append(
                {
                    "frame": frame_index,
                    "subject": subject_kind,
                    "subject_id": subject_id,
                    "batches": sorted(
                        {first["batch_id"], second["batch_id"]} - {None}
                    ),
                    "reason": "adopted supplemental tracks disagree at this frame",
                }
            )
    return clashes


def _track_interval(track: Any) -> tuple[int, int]:
    return track.start_frame, track.start_frame + len(track.points) - 1


def resolve_supplement_points(
    supplement_batches: list[Any] | None,
    supplement_adoptions: list[Any] | None,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Index every supplemental frame point by (subject_kind, subject_id).

    All active batch material is indexed with its adoption state:
    ``adopted`` points are usable for geometry; ``pending`` points stay on
    display so reviewers can choose them later. Each point keeps the frame,
    batch id and source for per-frame provenance.
    """
    batches = supplement_batches or []
    adoptions = supplement_adoptions or []
    applied_selections: dict[str, list[Any]] = {}
    for adoption in adoptions:
        if getattr(adoption, "status", "applied") != "applied":
            continue
        applied_selections.setdefault(adoption.batch_id, []).extend(
            adoption.selections
        )
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for batch in batches:
        if getattr(batch, "status", "active") != "active":
            continue
        selections = applied_selections.get(batch.batch_id, [])
        for track in batch.tracks:
            track_start, track_end = _track_interval(track)
            for offset, point in enumerate(track.points):
                if point is None:
                    continue
                frame_index = track_start + offset
                adopted = any(
                    selection.subject_id == track.subject_id
                    and selection.subject_kind == track.subject_kind
                    and selection.start_frame <= frame_index <= selection.end_frame
                    for selection in selections
                )
                # Selections are validated against their track ranges at
                # construction time; clip defensively to the track interval.
                if frame_index < track_start or frame_index > track_end:
                    continue
                index.setdefault(
                    (track.subject_kind, track.subject_id), []
                ).append(
                    {
                        "frame": frame_index,
                        "point": list(point),
                        "batch_id": batch.batch_id,
                        "source": batch.source,
                        "object_id": batch.object_id,
                        "adopted": adopted,
                    }
                )
    return index


def evaluate_contact_compatibility(
    contacts: list[Any],
    candidates: list[Any],
    raw_observations: list[dict[str, Any]],
    objects: dict[str, list[list[float] | None]],
    *,
    tolerance: float = CONTACT_DISTANCE_TOLERANCE,
    supplement_index: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Compare each identity candidate against sourced contact evidence.

    Only raw observed positions and explicitly adopted supplemental points
    are used; interpolated/recovered positions are never passed in here, so
    missing observations cannot manufacture certainty. Supplements never
    overwrite a raw-observed frame. Verdicts are
    ``geometrically_incompatible``, ``unobserved``, ``still_plausible`` and
    ``supplement_conflict`` (adopted supplemental batches disagree).
    """
    supplement_index = supplement_index or {}
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
        supported_by_supplement: list[int] = []
        conflicting_supplements: list[dict[str, Any]] = []
        duplicate_supplements: list[dict[str, Any]] = []
        pending_supplements: list[dict[str, Any]] = []
        start = max(0, contact.start_frame)
        # Keep the annotated range intact: frames past the capture length are
        # retained as missing evidence instead of truncating the annotation.
        end = contact.end_frame
        for frame_index in range(start, end + 1):
            raw_body = (
                raw_observations[frame_index].get(body_marker)
                if frame_index < len(raw_observations)
                else None
            )
            raw_obj = (
                object_track[frame_index]
                if frame_index < len(object_track)
                else None
            )
            body_options = _subject_point_options(
                supplement_index, "body", body_marker, frame_index, raw_body
            )
            object_options = _subject_point_options(
                supplement_index, "object", contact.object_id, frame_index, raw_obj
            )
            for duplicate in body_options["duplicates"] + object_options["duplicates"]:
                duplicate["frame"] = frame_index
                duplicate_supplements.append(duplicate)
            for pending in body_options["pending"] + object_options["pending"]:
                pending["frame"] = frame_index
                pending_supplements.append(pending)
            conflicting_supplements.extend(
                _same_subject_conflicts(
                    frame_index, body_options["points"], body_marker, "body"
                )
            )
            conflicting_supplements.extend(
                _same_subject_conflicts(
                    frame_index,
                    object_options["points"],
                    contact.object_id,
                    "object",
                )
            )
            body_points = body_options["points"]
            object_points = object_options["points"]
            if not body_points or not object_points:
                missing.append(frame_index)
                continue
            frame_supported = False
            frame_incompatible = False
            frame_via_supplement = False
            for body_choice in body_points:
                for object_choice in object_points:
                    gap = _distance(body_choice["point"], object_choice["point"])
                    used = {
                        "frame": frame_index,
                        "object_id": contact.object_id,
                        "observed_marker": body_marker,
                        "distance": gap,
                        "body_source": body_choice["source"],
                        "body_batch_id": body_choice["batch_id"],
                        "object_source": object_choice["source"],
                        "object_batch_id": object_choice["batch_id"],
                    }
                    observations_used.append(used)
                    if gap <= tolerance:
                        frame_supported = True
                        if (
                            body_choice["batch_id"] is not None
                            or object_choice["batch_id"] is not None
                        ):
                            frame_via_supplement = True
                    else:
                        # A single adopted point failing the threshold is
                        # evidence against contact; its source stays on the
                        # observation record. Same-subject batch clashes were
                        # recorded separately above.
                        frame_incompatible = True
            if frame_supported:
                supported.append(frame_index)
                if frame_via_supplement:
                    supported_by_supplement.append(frame_index)
            elif frame_incompatible:
                incompatible.append(
                    {
                        "frame": frame_index,
                        "distance": min(
                            _distance(b["point"], o["point"])
                            for b in body_points
                            for o in object_points
                        ),
                    }
                )
            # Otherwise every adopted combination clashes on the same
            # subject; the conflict record already documents the frame.
        if incompatible:
            verdict = "geometrically_incompatible"
        elif supported:
            verdict = "still_plausible"
        elif any(
            start <= item["frame"] <= end for item in conflicting_supplements
        ):
            verdict = "supplement_conflict"
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
                "supported_by_supplement_frames": supported_by_supplement,
                "conflicting_supplements": [
                    item
                    for item in conflicting_supplements
                    if start <= item["frame"] <= end
                ],
                "conflicting_supplement_intervals": _merge_intervals(
                    [
                        (item["frame"], item["frame"])
                        for item in conflicting_supplements
                        if start <= item["frame"] <= end
                    ]
                ),
                "duplicate_supplement_frames": sorted(
                    {item["frame"] for item in duplicate_supplements}
                ),
                "duplicate_supplements": duplicate_supplements,
                "pending_supplement_frames": sorted(
                    {item["frame"] for item in pending_supplements}
                ),
                "pending_supplements": pending_supplements,
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
                if frame_index >= len(raw_observations):
                    # Beyond the capture length: the per-contact results already
                    # retain these frames as missing evidence; missing is not
                    # treated as separation, so simply stop checking here.
                    break
                obj = objects.get(first.object_id, [])
                obj_options = _subject_point_options(
                    supplement_index,
                    "object",
                    first.object_id,
                    frame_index,
                    obj[frame_index] if frame_index < len(obj) else None,
                )
                first_options = _subject_point_options(
                    supplement_index,
                    "body",
                    first_marker,
                    frame_index,
                    raw_observations[frame_index].get(first_marker),
                )
                second_options = _subject_point_options(
                    supplement_index,
                    "body",
                    second_marker,
                    frame_index,
                    raw_observations[frame_index].get(second_marker),
                )
                if (
                    not obj_options["points"]
                    or not first_options["points"]
                    or not second_options["points"]
                ):
                    continue
                for obj_choice in obj_options["points"]:
                    for first_choice in first_options["points"]:
                        for second_choice in second_options["points"]:
                            if (
                                _distance(first_choice["point"], obj_choice["point"])
                                <= tolerance
                                and _distance(
                                    second_choice["point"], obj_choice["point"]
                                )
                                <= tolerance
                            ):
                                simultaneous.append(
                                    {
                                        "frame": frame_index,
                                        "object_id": first.object_id,
                                        "contacts": [
                                            first.contact_id,
                                            second.contact_id,
                                        ],
                                        "markers": [first_marker, second_marker],
                                        "body_batch_ids": [
                                            first_choice["batch_id"],
                                            second_choice["batch_id"],
                                        ],
                                        "object_batch_id": obj_choice["batch_id"],
                                    }
                                )

    return {
        "tolerance": tolerance,
        "contact_results": results,
        "simultaneous_contacts": simultaneous,
        "claim_conflicts": candidate_claim_conflicts(candidates),
        "supplement_review": summarize_supplement_review(
            results, supplement_index, raw_observations, objects
        ),
    }


def summarize_supplement_review(
    contact_results: list[dict[str, Any]],
    supplement_index: dict[tuple[str, str], list[dict[str, Any]]],
    raw_observations: list[dict[str, Any]] | None = None,
    objects: dict[str, list[list[float] | None]] | None = None,
) -> dict[str, Any]:
    supported: list[tuple[int, int]] = []
    conflicts: list[dict[str, Any]] = []
    pending: list[tuple[int, int]] = []
    duplicate_frames: set[int] = set()
    batch_ids: set[str] = set()
    for result in contact_results:
        for frame_index in result.get("supported_by_supplement_frames", []):
            supported.append((frame_index, frame_index))
        for frame_index in result.get("pending_supplement_frames", []):
            pending.append((frame_index, frame_index))
        duplicate_frames.update(result.get("duplicate_supplement_frames", []))
        for item in result.get("conflicting_supplements", []):
            if item not in conflicts:
                conflicts.append(item)
            for batch_id in item.get("batches", []):
                batch_ids.add(batch_id)
    # Every handed-in frame stays visible even when it overlaps a frame the
    # original analysis already saw: adopted such frames add no evidence, so
    # they are reported as duplicates rather than support.
    raw_duplicate: list[tuple[int, int]] = []
    for (subject_kind, subject_id), entries in supplement_index.items():
        for entry in entries:
            batch_ids.add(entry["batch_id"])
            if not entry["adopted"]:
                pending.append((entry["frame"], entry["frame"]))
                continue
            frame_index = entry["frame"]
            if subject_kind == "body":
                already_visible = bool(
                    raw_observations
                    and frame_index < len(raw_observations)
                    and raw_observations[frame_index].get(subject_id) is not None
                )
            else:
                track = (objects or {}).get(subject_id, [])
                already_visible = bool(
                    frame_index < len(track) and track[frame_index] is not None
                )
            if already_visible:
                raw_duplicate.append((frame_index, frame_index))
    return {
        "batch_ids": sorted(batch_ids),
        "supported_intervals": _merge_intervals(supported),
        "conflicting_intervals": _merge_intervals(
            [(item["frame"], item["frame"]) for item in conflicts]
        ),
        "conflicts": conflicts,
        "still_missing_intervals": _merge_intervals(
            [
                (frame_index, frame_index)
                for result in contact_results
                for frame_index in result.get("missing_observation_frames", [])
            ]
        ),
        "pending_intervals": _merge_intervals(pending),
        "duplicate_frames": sorted(
            duplicate_frames | {frame for frame, _end in raw_duplicate}
        ),
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
    supplemental_conflict: list[tuple[int, int]] = []
    for result in contact_review.get("contact_results", []):
        for interval in result["conflict_intervals"]:
            geometric.append((interval["start_frame"], interval["end_frame"]))
        for interval in result.get("conflicting_supplement_intervals", []):
            supplemental_conflict.append(
                (interval["start_frame"], interval["end_frame"])
            )
        if result["verdict"] == "unobserved":
            unobserved.append((result["start_frame"], result["end_frame"]))
    claim: list[tuple[int, int]] = []
    for conflict in contact_review.get("claim_conflicts", []):
        for interval in conflict["intervals"]:
            claim.append((interval["start_frame"], interval["end_frame"]))
    return {
        "contact_conflict_intervals": _merge_intervals(geometric),
        "unobserved_contact_intervals": _merge_intervals(unobserved),
        "supplement_conflict_intervals": _merge_intervals(supplemental_conflict),
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
