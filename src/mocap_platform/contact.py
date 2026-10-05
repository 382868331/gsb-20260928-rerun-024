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
        # Keep the annotated range intact: frames past the capture length are
        # retained as missing evidence instead of truncating the annotation.
        end = contact.end_frame
        for frame_index in range(start, end + 1):
            body = (
                raw_observations[frame_index].get(body_marker)
                if frame_index < len(raw_observations)
                else None
            )
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
                if frame_index >= len(raw_observations):
                    # Beyond the capture length: the per-contact results already
                    # retain these frames as missing evidence; missing is not
                    # treated as separation, so simply stop checking here.
                    break
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


OBJECT_TARGET = "object"


def _track_point(track: dict[str, Any] | None, frame_index: int) -> list[float] | None:
    if not track:
        return None
    offset = frame_index - int(track.get("start_frame", 0))
    positions = track.get("positions", [])
    if offset < 0 or offset >= len(positions):
        return None
    return positions[offset]


def _intervals_contain(intervals: list[dict[str, int]], frame_index: int) -> bool:
    return any(
        interval["start_frame"] <= frame_index <= interval["end_frame"]
        for interval in intervals
    )


def _active_supplement_batches(batches: list[Any]) -> list[Any]:
    return [
        batch
        for batch in batches
        if getattr(batch, "status", "pending") != "withdrawn"
    ]


def _position_sources(
    raw_point: list[float] | None,
    target: str,
    frame_index: int,
    batches: list[Any],
    body_marker: str = "",
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    if raw_point is not None:
        sources.append(
            {
                "source": "raw_observation",
                "batch_id": None,
                "target": target,
                "position": raw_point,
            }
        )
    for batch in _active_supplement_batches(batches):
        point = _track_point(batch.tracks.get(target), frame_index)
        if point is None or not _intervals_contain(
            getattr(batch, "adopted_intervals", []), frame_index
        ):
            continue
        sources.append(
            {
                "source": batch.source,
                "batch_id": batch.batch_id,
                "target": target,
                "position": point,
            }
        )
    return sources


def evaluate_supplementary_batches(
    batches: list[Any],
    contacts: list[Any],
    candidates: list[Any],
    raw_observations: list[dict[str, Any]],
    objects: dict[str, list[list[float] | None]],
    *,
    tolerance: float = CONTACT_DISTANCE_TOLERANCE,
) -> dict[str, Any]:
    """Compare supplementary batches against still-missing contact frames.

    Supplementary data is review evidence only: it never feeds back into
    ``evaluate_contact_compatibility``, so old analyses keep the observations
    and gaps that were visible at the time, and a contact whose identity
    interpretation is already adopted (a confirmed manual decision) keeps its
    verdict even when a batch contradicts it.  Adopting the same frame from
    two batches does not double-count evidence; genuinely conflicting
    adopted batches are listed side by side.
    """
    active_contacts = [
        contact
        for contact in contacts
        if getattr(contact, "status", "active") == "active"
    ]
    base_review = evaluate_contact_compatibility(
        active_contacts, candidates, raw_observations, objects, tolerance=tolerance
    )
    missing_by_contact = {
        result["contact_id"]: result["missing_observation_frames"]
        for result in base_review["contact_results"]
    }
    confirmed_by_contact = {
        result["contact_id"]: result["candidate_id"]
        for result in base_review["contact_results"]
    }
    adopted_candidate_ids = {
        candidate.candidate_id
        for candidate in candidates
        if getattr(candidate, "decision", "open") == "adopted"
    }

    # frame_claims[(contact_id, frame)] lists every adopted-source combination
    # so cross-batch coverage, duplicates and unresolved conflicts retain their
    # provenance instead of being overwritten by batch arrival order.
    frame_claims: dict[tuple[str, int], list[dict[str, Any]]] = {}
    source_conflicts: list[dict[str, Any]] = []
    source_conflict_keys: set[tuple[Any, ...]] = set()
    batch_results: list[dict[str, Any]] = []
    for batch in batches:
        status = getattr(batch, "status", "pending")
        if status == "withdrawn":
            batch_results.append(
                {
                    "batch_id": batch.batch_id,
                    "object_id": batch.object_id,
                    "status": status,
                    "source": batch.source,
                    "reason": batch.reason,
                    "adopted_intervals": list(batch.adopted_intervals),
                    "contacts": [],
                    "supported_frames": [],
                    "contradicting_frames": [],
                    "pending_frames": [],
                    "still_missing_frames": [],
                    "observations_used": [],
                }
            )
            continue
        adopted_intervals = list(getattr(batch, "adopted_intervals", []))
        active_batches = _active_supplement_batches(batches)
        contact_results: list[dict[str, Any]] = []
        supported: list[int] = []
        contradicting: list[int] = []
        pending: list[int] = []
        still_missing: list[int] = []
        observations_used: list[dict[str, Any]] = []
        for contact in active_contacts:
            if contact.object_id != batch.object_id:
                continue
            candidate = candidate_for_contact(candidates, contact)
            body_marker = interpreted_body_marker(candidate, contact)
            object_track = batch.tracks.get(OBJECT_TARGET)
            body_track = batch.tracks.get(body_marker)
            contact_supported: list[int] = []
            contact_contradicting: list[dict[str, Any]] = []
            contact_pending: list[int] = []
            contact_missing: list[int] = []
            for frame_index in missing_by_contact.get(contact.contact_id, []):
                raw_object_point = (
                    objects.get(contact.object_id, [None] * (frame_index + 1))[
                        frame_index
                    ]
                    if frame_index < len(objects.get(contact.object_id, []))
                    else None
                )
                raw_body_point = (
                    raw_observations[frame_index].get(body_marker)
                    if frame_index < len(raw_observations)
                    else None
                )
                current_object_point = _track_point(object_track, frame_index)
                current_body_point = _track_point(body_track, frame_index)
                current_object_adopted = current_object_point is not None and _intervals_contain(
                    adopted_intervals, frame_index
                )
                current_body_adopted = current_body_point is not None and _intervals_contain(
                    adopted_intervals, frame_index
                )

                object_sources = _position_sources(
                    raw_object_point, OBJECT_TARGET, frame_index, active_batches
                )
                body_sources = _position_sources(
                    raw_body_point, body_marker, frame_index, active_batches
                )

                def record_target_conflicts(
                    target: str, sources: list[dict[str, Any]]
                ) -> None:
                    for first_index, first in enumerate(sources):
                        for second in sources[first_index + 1 :]:
                            if _distance(first["position"], second["position"]) <= tolerance:
                                continue
                            ordered_sources = tuple(
                                sorted(
                                    (
                                        (
                                            item["source"],
                                            item["batch_id"] or "",
                                            tuple(item["position"]),
                                        )
                                        for item in (first, second)
                                    ),
                                    key=lambda item: (item[1], item[0]),
                                )
                            )
                            conflict_key = (
                                contact.contact_id,
                                frame_index,
                                target,
                                ordered_sources,
                            )
                            if conflict_key in source_conflict_keys:
                                continue
                            source_conflict_keys.add(conflict_key)
                            source_conflicts.append(
                                {
                                    "contact_id": contact.contact_id,
                                    "frame": frame_index,
                                    "target": target,
                                    "sources": [
                                        {
                                            "source": item["source"],
                                            "batch_id": item["batch_id"],
                                            "position": item["position"],
                                        }
                                        for item in (first, second)
                                    ],
                                    "resolution": "unresolved",
                                }
                            )

                record_target_conflicts(OBJECT_TARGET, object_sources)
                record_target_conflicts(body_marker, body_sources)

                current_covers_missing_object = (
                    raw_object_point is None and current_object_point is not None
                )
                current_covers_missing_body = (
                    raw_body_point is None and current_body_point is not None
                )
                current_covers_gap = (
                    current_covers_missing_object or current_covers_missing_body
                )
                available_with_current = (
                    raw_object_point is not None or current_object_point is not None
                ) and (
                    raw_body_point is not None or current_body_point is not None
                )
                if not current_covers_gap:
                    continue
                if not available_with_current:
                    contact_missing.append(frame_index)
                    continue
                if not current_object_adopted and not current_body_adopted:
                    contact_pending.append(frame_index)
                    continue

                frame_used = False
                frame_verdicts: set[str] = set()
                for object_source in object_sources:
                    for body_source in body_sources:
                        if batch.batch_id not in {
                            object_source["batch_id"],
                            body_source["batch_id"],
                        }:
                            continue
                        gap = _distance(
                            body_source["position"], object_source["position"]
                        )
                        verdict = "supported" if gap <= tolerance else "contradicting"
                        claim = {
                            "batch_id": batch.batch_id,
                            "verdict": verdict,
                            "distance": gap,
                            "object_source": {
                                "source": object_source["source"],
                                "batch_id": object_source["batch_id"],
                            },
                            "body_source": {
                                "source": body_source["source"],
                                "batch_id": body_source["batch_id"],
                            },
                        }
                        frame_claims.setdefault(
                            (contact.contact_id, frame_index), []
                        ).append(claim)
                        observations_used.append(
                            {
                                "frame": frame_index,
                                "contact_id": contact.contact_id,
                                "batch_id": batch.batch_id,
                                "object_id": contact.object_id,
                                "object_target": OBJECT_TARGET,
                                "body_target": body_marker,
                                "distance": gap,
                                "verdict": verdict,
                                "source": batch.source,
                                "object_position_source": object_source["source"],
                                "object_position_batch_id": object_source["batch_id"],
                                "body_position_source": body_source["source"],
                                "body_position_batch_id": body_source["batch_id"],
                            }
                        )
                        frame_used = True
                        frame_verdicts.add(verdict)
                if not frame_used:
                    contact_missing.append(frame_index)
                elif "contradicting" in frame_verdicts and "supported" in frame_verdicts:
                    contact_supported.append(frame_index)
                    contact_contradicting.append(
                        {"frame": frame_index, "distance": None, "unresolved": True}
                    )
                elif "supported" in frame_verdicts:
                    contact_supported.append(frame_index)
                else:
                    contact_contradicting.append({"frame": frame_index, "distance": None})
            if not (
                contact_supported
                or contact_contradicting
                or contact_pending
                or contact_missing
            ):
                continue
            contact_candidate_id = confirmed_by_contact.get(contact.contact_id)
            protected = (
                contact_candidate_id is not None
                and contact_candidate_id in adopted_candidate_ids
            )
            contact_results.append(
                {
                    "contact_id": contact.contact_id,
                    "candidate_id": confirmed_by_contact.get(contact.contact_id),
                    "protected_by_confirmed_decision": protected,
                    "supported_frames": contact_supported,
                    "contradicting_frames": contact_contradicting,
                    "pending_frames": contact_pending,
                    "still_missing_frames": contact_missing,
                }
            )
            supported.extend(contact_supported)
            contradicting.extend(item["frame"] for item in contact_contradicting)
            pending.extend(contact_pending)
            still_missing.extend(contact_missing)
        batch_results.append(
            {
                "batch_id": batch.batch_id,
                "object_id": batch.object_id,
                "status": status,
                "source": batch.source,
                "reason": batch.reason,
                "adopted_intervals": adopted_intervals,
                "contacts": contact_results,
                "supported_frames": sorted(set(supported)),
                "contradicting_frames": sorted(set(contradicting)),
                "pending_frames": sorted(set(pending)),
                "still_missing_frames": sorted(set(still_missing)),
                "observations_used": observations_used,
            }
        )

    duplicate_adoptions: list[dict[str, Any]] = []
    conflicting_supplements: list[dict[str, Any]] = []
    for (contact_id, frame_index), claims in sorted(frame_claims.items()):
        contributing_batches = sorted(
            {
                batch_id
                for claim in claims
                for batch_id in (
                    claim["object_source"]["batch_id"],
                    claim["body_source"]["batch_id"],
                )
                if batch_id is not None
            }
        )
        if len(contributing_batches) > 1:
            duplicate_adoptions.append(
                {
                    "contact_id": contact_id,
                    "frame": frame_index,
                    "batch_ids": contributing_batches,
                    "note": "frame evidence counted once despite repeated adoption",
                }
            )
        verdicts = {claim["verdict"] for claim in claims}
        if len(verdicts) > 1:
            conflicting_supplements.append(
                {
                    "contact_id": contact_id,
                    "frame": frame_index,
                    "claims": claims,
                }
            )

    effective_missing: list[tuple[int, int]] = []
    combined_contact_results: list[dict[str, Any]] = []
    for contact in active_contacts:
        candidate = candidate_for_contact(candidates, contact)
        body_marker = interpreted_body_marker(candidate, contact)
        combined_supported: list[int] = []
        combined_contradicting: list[dict[str, Any]] = []
        combined_missing: list[int] = []
        combined_evidence: list[dict[str, Any]] = []
        for frame_index in missing_by_contact.get(contact.contact_id, []):
            raw_object_point = (
                objects.get(contact.object_id, [])[frame_index]
                if frame_index < len(objects.get(contact.object_id, []))
                else None
            )
            raw_body_point = (
                raw_observations[frame_index].get(body_marker)
                if frame_index < len(raw_observations)
                else None
            )
            adopted_object_available = raw_object_point is not None or any(
                getattr(batch, "status", "pending") != "withdrawn"
                and _track_point(batch.tracks.get(OBJECT_TARGET), frame_index) is not None
                and _intervals_contain(
                    getattr(batch, "adopted_intervals", []), frame_index
                )
                for batch in batches
            )
            adopted_body_available = raw_body_point is not None or any(
                getattr(batch, "status", "pending") != "withdrawn"
                and _track_point(batch.tracks.get(body_marker), frame_index) is not None
                and _intervals_contain(
                    getattr(batch, "adopted_intervals", []), frame_index
                )
                for batch in batches
            )
            if not adopted_object_available or not adopted_body_available:
                effective_missing.append((frame_index, frame_index))
                combined_missing.append(frame_index)
                continue
            object_sources = _position_sources(
                raw_object_point, OBJECT_TARGET, frame_index, batches
            )
            body_sources = _position_sources(
                raw_body_point, body_marker, frame_index, batches
            )
            frame_verdicts = set()
            for object_source in object_sources:
                for body_source in body_sources:
                    gap = _distance(
                        body_source["position"], object_source["position"]
                    )
                    verdict = "supported" if gap <= tolerance else "contradicting"
                    frame_verdicts.add(verdict)
                    combined_evidence.append(
                        {
                            "frame": frame_index,
                            "contact_id": contact.contact_id,
                            "object_id": contact.object_id,
                            "body_target": body_marker,
                            "distance": gap,
                            "verdict": verdict,
                            "object_source": {
                                "source": object_source["source"],
                                "batch_id": object_source["batch_id"],
                            },
                            "body_source": {
                                "source": body_source["source"],
                                "batch_id": body_source["batch_id"],
                            },
                        }
                    )
            if "supported" in frame_verdicts and "contradicting" in frame_verdicts:
                combined_supported.append(frame_index)
                combined_contradicting.append(
                    {"frame": frame_index, "distance": None, "unresolved": True}
                )
            elif "supported" in frame_verdicts:
                combined_supported.append(frame_index)
            else:
                combined_contradicting.append(
                    {"frame": frame_index, "distance": None}
                )
        combined_contact_results.append(
            {
                "contact_id": contact.contact_id,
                "candidate_id": confirmed_by_contact.get(contact.contact_id),
                "protected_by_confirmed_decision": (
                    confirmed_by_contact.get(contact.contact_id) in adopted_candidate_ids
                ),
                "supported_frames": combined_supported,
                "contradicting_frames": combined_contradicting,
                "still_missing_frames": combined_missing,
                "supported_intervals": _merge_intervals(
                    [(frame, frame) for frame in combined_supported]
                ),
                "conflict_intervals": _merge_intervals(
                    [
                        (item["frame"], item["frame"])
                        for item in combined_contradicting
                    ]
                ),
                "missing_intervals": _merge_intervals(
                    [(frame, frame) for frame in combined_missing]
                ),
                "observations_used": combined_evidence,
            }
        )

    return {
        "tolerance": tolerance,
        "batch_results": batch_results,
        "combined_contact_results": combined_contact_results,
        "duplicate_adoptions": duplicate_adoptions,
        "conflicting_supplements": conflicting_supplements,
        "source_conflicts": source_conflicts,
        "supported_intervals": _merge_intervals(
            [
                (frame, frame)
                for result in batch_results
                for frame in result["supported_frames"]
            ]
        ),
        "contradicting_intervals": _merge_intervals(
            [
                (frame, frame)
                for result in batch_results
                for frame in result["contradicting_frames"]
            ]
        ),
        "still_missing_intervals": _merge_intervals(
            effective_missing
        ),
    }
