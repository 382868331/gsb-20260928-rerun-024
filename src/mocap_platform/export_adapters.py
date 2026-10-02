from __future__ import annotations

from typing import Any


# Placeholder for BVH/FBX/CSV adapters in a later milestone.
def supported_adapters() -> tuple[str, ...]:
    return ()


def build_review_export(
    capture: dict[str, Any],
    processing: dict[str, Any],
    decisions: list[Any],
    reassignments: list[Any],
    contacts: list[Any] | None = None,
    candidates: list[Any] | None = None,
    schemes: list[Any] | None = None,
    object_tracks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    impacts = {
        item["decision_id"]: item["status"]
        for item in processing.get("parameters", {}).get("decision_impacts", [])
    }
    reassignment_states = {
        item["reassignment_id"]: item["status"]
        for item in processing.get("parameters", {}).get("reassignment_impacts", [])
    }
    contact_review = processing.get("parameters", {}).get("contact_review", {})
    contact_results = {
        item["contact_id"]: item
        for item in contact_review.get("contact_results", [])
    }
    candidate_review = contact_review.get("claim_conflicts", [])
    conflicting_candidates = {
        candidate_id
        for item in candidate_review
        for candidate_id in item.get("candidate_ids", [])
    }
    return {
        "kind": "review_export",
        "version": 1,
        "capture_id": capture["capture_id"],
        "original_capture": {
            "action_name": capture["action_name"],
            "frame_count": capture["frame_count"],
            "frame_rate": capture["frame_rate"],
            "markers": capture["markers"],
            "observations": capture["observations"],
        },
        "processing_version": processing["version"],
        "reassignment_ranges": [
            {
                "reassignment_id": reassignment.reassignment_id,
                "source_person_id": reassignment.source_person_id,
                "target_person_id": reassignment.target_person_id,
                "joints": reassignment.joints,
                "target_joints": reassignment.target_joints or reassignment.joints,
                "start_frame": reassignment.start_frame,
                "end_frame": reassignment.end_frame,
                "status": reassignment_states.get(
                    reassignment.reassignment_id, reassignment.status
                ),
                "kept_decision_ids": reassignment.kept_decision_ids,
                "reason": reassignment.reason,
            }
            for reassignment in reassignments
        ],
        "decision_provenance": [
            {
                "decision_id": decision.decision_id,
                "decision": decision.decision,
                "person_id": decision.person_id,
                "alternative_person_id": decision.alternative_person_id,
                "marker": decision.marker,
                "start_frame": decision.start_frame,
                "end_frame": decision.end_frame,
                "status": impacts.get(decision.decision_id, decision.status),
                "source": "manual_review",
                "depends_on": decision.depends_on,
                "reason": decision.reason,
            }
            for decision in decisions
        ],
        "object_tracks": object_tracks or {},
        "manual_contact_notes": [
            {
                "contact_id": contact.contact_id,
                "person_id": contact.person_id,
                "body_part": contact.body_part,
                "object_id": contact.object_id,
                "start_frame": contact.start_frame,
                "end_frame": contact.end_frame,
                "source": contact.source,
                "status": contact.status,
                "reason": contact.reason,
                "simultaneous_with": contact.simultaneous_with,
                "compatibility": contact_results.get(contact.contact_id),
            }
            for contact in (contacts or [])
        ],
        "identity_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "person_id": candidate.person_id,
                "joint_mapping": candidate.joint_mapping,
                "start_frame": candidate.start_frame,
                "end_frame": candidate.end_frame,
                "decision": candidate.decision,
                "decision_reason": getattr(candidate, "decision_reason", ""),
                "depends_on": candidate.depends_on,
                "reason": candidate.reason,
                "claim_conflict": candidate.candidate_id in conflicting_candidates,
            }
            for candidate in (candidates or [])
        ],
        "identity_schemes": [
            {
                "scheme_id": scheme.scheme_id,
                "label": scheme.label,
                "status": scheme.status,
                "candidate_ids": list(scheme.candidate_ids),
                "adopted_reassignment_ids": list(scheme.adopted_reassignment_ids),
                "crossing_start_frame": scheme.crossing_start_frame,
                "crossing_end_frame": scheme.crossing_end_frame,
                "notes": scheme.notes,
            }
            for scheme in (schemes or [])
        ],
        "contact_review": contact_review,
        "simultaneous_contacts": contact_review.get("simultaneous_contacts", []),
    }
