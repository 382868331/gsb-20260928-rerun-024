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
    scenarios: list[Any] | None = None,
) -> dict[str, Any]:
    impacts = {
        item["decision_id"]: item["status"]
        for item in processing.get("parameters", {}).get("decision_impacts", [])
    }
    reassignment_states = {
        item["reassignment_id"]: item["status"]
        for item in processing.get("parameters", {}).get("reassignment_impacts", [])
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
            "objects": capture.get("objects", []),
            "observations": capture["observations"],
            "object_frames": capture.get("object_frames", []),
        },
        "processing_version": processing["version"],
        "contact_evidence": [
            {
                "contact_id": contact.contact_id,
                "person_id": contact.person_id,
                "joint": contact.joint,
                "object_id": contact.object_id,
                "start_frame": contact.start_frame,
                "end_frame": contact.end_frame,
                "source": contact.source,
                "status": contact.status,
                "reason": contact.reason,
            }
            for contact in (contacts or [])
        ],
        "identity_scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "label": scenario.label,
                "status": scenario.status,
                "decision_ids": scenario.decision_ids,
                "reassignment_ids": scenario.reassignment_ids,
                "contact_ids": scenario.contact_ids,
                "excluded_candidates": scenario.excluded_candidates,
                "reason": scenario.reason,
            }
            for scenario in (scenarios or [])
        ],
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
    }
