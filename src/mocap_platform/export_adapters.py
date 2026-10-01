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
    }
