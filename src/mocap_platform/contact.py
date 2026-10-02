from __future__ import annotations

from typing import Any

from .geometry import distance
from .identity import person_marker, resolve_decisions
from .models import ContactEvidence, Frame, InterpretationScenario
from .reassignment import resolve_reassignments


def _contiguous_intervals(frames: list[int]) -> list[dict[str, int]]:
    intervals: list[dict[str, int]] = []
    start: int | None = None
    previous: int | None = None
    for frame_index in sorted(frames):
        if start is None:
            start = previous = frame_index
            continue
        if frame_index == previous + 1:
            previous = frame_index
            continue
        intervals.append({"start_frame": start, "end_frame": previous})
        start = previous = frame_index
    if start is not None:
        intervals.append({"start_frame": start, "end_frame": previous})
    return intervals


def evaluate_contact(
    contact: ContactEvidence,
    frames: list[Frame],
    statuses: list[dict[str, str]],
    object_frames: list[Frame],
) -> dict[str, Any]:
    marker = person_marker(contact.person_id, contact.joint)
    compatible_frames: list[int] = []
    incompatible_frames: list[int] = []
    missing_frames: list[int] = []
    used_observations: list[dict[str, Any]] = []
    for frame_index in range(contact.start_frame, contact.end_frame + 1):
        position = frames[frame_index].get(marker) if frame_index < len(frames) else None
        marker_status = (
            statuses[frame_index].get(marker) if frame_index < len(statuses) else None
        )
        object_position = (
            object_frames[frame_index].get(contact.object_id)
            if frame_index < len(object_frames)
            else None
        )
        # Only genuine observations count as evidence; interpolated or
        # uncertain positions never make an explanation look certain.
        if marker_status != "observed" or position is None or object_position is None:
            missing_frames.append(frame_index)
            continue
        gap = distance(position, object_position)
        used_observations.append(
            {
                "frame": frame_index,
                "marker": marker,
                "object_id": contact.object_id,
                "distance": gap,
            }
        )
        if gap <= contact.tolerance:
            compatible_frames.append(frame_index)
        else:
            incompatible_frames.append(frame_index)
    if incompatible_frames:
        status = "incompatible"
    elif compatible_frames:
        status = "compatible"
    else:
        status = "unobserved"
    return {
        "contact_id": contact.contact_id,
        "person_id": contact.person_id,
        "joint": contact.joint,
        "object_id": contact.object_id,
        "start_frame": contact.start_frame,
        "end_frame": contact.end_frame,
        "status": status,
        "conflict_intervals": _contiguous_intervals(incompatible_frames),
        "compatible_frames": compatible_frames,
        "missing_frames": missing_frames,
        "used_observations": used_observations,
    }


def contact_assessments(
    contacts: list[ContactEvidence],
    frames: list[Frame],
    statuses: list[dict[str, str]],
    object_frames: list[Frame],
) -> list[dict[str, Any]]:
    assessments: list[dict[str, Any]] = []
    for contact in contacts:
        if contact.status == "revoked":
            assessments.append(
                {
                    "contact_id": contact.contact_id,
                    "person_id": contact.person_id,
                    "joint": contact.joint,
                    "object_id": contact.object_id,
                    "start_frame": contact.start_frame,
                    "end_frame": contact.end_frame,
                    "status": "revoked",
                    "conflict_intervals": [],
                    "compatible_frames": [],
                    "missing_frames": [],
                    "used_observations": [],
                }
            )
            continue
        assessments.append(evaluate_contact(contact, frames, statuses, object_frames))
    return assessments


def simultaneous_contacts(contacts: list[ContactEvidence]) -> list[dict[str, Any]]:
    active = [contact for contact in contacts if contact.status == "active"]
    groups: list[dict[str, Any]] = []
    for first_index, first in enumerate(active):
        for second in active[first_index + 1 :]:
            if first.object_id != second.object_id:
                continue
            if first.person_id == second.person_id:
                continue
            start = max(first.start_frame, second.start_frame)
            end = min(first.end_frame, second.end_frame)
            if start > end:
                continue
            groups.append(
                {
                    "object_id": first.object_id,
                    "contact_ids": sorted([first.contact_id, second.contact_id]),
                    "person_ids": sorted([first.person_id, second.person_id]),
                    "start_frame": start,
                    "end_frame": end,
                    "allowed": True,
                    "reason": (
                        "two people may touch the same object at the same time; "
                        "this does not merge their trajectories"
                    ),
                }
            )
    return groups


def review_scenario(
    scenario: InterpretationScenario,
    contacts: list[ContactEvidence],
    decisions: list[Any],
    reassignments: list[Any],
    frames: list[Frame],
    statuses: list[dict[str, str]],
    object_frames: list[Frame],
) -> dict[str, Any]:
    adopted_reassignments = [
        reassignment
        for reassignment in reassignments
        if reassignment.reassignment_id in scenario.reassignment_ids
    ]
    adopted_decisions = [
        decision
        for decision in decisions
        if decision.decision_id in scenario.decision_ids
    ]
    used_contacts = [
        contact
        for contact in contacts
        if contact.contact_id in scenario.contact_ids and contact.status == "active"
    ]
    assessments = [
        evaluate_contact(contact, frames, statuses, object_frames)
        for contact in used_contacts
    ]
    _applied, disputes, _disputed_ids = resolve_reassignments(adopted_reassignments)
    _accepted, decision_conflicts = resolve_decisions(adopted_decisions)
    claim_conflicts = [
        {
            "kind": "reassignment_dispute",
            "reassignment_id": dispute["reassignment_id"],
            "disputes_with": dispute["disputes_with"],
            "start_frame": dispute["start_frame"],
            "end_frame": dispute["end_frame"],
            "markers": dispute["markers"],
            "reason": (
                "the same body trajectory is claimed by two identities; "
                "simultaneous contact with one object never permits this"
            ),
        }
        for dispute in disputes
    ]
    claim_conflicts += [
        {
            "kind": "decision_conflict",
            "decision_id": conflict["decision_id"],
            "conflicts_with": conflict["conflicts_with"],
            "start_frame": conflict["start_frame"],
            "end_frame": conflict["end_frame"],
            "markers": conflict["markers"],
            "reason": conflict["reason"],
        }
        for conflict in decision_conflicts
    ]
    if claim_conflicts:
        overall = "contested"
    elif any(assessment["status"] == "incompatible" for assessment in assessments):
        overall = "incompatible"
    elif assessments and all(
        assessment["status"] == "unobserved" for assessment in assessments
    ):
        overall = "unobserved"
    else:
        overall = "compatible"
    return {
        "scenario_id": scenario.scenario_id,
        "label": scenario.label,
        "scenario_status": scenario.status,
        "automatic_assessment": overall,
        "note": (
            "automatic assessment is only a recommendation; it never counts "
            "as manual confirmation"
        ),
        "contact_assessments": assessments,
        "trajectory_claim_conflicts": claim_conflicts,
        "simultaneous_contacts": simultaneous_contacts(used_contacts),
        "excluded_candidates": scenario.excluded_candidates,
    }


def affected_scenarios(
    scenarios: list[InterpretationScenario],
    *,
    changed_contact_ids: set[str] | None = None,
    changed_decision_ids: set[str] | None = None,
    changed_reassignment_ids: set[str] | None = None,
) -> list[str]:
    changed_contact_ids = changed_contact_ids or set()
    changed_decision_ids = changed_decision_ids or set()
    changed_reassignment_ids = changed_reassignment_ids or set()
    affected: list[str] = []
    for scenario in scenarios:
        if (
            set(scenario.contact_ids) & changed_contact_ids
            or set(scenario.decision_ids) & changed_decision_ids
            or set(scenario.reassignment_ids) & changed_reassignment_ids
        ):
            affected.append(scenario.scenario_id)
    return affected
