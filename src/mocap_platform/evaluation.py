from __future__ import annotations

import math
from typing import Any

from .geometry import distance
from .identity import person_marker, split_marker
from .skeleton import BONES, JOINTS


def evaluate(processing: dict[str, Any], ground_truth: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[float] = []
    swap_errors: list[float] = []
    observed_count = recovered_count = uncertain_count = 0
    uncertain_truth = 0

    for frame_index, (frame, truth) in enumerate(zip(processing["frames"], ground_truth)):
        for marker, reconstructed in frame.items():
            status = processing["statuses"][frame_index][marker]
            actual = truth.get(marker)
            if actual is None:
                continue
            if status == "observed":
                observed_count += 1
            elif status == "recovered":
                recovered_count += 1
                errors.append(distance(reconstructed, actual))
            elif status == "uncertain":
                uncertain_count += 1
                uncertain_truth += 1
            if status == "observed" and reconstructed is not None:
                identity_status = None
                if processing.get("identity_statuses"):
                    identity_status = processing["identity_statuses"][frame_index].get(marker)
                if identity_status in {"ambiguous", "conflicted"}:
                    continue
                error = distance(reconstructed, actual)
                if error > 0.02:
                    swap_errors.append(error)

    rmse = math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None
    return {
        "version": processing["version"],
        "observed_marker_frames": observed_count,
        "recovered_marker_frames": recovered_count,
        "uncertain_marker_frames": uncertain_count,
        "recovered_rmse": rmse,
        "recovered_max_error": max(errors, default=None),
        "identity_mismatch_count": len(swap_errors),
        "identity_mismatch_rmse": math.sqrt(
            sum(error * error for error in swap_errors) / len(swap_errors)
        )
        if swap_errors
        else 0.0,
        "uncertain_had_ground_truth_frames": uncertain_truth,
        "no_fabricated_long_gap": uncertain_truth >= 10,
    }


CONTACT_MARKERS = ("l_wrist", "r_wrist", "l_ankle", "r_ankle", "head")
CONTACT_THRESHOLD = 0.15


def _people(frames: list[dict[str, Any]]) -> list[str | None]:
    people: set[str | None] = {None}
    for frame in frames:
        for marker in frame:
            person_id, _joint = split_marker(marker)
            people.add(person_id)
    return sorted(person for person in people if person is not None) or [None]


def _marker_name(person_id: str | None, joint_name: str) -> str:
    return joint_name if person_id is None else person_marker(person_id, joint_name)


def _bone_length_errors(processing: dict[str, Any], ground_truth: list[dict[str, Any]]) -> list[float]:
    errors: list[float] = []
    people = _people(ground_truth)
    for frame, truth in zip(processing["frames"], ground_truth):
        for person_id in people:
            for start_joint, end_joint in BONES:
                start_name = _marker_name(person_id, start_joint)
                end_name = _marker_name(person_id, end_joint)
                start, end = frame.get(start_name), frame.get(end_name)
                truth_start, truth_end = truth.get(start_name), truth.get(end_name)
                if None in (start, end, truth_start, truth_end):
                    continue
                errors.append(abs(distance(start, end) - distance(truth_start, truth_end)))
    return errors


def _contact_frames(frames: list[dict[str, Any]], people: list[str | None]) -> list[int]:
    contact_frames: list[int] = []
    for frame_index, frame in enumerate(frames):
        contact = False
        for person_index, first_person in enumerate(people):
            for second_person in people[person_index + 1 :]:
                for first_joint in CONTACT_MARKERS:
                    for second_joint in CONTACT_MARKERS:
                        first = frame.get(_marker_name(first_person, first_joint))
                        second = frame.get(_marker_name(second_person, second_joint))
                        if first is not None and second is not None and distance(first, second) <= CONTACT_THRESHOLD:
                            contact = True
        if contact:
            contact_frames.append(frame_index)
    return contact_frames


def _identity_continuity_rmse(
    processing: dict[str, Any], ground_truth: list[dict[str, Any]], people: list[str | None]
) -> float | None:
    errors: list[float] = []
    identity_statuses = processing.get("identity_statuses", [])
    unresolved = {"ambiguous", "conflicted"}
    for person_id in people:
        joints = JOINTS if person_id is not None else list(ground_truth[0])
        for joint_name in joints:
            marker = _marker_name(person_id, joint_name)
            for frame_index in range(1, len(processing["frames"])):
                current = processing["frames"][frame_index].get(marker)
                previous = processing["frames"][frame_index - 1].get(marker)
                truth_current = ground_truth[frame_index].get(marker)
                truth_previous = ground_truth[frame_index - 1].get(marker)
                if None in (current, previous, truth_current, truth_previous):
                    continue
                if identity_statuses:
                    if identity_statuses[frame_index].get(marker) in unresolved:
                        continue
                    if identity_statuses[frame_index - 1].get(marker) in unresolved:
                        continue
                actual_delta = distance(current, previous)
                expected_delta = distance(truth_current, truth_previous)
                errors.append(abs(actual_delta - expected_delta))
    if not errors:
        return None
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def evaluate_multi_person(
    processing: dict[str, Any], ground_truth: list[dict[str, Any]]
) -> dict[str, Any]:
    summary = evaluate(processing, ground_truth)
    people = _people(ground_truth)
    identity_statuses = processing.get("identity_statuses", [])
    unresolved = {"ambiguous", "conflicted"}
    ambiguous = sum(
        sum(status in unresolved for status in frame.values()) for frame in identity_statuses
    )
    bone_errors = _bone_length_errors(processing, ground_truth)
    contact_frames = _contact_frames(processing["frames"], people)
    summary.update(
        {
            "bone_length_rmse": math.sqrt(
                sum(error * error for error in bone_errors) / len(bone_errors)
            )
            if bone_errors
            else None,
            "contact_frame_count": len(contact_frames),
            "contact_frames": contact_frames,
            "ambiguous_identity_marker_frames": ambiguous,
            "identity_continuity_rmse": _identity_continuity_rmse(
                processing, ground_truth, people
            ),
        }
    )
    return summary


def compare_identity_revisions(
    versions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]
) -> dict[str, Any]:
    summaries = [evaluate_multi_person(version, ground_truth) for version in versions]
    return {
        "versions": [
            {
                "version": version["version"],
                "based_on": version.get("based_on"),
                "decisions": version.get("identity_decisions", []),
                "metrics": summary,
            }
            for version, summary in zip(versions, summaries)
        ],
        "bone_length_rmse_improvement": summaries[0]["bone_length_rmse"] - summaries[-1]["bone_length_rmse"]
        if summaries[0]["bone_length_rmse"] is not None and summaries[-1]["bone_length_rmse"] is not None
        else None,
        "contact_frame_count_preserved": len(
            {tuple(summary["contact_frames"]) for summary in summaries}
        )
        == 1,
    }
