from __future__ import annotations

from typing import Any

from .geometry import distance, mean
from .skeleton import BONES


def _joint_name(marker: str) -> str:
    if len(marker) > 3 and marker[0] == "p" and marker[2] == "_":
        return marker[3:]
    return marker


def bone_length_errors(
    processing: dict[str, Any],
    window: tuple[int, int] | None = None,
) -> dict[str, float]:
    """Mean absolute bone-length deviation per person over a window."""
    start, end = window or (0, len(processing["frames"]) - 1)
    totals: dict[str, list[float]] = {"A": [], "B": []}
    calibration = processing.get("calibration_lengths")
    for frame_index in range(start, end + 1):
        frame = processing["frames"][frame_index]
        persons = sorted({marker[1] for marker in frame if marker.startswith("p")})
        for person in persons:
            prefix = f"p{person}_"
            for bone_start, bone_end in BONES:
                a = frame.get(f"{prefix}{bone_start}")
                b = frame.get(f"{prefix}{bone_end}")
                if a is None or b is None:
                    continue
                expected = (
                    calibration.get(f"{bone_start}->{bone_end}", {}).get("length")
                    if calibration
                    else None
                )
                if expected is None:
                    continue
                totals.setdefault(person, []).append(abs(distance(a, b) - expected))
    return {
        f"person_{person}_mean_error": round(mean(values), 6) if values else 0.0
        for person, values in totals.items()
    }


def contact_frames(
    frames: list[dict[str, Any]],
    pair: tuple[str, str],
    *,
    threshold: float = 0.2,
) -> list[int]:
    a_marker, b_marker = pair
    contacts = []
    for frame_index, frame in enumerate(frames):
        a, b = frame.get(a_marker), frame.get(b_marker)
        if a is not None and b is not None and distance(a, b) <= threshold:
            contacts.append(frame_index)
    return contacts


def contact_recovery(
    processing: dict[str, Any],
    truth: list[dict[str, Any]],
    pair: tuple[str, str],
    *,
    threshold: float = 0.2,
) -> dict[str, Any]:
    truth_contacts = set(contact_frames(truth, pair, threshold=threshold))
    recovered_contacts = set(
        contact_frames(processing["frames"], pair, threshold=threshold)
    )
    missing = sorted(truth_contacts - recovered_contacts)
    spurious = sorted(recovered_contacts - truth_contacts)
    return {
        "pair": list(pair),
        "threshold": threshold,
        "truth_contact_frames": sorted(truth_contacts),
        "reconstructed_contact_frames": sorted(recovered_contacts),
        "missing_contact_frames": missing,
        "spurious_contact_frames": spurious,
        "precision": (
            len(recovered_contacts & truth_contacts) / len(recovered_contacts)
            if recovered_contacts
            else 1.0
        ),
        "recall": (
            len(recovered_contacts & truth_contacts) / len(truth_contacts)
            if truth_contacts
            else 1.0
        ),
    }


def identity_continuity(
    processing: dict[str, Any],
    truth: list[dict[str, Any]],
    marker_groups: list[tuple[str, str]],
    *,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Fraction of frames where each marker sits at its own ground-truth joint.

    Swapped labels and uncertain gaps break continuity; recovered short
    occlusions that land near truth preserve it.
    """
    correct = 0
    evaluated = 0
    breaks: list[dict[str, Any]] = []
    for marker, _other in marker_groups:
        joint = _joint_name(marker)
        for frame_index, frame in enumerate(processing["frames"]):
            actual = truth[frame_index].get(marker)
            value = frame.get(marker)
            status = processing["statuses"][frame_index].get(marker)
            if actual is None:
                continue
            evaluated += 1
            if value is not None and distance(value, actual) <= tolerance:
                correct += 1
            else:
                breaks.append(
                    {
                        "frame": frame_index,
                        "marker": marker,
                        "status": status,
                    }
                )
    return {
        "marker_groups": [list(group) for group in marker_groups],
        "continuity_ratio": round(correct / evaluated, 6) if evaluated else 1.0,
        "break_count": len(breaks),
        "break_frames": sorted({item["frame"] for item in breaks}),
        "breaks": breaks,
    }
