from __future__ import annotations

from .geometry import Vec3
from .model import CalibrationRecord, CaptureRecord, Observation, SkeletonDefinition


def average_bone_lengths(
    observations: list[Observation],
    frame_start: int,
    frame_end: int,
    bones: tuple[tuple[str, str], ...],
) -> dict[tuple[str, str], float]:
    sums: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    rows: dict[int, dict[str, Vec3]] = {}
    for item in observations:
        if frame_start <= item.frame <= frame_end:
            rows.setdefault(item.frame, {})[item.marker] = item.position

    for row in rows.values():
        for parent, child in bones:
            if parent in row and child in row:
                key = (parent, child)
                sums[key] = sums.get(key, 0.0) + row[parent].distance(row[child])
                counts[key] = counts.get(key, 0) + 1

    missing = [bone for bone in bones if counts.get(bone, 0) == 0]
    if missing:
        raise ValueError(f"cannot calibrate bones without observations: {missing}")
    return {bone: sums[bone] / counts[bone] for bone in bones}


def calibrate(
    calibration_id: str,
    capture: CaptureRecord,
    capture_version: str,
    joints: tuple[str, ...],
    root: str,
    bones: tuple[tuple[str, str], ...],
    frame_start: int,
    frame_end: int,
    joint_limits: dict[tuple[str, str, str], tuple[float, float]] | None = None,
) -> CalibrationRecord:
    lengths = average_bone_lengths(list(capture.observations), frame_start, frame_end, bones)
    skeleton = SkeletonDefinition(joints, root, bones, lengths, joint_limits or {})
    return CalibrationRecord(calibration_id, capture_version, skeleton, frame_start, frame_end)
