from __future__ import annotations

import tempfile
from math import pi
from pathlib import Path

from .calibration import calibrate
from .geometry import Vec3
from .model import CaptureRecord, IdentityRevision, Observation
from .processing import ProcessingConfig, rmse
from .storage import WorkspaceStore


JOINTS = ("pelvis", "l_elbow", "l_wrist", "r_elbow", "r_wrist")
BONES = (
    ("pelvis", "l_elbow"),
    ("l_elbow", "l_wrist"),
    ("pelvis", "r_elbow"),
    ("r_elbow", "r_wrist"),
)

JOINT_LIMITS = {
    ("pelvis", "l_elbow", "l_wrist"): (0.0, pi / 2),
    ("pelvis", "r_elbow", "r_wrist"): (0.0, pi / 2),
}


def truth_motion(frame_count: int = 17) -> list[dict[str, Vec3]]:
    frames: list[dict[str, Vec3]] = []
    for frame in range(frame_count):
        phase = frame / (frame_count - 1)
        pelvis = Vec3(0.0, 1.0, 0.0)
        l_elbow = Vec3(-0.5, 1.1 + 0.05 * phase, 0.05 * phase)
        r_elbow = Vec3(0.5, 1.1 - 0.05 * phase, -0.05 * phase)
        l_wrist = Vec3(-0.9, 1.25 + 0.15 * phase, 0.05)
        r_wrist = Vec3(0.9, 1.25 - 0.15 * phase, -0.05)
        frames.append(
            {
                "pelvis": pelvis,
                "l_elbow": l_elbow,
                "l_wrist": l_wrist,
                "r_elbow": r_elbow,
                "r_wrist": r_wrist,
            }
        )
    return frames


def make_observations(truth: list[dict[str, Vec3]]) -> list[Observation]:
    observations: list[Observation] = []
    for frame, row in enumerate(truth):
        for joint, position in row.items():
            if joint == "l_wrist" and frame in {3, 4}:
                continue
            if joint == "r_wrist" and 10 <= frame <= 16:
                continue
            if 6 <= frame <= 8:
                if joint == "l_elbow":
                    marker = "r_elbow"
                elif joint == "r_elbow":
                    marker = "l_elbow"
                elif joint == "l_wrist":
                    marker = "r_wrist"
                elif joint == "r_wrist":
                    marker = "l_wrist"
                else:
                    marker = joint
            else:
                marker = joint
            observations.append(Observation(frame, marker, position))
    return observations


def run_demo() -> dict[str, object]:
    truth = truth_motion()
    observations = make_observations(truth)
    capture = CaptureRecord("synthetic-wave", 60.0, tuple(observations), {"source": "synthetic"})
    config = ProcessingConfig(max_short_gap=2, bone_tolerance=0.05)

    with tempfile.TemporaryDirectory() as directory:
        store = WorkspaceStore(Path(directory))
        capture_version = store.save_capture(capture)

        clean_capture = CaptureRecord("synthetic-wave-clean", 60.0, tuple(
            Observation(frame, joint, position)
            for frame, row in enumerate(truth)
            for joint, position in row.items()
        ))
        clean_version = store.save_capture(clean_capture)
        calibration_record = calibrate(
            "synthetic-calibration",
            clean_capture,
            clean_version,
            JOINTS,
            "pelvis",
            BONES,
            0,
            len(truth) - 1,
            JOINT_LIMITS,
        )
        calibration_version = store.save_calibration(calibration_record)

        first_version = store.create_processing(
            "baseline",
            capture_version,
            calibration_version,
            config=config,
        )
        first = store.get_processing(first_version)

        revisions = (
            IdentityRevision(
                "fix-arm-swap",
                6,
                8,
                {
                    "l_elbow": "r_elbow",
                    "r_elbow": "l_elbow",
                    "l_wrist": "r_wrist",
                    "r_wrist": "l_wrist",
                },
                "manual correction of swapped arm markers",
            ),
        )
        revised_version = store.create_processing(
            "identity-revised",
            capture_version,
            calibration_version,
            revisions,
            config,
            first_version,
        )
        revised = store.get_processing(revised_version)

        result = {
            "store": directory,
            "capture_version": capture_version,
            "calibration_version": calibration_version,
            "baseline_version": first_version,
            "revised_version": revised_version,
            "baseline_unknown_frames": list(first.unknown_frames),
            "revised_unknown_frames": list(revised.unknown_frames),
            "baseline_rmse": rmse(list(first.frames), truth),
            "revised_rmse": rmse(list(revised.frames), truth),
            "warnings": list(revised.warnings),
        }

    return result


if __name__ == "__main__":
    for key, value in run_demo().items():
        if key != "store":
            print(f"{key}: {value}")
