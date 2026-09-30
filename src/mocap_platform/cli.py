from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import calibrate_from_action
from .metrics import bone_length_errors, contact_recovery, identity_continuity
from .models import Calibration, CaptureRecord, IdentityRevision
from .reconstruction import reconstruct
from .storage import PlatformStore
from .synthetic import (
    generate_two_person_action,
    inject_two_person_events,
)

CONTACT_PAIRS = [
    ("pA_r_wrist", "pB_l_wrist"),
    ("pA_r_wrist", "pB_r_shoulder"),
]
IDENTITY_GROUPS = [
    ("pA_r_shoulder", "pB_l_shoulder"),
    ("pA_r_elbow", "pB_l_elbow"),
    ("pA_r_wrist", "pB_l_wrist"),
]
CROSS_WINDOW = (18, 31)


def _version_metrics(result, truth, calibration) -> dict:
    document = result.to_dict()
    document["calibration_lengths"] = calibration["bone_lengths"]
    metrics = {
        "bone_length_errors": bone_length_errors(document, CROSS_WINDOW),
        "identity_continuity": identity_continuity(document, truth, IDENTITY_GROUPS),
        "contacts": {
            f"{a}__{b}": contact_recovery(document, truth, (a, b), threshold=0.15)
            for a, b in CONTACT_PAIRS
        },
    }
    metrics["unresolved_ambiguity_frames"] = [
        {
            "groups": ambiguity["groups"],
            "start_frame": ambiguity["start_frame"],
            "end_frame": ambiguity["end_frame"],
            "resolution": ambiguity["resolution"],
        }
        for ambiguity in document["ambiguities"]
        if ambiguity["resolution"] == "unresolved"
    ]
    return metrics


def demo(output: str) -> None:
    store = PlatformStore(Path(output))
    action_name = "demo_action"
    pipeline_name = "two_person_crossing"
    capture_id = "demo_capture_0001"

    truth = generate_two_person_action()
    observed, events = inject_two_person_events(truth)
    markers = list(truth[0])
    capture = CaptureRecord(
        capture_id=capture_id,
        action_name=action_name,
        frame_count=len(observed),
        frame_rate=30.0,
        markers=markers,
        observed_frames=observed,
        ground_truth_frames=truth,
        source="synthetic_known_action",
        notes=(
            "Two-person crossing take: arm identity cross, simultaneous short "
            "occlusion, symmetric reappearance and a long head gap. Raw "
            "observations are immutable; identity revisions only create new "
            "processing versions."
        ),
    )
    store.save_capture(capture)

    bone_lengths, joint_limits = calibrate_from_action(truth)
    calibration = Calibration(
        calibration_id="calibration_demo_0001",
        capture_id=capture_id,
        bone_lengths=bone_lengths,
        joint_limits=joint_limits,
    )
    store.save_calibration(calibration)

    v1 = reconstruct(
        action_name,
        pipeline_name,
        capture_id,
        observed,
        calibration.to_dict(),
        max_gap_frames=5,
        detect_ambiguity=True,
    )
    store.save_processing(v1)

    deny_revision = IdentityRevision(
        revision_id="rev_deny_cross_01",
        marker="pA_r_wrist",
        start_frame=18,
        end_frame=23,
        corrected_identity="pB_l_arm",
        decision="deny",
        group="r_arm",
        propagated=True,
        reason="manual review: pA right arm markers follow person B through the cross",
    )
    confirm_revision = IdentityRevision(
        revision_id="rev_confirm_reappear_01",
        marker="pA_r_wrist",
        start_frame=28,
        end_frame=31,
        corrected_identity="pA_r_arm",
        decision="confirm",
        group="r_arm",
        propagated=True,
        reason="manual review: after occlusion the near-side arm belongs to A",
    )
    v2 = reconstruct(
        action_name,
        pipeline_name,
        capture_id,
        observed,
        calibration.to_dict(),
        max_gap_frames=5,
        revisions=[deny_revision, confirm_revision],
        based_on=v1,
        detect_ambiguity=True,
    )
    store.save_processing(v2)

    # Undo the reappearance confirmation: only that window must regress to
    # uncertain; the denied cross stays corrected.
    v3 = reconstruct(
        action_name,
        pipeline_name,
        capture_id,
        observed,
        calibration.to_dict(),
        max_gap_frames=5,
        revisions=[],
        based_on=v2,
        revoked_revision_ids={"rev_confirm_reappear_01"},
        detect_ambiguity=True,
    )
    store.save_processing(v3)

    report = {
        "capture_id": capture_id,
        "injected_events": events,
        "contact_pairs": [list(pair) for pair in CONTACT_PAIRS],
        "identity_groups": [list(group) for group in IDENTITY_GROUPS],
        "v1_before_revision": _version_metrics(v1, truth, calibration.to_dict()),
        "v2_after_confirmation": _version_metrics(v2, truth, calibration.to_dict()),
        "v3_after_undo": _version_metrics(v3, truth, calibration.to_dict()),
    }
    store.save_report(action_name, pipeline_name, report)

    continuity_before = report["v1_before_revision"]["identity_continuity"]["continuity_ratio"]
    continuity_after = report["v2_after_confirmation"]["identity_continuity"]["continuity_ratio"]
    unresolved_v1 = len(
        report["v1_before_revision"]["unresolved_ambiguity_frames"]
    )
    unresolved_v3 = len(report["v3_after_undo"]["unresolved_ambiguity_frames"])
    print(f"saved {output}")
    print(f"v1 continuity={continuity_before:.3f} unresolved_ambiguities={unresolved_v1}")
    print(f"v2 continuity={continuity_after:.3f}")
    print(f"v3 unresolved_ambiguities={unresolved_v3} (confirmation revoked)")



def main() -> None:
    parser = argparse.ArgumentParser(prog="mocap-platform")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo_parser = subparsers.add_parser("demo", help="create synthetic evaluation dataset")
    demo_parser.add_argument("--output", default="data", help="data root directory")
    args = parser.parse_args()
    if args.command == "demo":
        demo(args.output)


if __name__ == "__main__":
    main()
