from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import calibrate_from_action
from .evaluation import evaluate
from .models import Calibration, CaptureRecord, IdentityRevision
from .reconstruction import reconstruct
from .storage import PlatformStore
from .synthetic import generate_known_action, inject_occlusions_and_swaps


def demo(output: str) -> None:
    store = PlatformStore(Path(output))
    action_name = "demo_action"
    pipeline_name = "identity_swap_demo"
    capture_id = "demo_capture_0001"

    truth = generate_known_action()
    observed, events = inject_occlusions_and_swaps(truth)
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
        notes="Synthetic occlusion and marker swap fixture; ground truth retained for evaluation.",
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
    )
    store.save_processing(v1)

    revisions = [
        IdentityRevision(
            marker="l_wrist",
            start_frame=30,
            end_frame=35,
            corrected_identity="r_wrist",
            reason="manual review: left/right wrist labels exchanged",
        ),
        IdentityRevision(
            marker="r_wrist",
            start_frame=30,
            end_frame=35,
            corrected_identity="l_wrist",
            reason="manual review: reciprocal wrist label exchange",
        ),
    ]
    v2 = reconstruct(
        action_name,
        pipeline_name,
        capture_id,
        observed,
        calibration.to_dict(),
        max_gap_frames=5,
        revisions=revisions,
        based_on=v1,
    )
    store.save_processing(v2)

    report = {
        "capture_id": capture_id,
        "injected_events": events,
        "before_identity_revision": evaluate(v1.to_dict(), truth),
        "after_identity_revision": evaluate(v2.to_dict(), truth),
    }
    store.save_report(action_name, pipeline_name, report)
    print(f"saved {output}")
    print(f"v1 recovered rmse={report['before_identity_revision']['recovered_rmse']:.6f}")
    print(f"v2 recovered rmse={report['after_identity_revision']['recovered_rmse']:.6f}")
    print(
        "v2 identity mismatches="
        f"{report['after_identity_revision']['identity_mismatch_count']}"
    )


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
