from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import calibrate_from_action
from .evaluation import evaluate
from .models import Calibration, CaptureRecord, IdentityDecision, IdentityRevision
from .synthetic_multi import (
    generate_multi_person_action,
    inject_multi_person_crossing,
    inject_scoped_review_events,
    multi_person_markers,
)
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

    action_name = "demo_multi_person_action"
    pipeline_name = "crossing_identity_review"
    capture_id = "demo_capture_0002"

    multi_truth = generate_multi_person_action()
    multi_observed, multi_events, ambiguity_windows = inject_multi_person_crossing(
        multi_truth
    )
    multi_capture = CaptureRecord(
        capture_id=capture_id,
        action_name=action_name,
        frame_count=len(multi_observed),
        frame_rate=30.0,
        markers=multi_person_markers(),
        observed_frames=multi_observed,
        ground_truth_frames=multi_truth,
        source="synthetic_known_multi_person_action",
        notes="Two-person crossing, simultaneous occlusion, long gaps, and reappearance ambiguity.",
    )
    store.save_capture(multi_capture)

    multi_bone_lengths, multi_joint_limits = calibrate_from_action(multi_truth)
    multi_calibration = Calibration(
        calibration_id="calibration_demo_0002",
        capture_id=capture_id,
        bone_lengths=multi_bone_lengths,
        joint_limits=multi_joint_limits,
    )
    store.save_calibration(multi_calibration)

    multi_v1 = reconstruct(
        action_name,
        pipeline_name,
        capture_id,
        multi_observed,
        multi_calibration.to_dict(),
        ambiguity_windows=ambiguity_windows,
    )
    store.save_processing(multi_v1)

    crossing_confirmation = IdentityDecision(
        decision_id="confirm_p0_left_arm_during_crossing",
        person_id="p0",
        marker="l_wrist",
        start_frame=20,
        end_frame=29,
        decision="confirm",
        alternative_person_id="p1",
        reason="manual review confirms the left arm track belongs to p0 during crossing",
        related_markers=["l_shoulder", "l_elbow", "l_wrist"],
        alternative_related_markers=["r_shoulder", "r_elbow", "r_wrist"],
    )
    multi_v2 = reconstruct(
        action_name,
        pipeline_name,
        capture_id,
        multi_observed,
        multi_calibration.to_dict(),
        decisions=[crossing_confirmation],
        ambiguity_windows=ambiguity_windows,
        based_on=multi_v1,
    )
    store.save_processing(multi_v2)

    revoked_confirmation = IdentityDecision(
        decision_id=crossing_confirmation.decision_id,
        person_id=crossing_confirmation.person_id,
        marker=crossing_confirmation.marker,
        start_frame=crossing_confirmation.start_frame,
        end_frame=crossing_confirmation.end_frame,
        decision=crossing_confirmation.decision,
        alternative_person_id=crossing_confirmation.alternative_person_id,
        reason=crossing_confirmation.reason,
        related_markers=crossing_confirmation.related_markers,
        alternative_related_markers=crossing_confirmation.alternative_related_markers,
        status="revoked",
        created_at=crossing_confirmation.created_at,
    )
    multi_v3 = reconstruct(
        action_name,
        pipeline_name,
        capture_id,
        multi_observed,
        multi_calibration.to_dict(),
        decisions=[revoked_confirmation],
        ambiguity_windows=ambiguity_windows,
        based_on=multi_v2,
    )
    store.save_processing(multi_v3)

    from .evaluation import compare_identity_revisions

    multi_report = {
        "capture_id": capture_id,
        "injected_events": multi_events,
        "comparison": compare_identity_revisions(
            [multi_v1.to_dict(), multi_v2.to_dict(), multi_v3.to_dict()], multi_truth
        ),
    }
    store.save_report(action_name, pipeline_name, multi_report)
    print(
        "multi v1/v2/v3 bone RMSE="
        f"{multi_report['comparison']['versions'][0]['metrics']['bone_length_rmse']:.6f}/"
        f"{multi_report['comparison']['versions'][1]['metrics']['bone_length_rmse']:.6f}/"
        f"{multi_report['comparison']['versions'][2]['metrics']['bone_length_rmse']:.6f}"
    )

    scoped_pipeline = "scoped_confirmation_review"
    scoped_capture_id = "demo_capture_0003"

    scoped_truth = generate_multi_person_action()
    scoped_observed, scoped_events, scoped_windows = inject_scoped_review_events(
        scoped_truth
    )
    scoped_capture = CaptureRecord(
        capture_id=scoped_capture_id,
        action_name=action_name,
        frame_count=len(scoped_observed),
        frame_rate=30.0,
        markers=multi_person_markers(),
        observed_frames=scoped_observed,
        ground_truth_frames=scoped_truth,
        source="synthetic_known_multi_person_action",
        notes="Arm crossing, short right-ankle swap with local occlusion, and reappearance ambiguity.",
    )
    store.save_capture(scoped_capture)

    scoped_bone_lengths, scoped_joint_limits = calibrate_from_action(scoped_truth)
    scoped_calibration = Calibration(
        calibration_id="calibration_demo_0003",
        capture_id=scoped_capture_id,
        bone_lengths=scoped_bone_lengths,
        joint_limits=scoped_joint_limits,
    )
    store.save_calibration(scoped_calibration)
    scoped_calibration_dict = scoped_calibration.to_dict()

    scoped_v1 = reconstruct(
        action_name,
        scoped_pipeline,
        scoped_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        ambiguity_windows=scoped_windows,
    )
    store.save_processing(scoped_v1)

    ankle_confirmation = IdentityDecision(
        decision_id="decision_001_confirm_p0_right_ankle",
        person_id="p0",
        marker="r_ankle",
        start_frame=25,
        end_frame=29,
        decision="confirm",
        alternative_person_id="p1",
        reason="manual review confirms the right ankle track belongs to p0",
    )
    store.save_decision(action_name, scoped_pipeline, ankle_confirmation)
    scoped_v2 = reconstruct(
        action_name,
        scoped_pipeline,
        scoped_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        decisions=[ankle_confirmation],
        ambiguity_windows=scoped_windows,
        based_on=scoped_v1,
    )
    store.save_processing(scoped_v2)

    arm_confirmation = IdentityDecision(
        decision_id="decision_002_confirm_p0_left_arm",
        person_id="p0",
        marker="l_wrist",
        start_frame=20,
        end_frame=24,
        decision="confirm",
        alternative_person_id="p1",
        reason="manual review confirms the left arm track belongs to p0 for the first crossing half",
        related_markers=["l_shoulder", "l_elbow", "l_wrist"],
        alternative_related_markers=["r_shoulder", "r_elbow", "r_wrist"],
    )
    store.save_decision(action_name, scoped_pipeline, arm_confirmation)
    scoped_v3 = reconstruct(
        action_name,
        scoped_pipeline,
        scoped_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        decisions=[ankle_confirmation, arm_confirmation],
        ambiguity_windows=scoped_windows,
        based_on=scoped_v2,
    )
    store.save_processing(scoped_v3)

    conflicting_deny = IdentityDecision(
        decision_id="decision_003_deny_p0_left_wrist",
        person_id="p0",
        marker="l_wrist",
        start_frame=22,
        end_frame=26,
        decision="deny",
        alternative_person_id="p1",
        reason="conflicting review claims the wrist track does not belong to p0",
    )
    store.save_decision(action_name, scoped_pipeline, conflicting_deny)
    scoped_v4 = reconstruct(
        action_name,
        scoped_pipeline,
        scoped_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        decisions=[ankle_confirmation, arm_confirmation, conflicting_deny],
        ambiguity_windows=scoped_windows,
        based_on=scoped_v3,
    )
    store.save_processing(scoped_v4)

    store.revoke_decision(
        action_name,
        scoped_pipeline,
        ankle_confirmation.decision_id,
        reason="reviewer withdrew the ankle confirmation",
    )
    reloaded_decisions = [
        IdentityDecision(**data)
        for data in store.load_decisions(action_name, scoped_pipeline)
    ]
    scoped_v5 = reconstruct(
        action_name,
        scoped_pipeline,
        scoped_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        decisions=reloaded_decisions,
        ambiguity_windows=scoped_windows,
        based_on=scoped_v4,
    )
    store.save_processing(scoped_v5)

    scoped_versions = [
        scoped_v1.to_dict(),
        scoped_v2.to_dict(),
        scoped_v3.to_dict(),
        scoped_v4.to_dict(),
        scoped_v5.to_dict(),
    ]
    scoped_report = {
        "capture_id": scoped_capture_id,
        "injected_events": scoped_events,
        "comparison": compare_identity_revisions(scoped_versions, scoped_truth),
        "decision_review": {
            "decision_impacts": scoped_v5.parameters["decision_impacts"],
            "decision_conflicts": scoped_v5.parameters["decision_conflicts"],
            "restored_ambiguities": scoped_v5.parameters["restored_ambiguities"],
        },
    }
    store.save_report(action_name, scoped_pipeline, scoped_report)
    scoped_metrics = [
        item["metrics"]["ambiguous_identity_marker_frames"]
        for item in scoped_report["comparison"]["versions"]
    ]
    print(f"scoped v1..v5 ambiguous marker frames={scoped_metrics}")


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
