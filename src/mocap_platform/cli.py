from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import calibrate_from_action
from .contact import affected_scenarios, review_scenario
from .evaluation import evaluate
from .export_adapters import build_review_export
from .models import (
    Calibration,
    CaptureRecord,
    ContactEvidence,
    IdentityDecision,
    IdentityRevision,
    InterpretationScenario,
    TrajectoryReassignment,
)
from .synthetic_multi import (
    CONTACT_OBJECT,
    generate_multi_person_action,
    inject_contact_review_events,
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

    review_pipeline = "trajectory_reassignment_review"
    review_capture_id = scoped_capture_id

    review_v1 = reconstruct(
        action_name,
        review_pipeline,
        review_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        ambiguity_windows=scoped_windows,
    )
    store.save_processing(review_v1)

    arm_decision = IdentityDecision(
        decision_id="review_001_confirm_p0_left_arm",
        person_id="p0",
        marker="l_wrist",
        start_frame=20,
        end_frame=24,
        decision="confirm",
        alternative_person_id="p1",
        reason="manual review confirms the left arm track belongs to p0",
        related_markers=["l_shoulder", "l_elbow", "l_wrist"],
        alternative_related_markers=["r_shoulder", "r_elbow", "r_wrist"],
    )
    store.save_decision(action_name, review_pipeline, arm_decision)
    review_v2 = reconstruct(
        action_name,
        review_pipeline,
        review_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        decisions=[arm_decision],
        ambiguity_windows=scoped_windows,
        based_on=review_v1,
    )
    store.save_processing(review_v2)

    arm_reassignment = TrajectoryReassignment(
        reassignment_id="reassign_001_p1_right_arm_to_p0",
        source_person_id="p1",
        target_person_id="p0",
        joints=["r_shoulder", "r_elbow", "r_wrist"],
        target_joints=["l_shoulder", "l_elbow", "l_wrist"],
        start_frame=20,
        end_frame=29,
        reason="the track labeled p1 right arm is p0's left arm after the crossing",
    )
    store.save_reassignment(action_name, review_pipeline, arm_reassignment)
    review_v3 = reconstruct(
        action_name,
        review_pipeline,
        review_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        decisions=[arm_decision],
        reassignments=[arm_reassignment],
        ambiguity_windows=scoped_windows,
        based_on=review_v2,
    )
    store.save_processing(review_v3)

    store.apply_reassignment(
        action_name,
        review_pipeline,
        arm_reassignment.reassignment_id,
        kept_decision_ids=[arm_decision.decision_id],
    )
    applied_arm_reassignment = TrajectoryReassignment(
        reassignment_id=arm_reassignment.reassignment_id,
        source_person_id=arm_reassignment.source_person_id,
        target_person_id=arm_reassignment.target_person_id,
        joints=arm_reassignment.joints,
        target_joints=arm_reassignment.target_joints,
        start_frame=arm_reassignment.start_frame,
        end_frame=arm_reassignment.end_frame,
        reason=arm_reassignment.reason,
        kept_decision_ids=[arm_decision.decision_id],
        status="applied",
        created_at=arm_reassignment.created_at,
    )
    review_v4 = reconstruct(
        action_name,
        review_pipeline,
        review_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        decisions=[arm_decision],
        reassignments=[applied_arm_reassignment],
        ambiguity_windows=scoped_windows,
        based_on=review_v3,
    )
    store.save_processing(review_v4)

    ankle_decision = IdentityDecision(
        decision_id="review_002_confirm_p0_right_ankle",
        person_id="p0",
        marker="r_ankle",
        start_frame=25,
        end_frame=29,
        decision="confirm",
        alternative_person_id="p1",
        reason="manual review confirms the right ankle track belongs to p0",
        depends_on=[arm_reassignment.reassignment_id],
    )
    store.save_decision(action_name, review_pipeline, ankle_decision)
    review_v5 = reconstruct(
        action_name,
        review_pipeline,
        review_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        decisions=[arm_decision, ankle_decision],
        reassignments=[applied_arm_reassignment],
        ambiguity_windows=scoped_windows,
        based_on=review_v4,
    )
    store.save_processing(review_v5)

    contested_reassignment = TrajectoryReassignment(
        reassignment_id="reassign_002_same_wrist_track",
        source_person_id="p1",
        target_person_id="p0",
        joints=["r_wrist"],
        target_joints=["r_wrist"],
        start_frame=22,
        end_frame=26,
        reason="a second reviewer claims the same wrist track for p0's right wrist",
        status="applied",
    )
    store.save_reassignment(action_name, review_pipeline, contested_reassignment)
    review_v6 = reconstruct(
        action_name,
        review_pipeline,
        review_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        decisions=[arm_decision, ankle_decision],
        reassignments=[applied_arm_reassignment, contested_reassignment],
        ambiguity_windows=scoped_windows,
        based_on=review_v5,
    )
    store.save_processing(review_v6)

    store.revoke_reassignment(
        action_name,
        review_pipeline,
        contested_reassignment.reassignment_id,
        reason="reviewers adjusted both sides and dropped the duplicate claim",
    )
    store.revoke_reassignment(
        action_name,
        review_pipeline,
        arm_reassignment.reassignment_id,
        reason="reviewer withdrew the arm trajectory reassignment",
    )
    reloaded_decisions = [
        IdentityDecision(**data)
        for data in store.load_decisions(action_name, review_pipeline)
    ]
    reloaded_reassignments = [
        TrajectoryReassignment(**data)
        for data in store.load_reassignments(action_name, review_pipeline)
    ]
    review_v7 = reconstruct(
        action_name,
        review_pipeline,
        review_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        decisions=reloaded_decisions,
        reassignments=reloaded_reassignments,
        ambiguity_windows=scoped_windows,
        based_on=review_v6,
    )
    store.save_processing(review_v7)

    review_export = build_review_export(
        store.load_capture(review_capture_id),
        review_v7.to_dict(),
        reloaded_decisions,
        reloaded_reassignments,
    )
    store.save_review_export(action_name, review_pipeline, review_export)

    review_versions = [
        review_v1.to_dict(),
        review_v2.to_dict(),
        review_v3.to_dict(),
        review_v4.to_dict(),
        review_v5.to_dict(),
        review_v6.to_dict(),
        review_v7.to_dict(),
    ]
    review_report = {
        "capture_id": review_capture_id,
        "injected_events": scoped_events,
        "comparison": compare_identity_revisions(review_versions, scoped_truth),
        "reassignment_review": {
            "reassignment_impacts": review_v7.parameters["reassignment_impacts"],
            "reassignment_disputes": review_v6.parameters["reassignment_disputes"],
            "reassignment_dependencies": review_v7.parameters[
                "reassignment_dependencies"
            ],
            "pending_confirmations": review_v7.parameters["pending_confirmations"],
            "pose_comparison": review_v4.parameters["pose_comparison"],
        },
        "review_export": "review_export.json",
    }
    store.save_report(action_name, review_pipeline, review_report)
    review_metrics = [
        item["metrics"]["ambiguous_identity_marker_frames"]
        for item in review_report["comparison"]["versions"]
    ]
    print(f"reassignment review v1..v7 ambiguous marker frames={review_metrics}")

    contact_pipeline = "contact_evidence_review"
    contact_capture_id = "demo_capture_0004"

    contact_truth = generate_multi_person_action()
    contact_observed, object_frames, contact_events, contact_windows = (
        inject_contact_review_events(contact_truth)
    )
    contact_capture = CaptureRecord(
        capture_id=contact_capture_id,
        action_name=action_name,
        frame_count=len(contact_observed),
        frame_rate=30.0,
        markers=multi_person_markers(),
        observed_frames=contact_observed,
        ground_truth_frames=contact_truth,
        objects=[CONTACT_OBJECT],
        object_frames=object_frames,
        source="synthetic_known_multi_person_action",
        notes="Arm crossing while both people touch a moving object; the object track has a missing segment.",
    )
    store.save_capture(contact_capture)

    contact_bone_lengths, contact_joint_limits = calibrate_from_action(contact_truth)
    contact_calibration = Calibration(
        calibration_id="calibration_demo_0004",
        capture_id=contact_capture_id,
        bone_lengths=contact_bone_lengths,
        joint_limits=contact_joint_limits,
    )
    store.save_calibration(contact_calibration)
    contact_calibration_dict = contact_calibration.to_dict()

    wrist_contact = ContactEvidence(
        contact_id="contact_001_p0_left_wrist_crate",
        person_id="p0",
        joint="l_wrist",
        object_id=CONTACT_OBJECT,
        start_frame=18,
        end_frame=24,
        reason="reviewer sees p0's left hand on the crate before and during the crossing",
    )
    shared_contact = ContactEvidence(
        contact_id="contact_002_p1_right_wrist_crate",
        person_id="p1",
        joint="r_wrist",
        object_id=CONTACT_OBJECT,
        start_frame=23,
        end_frame=28,
        reason="p1's right hand also rests on the crate as the two people separate",
    )
    unobserved_contact = ContactEvidence(
        contact_id="contact_003_p0_right_ankle_crate",
        person_id="p0",
        joint="r_ankle",
        object_id=CONTACT_OBJECT,
        start_frame=26,
        end_frame=28,
        reason="a hypothesized foot contact exactly where the object track is unobserved",
    )
    contacts = [wrist_contact, shared_contact, unobserved_contact]
    for contact in contacts:
        store.save_contact(action_name, contact_pipeline, contact)

    contact_v1 = reconstruct(
        action_name,
        contact_pipeline,
        contact_capture_id,
        contact_observed,
        contact_calibration_dict,
        contacts=contacts,
        object_frames=object_frames,
        ambiguity_windows=contact_windows,
    )
    store.save_processing(contact_v1)

    contact_reassignment = TrajectoryReassignment(
        reassignment_id="contact_reassign_001_p1_right_arm_to_p0",
        source_person_id="p1",
        target_person_id="p0",
        joints=["r_shoulder", "r_elbow", "r_wrist"],
        target_joints=["l_shoulder", "l_elbow", "l_wrist"],
        start_frame=20,
        end_frame=29,
        reason="the track labeled p1 right arm is p0's left arm while both touch the crate",
    )
    store.save_reassignment(action_name, contact_pipeline, contact_reassignment)
    store.apply_reassignment(
        action_name, contact_pipeline, contact_reassignment.reassignment_id
    )
    applied_contact_reassignment = TrajectoryReassignment(
        reassignment_id=contact_reassignment.reassignment_id,
        source_person_id=contact_reassignment.source_person_id,
        target_person_id=contact_reassignment.target_person_id,
        joints=contact_reassignment.joints,
        target_joints=contact_reassignment.target_joints,
        start_frame=contact_reassignment.start_frame,
        end_frame=contact_reassignment.end_frame,
        reason=contact_reassignment.reason,
        status="applied",
        created_at=contact_reassignment.created_at,
    )
    contact_v2 = reconstruct(
        action_name,
        contact_pipeline,
        contact_capture_id,
        contact_observed,
        contact_calibration_dict,
        reassignments=[applied_contact_reassignment],
        contacts=contacts,
        object_frames=object_frames,
        ambiguity_windows=contact_windows,
        based_on=contact_v1,
    )
    store.save_processing(contact_v2)

    duplicate_claim = TrajectoryReassignment(
        reassignment_id="contact_reassign_002_same_wrist_track",
        source_person_id="p1",
        target_person_id="p0",
        joints=["r_wrist"],
        target_joints=["r_wrist"],
        start_frame=22,
        end_frame=26,
        reason="a second reviewer claims the same wrist track for p0's right wrist",
        status="applied",
    )
    store.save_reassignment(action_name, contact_pipeline, duplicate_claim)

    scenario_a = InterpretationScenario(
        scenario_id="scenario_a_reassigned_contact",
        label="p0's left arm follows the crate contact through the crossing",
        reason="contact evidence places p0's left hand on the crate frames 18-24",
        reassignment_ids=[contact_reassignment.reassignment_id],
        contact_ids=[contact.contact_id for contact in contacts],
    )
    scenario_b = InterpretationScenario(
        scenario_id="scenario_b_no_reassignment",
        label="labels stay as observed, no reassignment",
        reason="trajectory proximity alone suggests keeping the raw labels",
        contact_ids=[wrist_contact.contact_id],
    )
    scenario_c = InterpretationScenario(
        scenario_id="scenario_c_duplicate_claim",
        label="both reassignments claim the wrist track",
        reason="a second reviewer additionally claims the same wrist track",
        reassignment_ids=[
            contact_reassignment.reassignment_id,
            duplicate_claim.reassignment_id,
        ],
        contact_ids=[wrist_contact.contact_id],
    )
    scenarios = [scenario_a, scenario_b, scenario_c]
    for scenario in scenarios:
        store.save_scenario(action_name, contact_pipeline, scenario)

    all_reassignments = [applied_contact_reassignment, duplicate_claim]
    assessment_a = review_scenario(
        scenario_a,
        contacts,
        [],
        [applied_contact_reassignment],
        contact_v2.frames,
        contact_v2.statuses,
        object_frames,
    )
    assessment_b = review_scenario(
        scenario_b,
        contacts,
        [],
        [],
        contact_v1.frames,
        contact_v1.statuses,
        object_frames,
    )
    assessment_c = review_scenario(
        scenario_c,
        contacts,
        [],
        all_reassignments,
        contact_v2.frames,
        contact_v2.statuses,
        object_frames,
    )
    store.exclude_scenario(
        action_name,
        contact_pipeline,
        scenario_b.scenario_id,
        reason="contact_001 is geometrically incompatible at frame 20 without the reassignment",
    )
    contact_v3 = reconstruct(
        action_name,
        contact_pipeline,
        contact_capture_id,
        contact_observed,
        contact_calibration_dict,
        reassignments=[applied_contact_reassignment],
        contacts=contacts,
        object_frames=object_frames,
        scenario_assessments=[assessment_a, assessment_b, assessment_c],
        ambiguity_windows=contact_windows,
        based_on=contact_v2,
    )
    store.save_processing(contact_v3)

    ankle_decision = IdentityDecision(
        decision_id="contact_decision_001_confirm_p0_right_ankle",
        person_id="p0",
        marker="r_ankle",
        start_frame=25,
        end_frame=29,
        decision="confirm",
        alternative_person_id="p1",
        reason="ankle confirmation relied on the shared crate contact",
        depends_on=[shared_contact.contact_id],
    )
    store.save_decision(action_name, contact_pipeline, ankle_decision)
    store.revoke_contact(
        action_name,
        contact_pipeline,
        shared_contact.contact_id,
        reason="the shared-contact annotation was withdrawn after re-inspection",
    )
    reloaded_contacts = [
        ContactEvidence(**data)
        for data in store.load_contacts(action_name, contact_pipeline)
    ]
    reloaded_scenarios = [
        InterpretationScenario(**data)
        for data in store.load_scenarios(action_name, contact_pipeline)
    ]
    rechecked_ids = affected_scenarios(
        reloaded_scenarios, changed_contact_ids={shared_contact.contact_id}
    )
    reassessed = [
        review_scenario(
            scenario,
            reloaded_contacts,
            [ankle_decision],
            all_reassignments
            if scenario.scenario_id == scenario_c.scenario_id
            else [applied_contact_reassignment],
            contact_v2.frames,
            contact_v2.statuses,
            object_frames,
        )
        for scenario in reloaded_scenarios
        if scenario.scenario_id in rechecked_ids
    ]
    contact_v4 = reconstruct(
        action_name,
        contact_pipeline,
        contact_capture_id,
        contact_observed,
        contact_calibration_dict,
        decisions=[ankle_decision],
        reassignments=[applied_contact_reassignment],
        contacts=reloaded_contacts,
        object_frames=object_frames,
        scenario_assessments=reassessed,
        review_annotations={
            "contact_recheck": {
                "trigger": "contact_revoked",
                "contact_id": shared_contact.contact_id,
                "rechecked_scenarios": rechecked_ids,
                "unaffected_scenarios": [
                    scenario.scenario_id
                    for scenario in reloaded_scenarios
                    if scenario.scenario_id not in rechecked_ids
                ],
            }
        },
        ambiguity_windows=contact_windows,
        based_on=contact_v3,
    )
    store.save_processing(contact_v4)

    store.adopt_scenario(
        action_name,
        contact_pipeline,
        scenario_a.scenario_id,
        reason="reviewer adopted the contact-consistent interpretation after resolving disputes",
    )
    final_scenarios = [
        InterpretationScenario(**data)
        for data in store.load_scenarios(action_name, contact_pipeline)
    ]
    contact_export = build_review_export(
        store.load_capture(contact_capture_id),
        contact_v4.to_dict(),
        [ankle_decision],
        all_reassignments,
        contacts=reloaded_contacts,
        scenarios=final_scenarios,
    )
    store.save_review_export(action_name, contact_pipeline, contact_export)

    contact_versions = [
        contact_v1.to_dict(),
        contact_v2.to_dict(),
        contact_v3.to_dict(),
        contact_v4.to_dict(),
    ]
    contact_report = {
        "capture_id": contact_capture_id,
        "injected_events": contact_events,
        "comparison": compare_identity_revisions(contact_versions, contact_truth),
        "contact_review": {
            "baseline_assessments": contact_v1.parameters["contact_review"],
            "scenario_assessments": contact_v3.parameters["scenario_assessments"],
            "contact_recheck": contact_v4.parameters["review_annotations"][
                "contact_recheck"
            ],
            "pending_confirmations": contact_v4.parameters["pending_confirmations"],
        },
        "review_export": "review_export.json",
    }
    store.save_report(action_name, contact_pipeline, contact_report)
    print(
        "contact review scenarios: "
        f"A={assessment_a['automatic_assessment']} "
        f"B={assessment_b['automatic_assessment']} "
        f"C={assessment_c['automatic_assessment']}"
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
