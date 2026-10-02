from __future__ import annotations

import argparse
from pathlib import Path

from .calibration import calibrate_from_action
from .evaluation import evaluate
from .export_adapters import build_review_export
from .models import (
    Calibration,
    ContactObservation,
    IdentityCandidate,
    IdentityScheme,
    ObjectTrack,
    CaptureRecord,
    IdentityDecision,
    IdentityRevision,
    TrajectoryReassignment,
)
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
    contact_capture_id = scoped_capture_id

    def ball_position(frame_index: int) -> list[float] | None:
        # The ball follows the true p0 left wrist path; during the crossing that
        # path is observed under the swapped ``p1_r_wrist`` label.
        point = scoped_truth[frame_index]["p0_l_wrist"]
        if frame_index in {22, 23, 27}:
            return None  # moving object with short missing segments
        if 20 <= frame_index <= 24:
            midpoint = [
                (point[axis] + scoped_truth[frame_index]["p1_r_wrist"][axis]) / 2
                for axis in range(3)
            ]
            return midpoint
        return list(point)

    ball_track = ObjectTrack(
        object_id="ball_01",
        positions=[ball_position(index) for index in range(len(scoped_observed))],
        source="manual_annotation",
        notes="Hand-held ball with one missing frame; midpoint during a two-hand grasp.",
    )
    store.save_object_track(action_name, contact_pipeline, ball_track)

    crossing_contact_p0 = ContactObservation(
        contact_id="contact_001_p0_left_hand_ball_during_crossing",
        person_id="p0",
        body_part="l_wrist",
        object_id="ball_01",
        start_frame=24,
        end_frame=29,
        source="manual_review",
        reason="reviewer marks p0's left hand holding the ball through the crossing",
        simultaneous_with=["contact_002_p1_left_hand_ball_during_grasp"],
    )
    crossing_contact_p1 = ContactObservation(
        contact_id="contact_002_p1_left_hand_ball_during_grasp",
        person_id="p1",
        body_part="r_wrist",
        object_id="ball_01",
        start_frame=24,
        end_frame=24,
        source="manual_review",
        reason="reviewer marks p1 briefly touching the same ball in a two-hand grasp",
        simultaneous_with=["contact_001_p0_left_hand_ball_during_crossing"],
    )
    post_separation_contact = ContactObservation(
        contact_id="contact_003_p0_left_hand_ball_after_separation",
        person_id="p0",
        body_part="l_wrist",
        object_id="ball_01",
        start_frame=33,
        end_frame=36,
        source="manual_review_clip",
        reason="post-separation clip shows the same marked hand still carrying the ball",
    )
    automatic_hint = ContactObservation(
        contact_id="contact_004_automatic_near_object_hint",
        person_id="p1",
        body_part="r_wrist",
        object_id="ball_01",
        start_frame=40,
        end_frame=43,
        source="automatic_recommendation",
        reason="proximity recommender hint; not treated as manual confirmation",
    )
    short_occlusion_contact = ContactObservation(
        contact_id="contact_005_p1_hand_behind_ball_short_occlusion",
        person_id="p1",
        body_part="r_wrist",
        object_id="ball_01",
        start_frame=22,
        end_frame=23,
        source="manual_review_clip",
        reason="short occlusion clip suggests contact but the hand marker is absent",
    )
    for contact in (
        crossing_contact_p0,
        crossing_contact_p1,
        post_separation_contact,
        automatic_hint,
        short_occlusion_contact,
    ):
        store.save_contact(action_name, contact_pipeline, contact)

    candidate_swap = IdentityCandidate(
        candidate_id="candidate_swap_p0_left_is_p1_right_track",
        person_id="p0",
        joint_mapping={
            "l_shoulder": "p1_r_shoulder",
            "l_elbow": "p1_r_elbow",
            "l_wrist": "p1_r_wrist",
        },
        start_frame=20,
        end_frame=29,
        reason="after the crossing p0's left arm is on the track labeled p1 right arm",
        depends_on=["contact_001_p0_left_hand_ball_during_crossing"],
    )
    candidate_no_swap = IdentityCandidate(
        candidate_id="candidate_no_swap_keep_observed_labels",
        person_id="p0",
        joint_mapping={"l_wrist": "p0_l_wrist"},
        start_frame=20,
        end_frame=29,
        reason="keep the observed labels despite post-separation ball evidence",
        depends_on=["contact_003_p0_left_hand_ball_after_separation"],
    )
    candidate_double_claim = IdentityCandidate(
        candidate_id="candidate_illegal_double_claim",
        person_id="p1",
        joint_mapping={"l_wrist": "p1_r_wrist"},
        start_frame=22,
        end_frame=26,
        reason="illustration: a second identity claims the same observed wrist track",
        decision="open",
    )
    for candidate in (candidate_swap, candidate_no_swap, candidate_double_claim):
        store.save_candidate(action_name, contact_pipeline, candidate)

    scheme_swap = IdentityScheme(
        scheme_id="scheme_001_arm_swap",
        label="p0 left arm follows the track currently labeled p1 right arm",
        crossing_start_frame=20,
        crossing_end_frame=29,
        candidate_ids=[candidate_swap.candidate_id],
        notes="Independent interpretation supported by crossing and post-separation contact.",
        status="open",
    )
    scheme_keep = IdentityScheme(
        scheme_id="scheme_002_keep_labels",
        label="the observed arm labels remain with their named people",
        crossing_start_frame=20,
        crossing_end_frame=29,
        candidate_ids=[candidate_no_swap.candidate_id],
        notes="Alternative kept open while contact evidence is reviewed.",
        status="open",
    )
    scheme_conflict = IdentityScheme(
        scheme_id="scheme_003_double_claim",
        label="both identities claim the same wrist track",
        crossing_start_frame=22,
        crossing_end_frame=26,
        candidate_ids=[candidate_swap.candidate_id, candidate_double_claim.candidate_id],
        notes="Saved to demonstrate that simultaneous contact cannot share one body track.",
        status="open",
    )
    for scheme in (scheme_swap, scheme_keep, scheme_conflict):
        store.save_scheme(action_name, contact_pipeline, scheme)

    contact_objects = {ball_track.object_id: ball_track.positions}
    contact_v1 = reconstruct(
        action_name,
        contact_pipeline,
        contact_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        contact_observations=[
            crossing_contact_p0,
            crossing_contact_p1,
            post_separation_contact,
            automatic_hint,
            short_occlusion_contact,
        ],
        identity_candidates=[candidate_swap, candidate_no_swap, candidate_double_claim],
        contact_objects=contact_objects,
        identity_schemes=[scheme_swap, scheme_keep, scheme_conflict],
        ambiguity_windows=scoped_windows,
    )
    store.save_processing(contact_v1)

    # Reviewers adopt/exclude interpretations explicitly; an automatic hint is
    # recorded as evidence but never counts as a human confirmation.
    store.record_candidate_decision(
        action_name,
        contact_pipeline,
        candidate_swap.candidate_id,
        decision="adopted",
        reason="contact geometry and post-separation clip support the swap",
    )
    store.record_candidate_decision(
        action_name,
        contact_pipeline,
        candidate_no_swap.candidate_id,
        decision="excluded",
        reason="the retained-label interpretation is geometrically incompatible",
    )
    adopted_candidates = [
        IdentityCandidate(**data)
        for data in store.load_candidates(action_name, contact_pipeline)
    ]
    contact_v2 = reconstruct(
        action_name,
        contact_pipeline,
        contact_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        contact_observations=[
            crossing_contact_p0,
            crossing_contact_p1,
            post_separation_contact,
            automatic_hint,
            short_occlusion_contact,
        ],
        identity_candidates=adopted_candidates,
        contact_objects=contact_objects,
        identity_schemes=[scheme_swap, scheme_keep, scheme_conflict],
        ambiguity_windows=scoped_windows,
        based_on=contact_v1,
    )
    store.save_processing(contact_v2)

    # Changing one contact source rechecks only candidates/schemes that depend
    # on it; the crossing evidence and automatic hint remain unchanged.
    store.revoke_contact(
        action_name,
        contact_pipeline,
        post_separation_contact.contact_id,
        reason="reviewer withdraws the post-separation contact annotation for recutting",
    )
    reloaded_contacts = [
        ContactObservation(**data)
        for data in store.load_contacts(action_name, contact_pipeline)
    ]
    contact_v3 = reconstruct(
        action_name,
        contact_pipeline,
        contact_capture_id,
        scoped_observed,
        scoped_calibration_dict,
        contact_observations=reloaded_contacts,
        identity_candidates=adopted_candidates,
        contact_objects=contact_objects,
        identity_schemes=[scheme_swap, scheme_keep, scheme_conflict],
        ambiguity_windows=scoped_windows,
        based_on=contact_v2,
    )
    store.save_processing(contact_v3)

    contact_export = build_review_export(
        store.load_capture(contact_capture_id),
        contact_v3.to_dict(),
        [],
        [],
        contacts=reloaded_contacts,
        candidates=adopted_candidates,
        schemes=[scheme_swap, scheme_keep, scheme_conflict],
        object_tracks=store.load_object_tracks(action_name, contact_pipeline),
    )
    store.save_review_export(action_name, contact_pipeline, contact_export)
    contact_report = {
        "capture_id": contact_capture_id,
        "contact_review": contact_v2.parameters["contact_review"],
        "scheme_reviews": contact_v3.parameters["scheme_reviews"],
        "contact_dependent_candidates": contact_v3.parameters[
            "contact_dependent_candidates"
        ],
        "contact_dependent_schemes": contact_v3.parameters[
            "contact_dependent_schemes"
        ],
        "review_export": "review_export.json",
    }
    store.save_report(action_name, contact_pipeline, contact_report)
    scheme_verdicts = {
        item["scheme_id"]: [
            result["verdict"] for result in item["contact_review"]["contact_results"]
        ]
        for item in contact_v2.parameters["scheme_reviews"]
    }
    print(f"contact evidence review schemes={scheme_verdicts}")


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
