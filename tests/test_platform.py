from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mocap_platform.cli import demo
from mocap_platform.evaluation import evaluate
from mocap_platform.models import CaptureRecord, IdentityDecision, IdentityRevision
from mocap_platform.contact import (
    candidates_dependent_on,
    evaluate_contact_compatibility,
)
from mocap_platform.models import (
    ContactObservation,
    IdentityCandidate,
    SupplementAdoption,
    SupplementBatch,
    SupplementSelection,
    SupplementTrack,
)
from mocap_platform.contact import resolve_supplement_points
from mocap_platform.reconstruction import reconstruct
from mocap_platform.storage import PlatformStore
from mocap_platform.synthetic import generate_known_action, inject_occlusions_and_swaps
from mocap_platform.synthetic_multi import (
    generate_multi_person_action,
    inject_multi_person_crossing,
)


class PlatformIntegrationTest(unittest.TestCase):
    def test_candidate_recheck_follows_revoked_evidence_and_decisions(self) -> None:
        candidate = IdentityCandidate(
            candidate_id="candidate_001",
            person_id="p0",
            joint_mapping={"l_wrist": "p1_r_wrist"},
            start_frame=20,
            end_frame=29,
            depends_on=["contact_001", "decision_001", "reassignment_001"],
        )
        self.assertEqual(
            candidates_dependent_on(
                [candidate],
                revoked_contact_ids=set(),
                revoked_reassignment_ids=set(),
                revoked_decision_ids={"decision_001"},
            ),
            ["candidate_001"],
        )
        self.assertEqual(
            candidates_dependent_on(
                [candidate],
                revoked_contact_ids={"contact_999"},
                revoked_reassignment_ids={"reassignment_999"},
                revoked_decision_ids=set(),
            ),
            [],
        )

    def test_scoped_confirmation_conflicts_revocation_and_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo(directory)
            store = PlatformStore(directory)
            action = "demo_multi_person_action"
            pipeline = "scoped_confirmation_review"
            base = Path(directory, "processing", action, pipeline)

            def load(version: str) -> dict:
                return json.loads(
                    base.joinpath("versions", version, "reconstruction.json").read_text(
                        encoding="utf-8"
                    )
                )

            v1, v2, v3, v4, v5 = (load(v) for v in ("v001", "v002", "v003", "v004", "v005"))
            truth = store.load_capture("demo_capture_0003")["ground_truth"]

            # A decision only resolves its own markers and frame range.
            self.assertEqual(v1["identity_statuses"][26]["p0_r_ankle"], "ambiguous")
            self.assertEqual(v2["identity_statuses"][26]["p0_r_ankle"], "confirmed")
            self.assertEqual(v2["frames"][26]["p0_r_ankle"], truth[26]["p0_r_ankle"])
            # Other markers in an overlapping window stay ambiguous.
            self.assertEqual(v2["identity_statuses"][26]["p0_l_wrist"], "ambiguous")
            # Same markers in another time range stay ambiguous.
            self.assertEqual(v2["identity_statuses"][41]["p0_l_wrist"], "ambiguous")
            # The ankle window outside the decision range keeps its ambiguity.
            self.assertEqual(v3["identity_statuses"][22]["p0_l_wrist"], "confirmed")
            self.assertEqual(v3["identity_statuses"][26]["p0_l_wrist"], "ambiguous")
            self.assertEqual(v3["identity_statuses"][41]["p0_l_wrist"], "ambiguous")

            # Adjacent confirmation windows keep separate, exact ranges.
            windows = {
                item["decision_id"]: (item["start_frame"], item["end_frame"])
                for item in v3["parameters"]["decision_windows"]
            }
            self.assertEqual(
                windows,
                {
                    "decision_001_confirm_p0_right_ankle": (25, 29),
                    "decision_002_confirm_p0_left_arm": (20, 24),
                },
            )

            # Conflicting decisions are itemized; the earlier one stays applied.
            conflicts = v4["parameters"]["decision_conflicts"]
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0]["decision_id"], "decision_003_deny_p0_left_wrist")
            self.assertEqual(conflicts[0]["conflicts_with"], "decision_002_confirm_p0_left_arm")
            self.assertEqual((conflicts[0]["start_frame"], conflicts[0]["end_frame"]), (22, 24))
            impacts = {
                item["decision_id"]: item["status"]
                for item in v4["parameters"]["decision_impacts"]
            }
            self.assertEqual(impacts["decision_001_confirm_p0_right_ankle"], "applied")
            self.assertEqual(impacts["decision_002_confirm_p0_left_arm"], "applied")
            self.assertEqual(impacts["decision_003_deny_p0_left_wrist"], "conflict")
            self.assertTrue(
                all(item["trajectories"] for item in v4["parameters"]["decision_impacts"])
            )
            self.assertEqual(v4["frames"][23]["p0_l_wrist"], truth[23]["p0_l_wrist"])
            self.assertEqual(v4["frames"], v3["frames"])

            # Revoking one decision restores only its own ambiguity.
            self.assertEqual(v5["identity_statuses"][26]["p0_r_ankle"], "ambiguous")
            self.assertEqual(v5["identity_statuses"][22]["p0_l_wrist"], "confirmed")
            restored = v5["parameters"]["restored_ambiguities"]
            self.assertEqual(
                restored,
                [
                    {"marker": "p0_r_ankle", "start_frame": 25, "end_frame": 29},
                    {"marker": "p1_r_ankle", "start_frame": 25, "end_frame": 29},
                ],
            )
            self.assertEqual(v5["version"], "v005")
            self.assertEqual(v5["based_on"], "v004")

            # Decisions persist immutably and reload for continued review.
            decisions = store.load_decisions(action, pipeline)
            self.assertEqual(len(decisions), 3)
            statuses = {item["decision_id"]: item["status"] for item in decisions}
            self.assertEqual(statuses["decision_001_confirm_p0_right_ankle"], "revoked")
            self.assertEqual(statuses["decision_002_confirm_p0_left_arm"], "active")
            with self.assertRaises(FileExistsError):
                store.save_decision(
                    action,
                    pipeline,
                    IdentityDecision(
                        decision_id="decision_002_confirm_p0_left_arm",
                        person_id="p0",
                        marker="l_wrist",
                        start_frame=20,
                        end_frame=24,
                        decision="deny",
                        alternative_person_id="p1",
                        reason="attempted overwrite",
                    ),
                )
            with self.assertRaises(FileExistsError):
                store.revoke_decision(
                    action, pipeline, "decision_001_confirm_p0_right_ankle"
                )
            with self.assertRaises(FileNotFoundError):
                store.revoke_decision(action, pipeline, "decision_999_missing")
            with self.assertRaises(FileExistsError):
                store.save_capture(
                    CaptureRecord(
                        capture_id="demo_capture_0003",
                        action_name=action,
                        frame_count=1,
                        frame_rate=30.0,
                        markers=[],
                        observed_frames=[],
                    )
                )

    def test_versioned_recovery_identity_revision_and_long_gap(self) -> None:
        truth = generate_known_action()
        observed, _events = inject_occlusions_and_swaps(truth)
        with tempfile.TemporaryDirectory() as directory:
            demo(directory)
            store = PlatformStore(directory)

            capture_path = Path(directory, "captures", "demo_capture_0001", "v001_capture.json")
            v1_path = Path(
                directory,
                "processing",
                "demo_action",
                "identity_swap_demo",
                "versions",
                "v001",
                "reconstruction.json",
            )
            v2_path = Path(
                directory,
                "processing",
                "demo_action",
                "identity_swap_demo",
                "versions",
                "v002",
                "reconstruction.json",
            )
            report_path = Path(
                directory,
                "processing",
                "demo_action",
                "identity_swap_demo",
                "evaluation_report.json",
            )
            self.assertTrue(capture_path.exists())
            self.assertTrue(v1_path.exists())
            self.assertTrue(v2_path.exists())
            self.assertTrue(report_path.exists())

            capture = json.loads(capture_path.read_text(encoding="utf-8"))
            v1 = json.loads(v1_path.read_text(encoding="utf-8"))
            v2 = json.loads(v2_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertIsNone(capture["observations"][15]["head"])
            self.assertIsNotNone(capture["ground_truth"][15]["head"])
            self.assertIsNone(v1["frames"][15]["head"])
            self.assertEqual(v1["statuses"][15]["head"], "uncertain")
            self.assertIsNone(v2["frames"][15]["head"])

            self.assertEqual(v1["version"], "v001")
            self.assertEqual(v2["version"], "v002")
            self.assertEqual(v2["based_on"], "v001")
            self.assertEqual(v2["parameters"]["recompute_window"]["start_frame"], 28)
            self.assertEqual(v2["parameters"]["recompute_window"]["end_frame"], 37)
            self.assertEqual(len(v2["identity_revisions"]), 2)

            before = evaluate(v1, truth)
            after = evaluate(v2, truth)
            self.assertGreater(before["identity_mismatch_count"], 0)
            self.assertEqual(after["identity_mismatch_count"], 0)
            self.assertTrue(after["no_fabricated_long_gap"])

    def test_short_occlusions_are_recovered(self) -> None:
        truth = generate_known_action()
        observed, _events = inject_occlusions_and_swaps(truth)
        from mocap_platform.calibration import calibrate_from_action
        from mocap_platform.models import Calibration

        bone_lengths, joint_limits = calibrate_from_action(truth)
        calibration = Calibration("c", "capture", bone_lengths, joint_limits).to_dict()
        result = reconstruct("a", "p", "capture", observed, calibration)
        statuses = result.statuses
        self.assertTrue(
            all(statuses[index]["l_wrist"] == "recovered" for index in range(8, 12))
        )
        self.assertTrue(
            all(statuses[index]["r_knee"] == "recovered" for index in range(25, 28))
        )

    def test_multi_person_confirmation_propagation_revocation_and_version_review(self) -> None:
        truth = generate_multi_person_action()
        observed, events, ambiguity_windows = inject_multi_person_crossing(truth)
        with tempfile.TemporaryDirectory() as directory:
            demo(directory)
            store = PlatformStore(directory)
            action = "demo_multi_person_action"
            pipeline = "crossing_identity_review"
            capture = store.load_capture("demo_capture_0002")

            v1 = json.loads(
                Path(
                    directory,
                    "processing",
                    action,
                    pipeline,
                    "versions",
                    "v001",
                    "reconstruction.json",
                ).read_text(encoding="utf-8")
            )
            v2 = json.loads(
                Path(
                    directory,
                    "processing",
                    action,
                    pipeline,
                    "versions",
                    "v002",
                    "reconstruction.json",
                ).read_text(encoding="utf-8")
            )
            v3 = json.loads(
                Path(
                    directory,
                    "processing",
                    action,
                    pipeline,
                    "versions",
                    "v003",
                    "reconstruction.json",
                ).read_text(encoding="utf-8")
            )
            report = json.loads(
                Path(
                    directory,
                    "processing",
                    action,
                    pipeline,
                    "evaluation_report.json",
                ).read_text(encoding="utf-8")
            )

            self.assertEqual(capture["observations"][20]["p0_l_wrist"], observed[20]["p0_l_wrist"])
            self.assertNotEqual(
                capture["observations"][20]["p0_l_wrist"],
                truth[20]["p0_l_wrist"],
            )
            self.assertEqual(v2["frames"][20]["p0_l_wrist"], truth[20]["p0_l_wrist"])
            self.assertEqual(v2["frames"][20]["p0_l_elbow"], truth[20]["p0_l_elbow"])
            self.assertEqual(v2["frames"][20]["p0_l_shoulder"], truth[20]["p0_l_shoulder"])

            for frame_index in range(22, 26):
                self.assertIsNone(capture["observations"][frame_index]["p0_l_wrist"])
                self.assertEqual(v2["statuses"][frame_index]["p0_l_wrist"], "recovered")
            self.assertEqual(v2["statuses"][12]["p0_head"], "uncertain")
            self.assertIsNone(v2["frames"][12]["p0_head"])
            self.assertEqual(v1["identity_statuses"][20]["p0_l_wrist"], "ambiguous")
            self.assertEqual(v2["identity_statuses"][20]["p0_l_wrist"], "confirmed")
            self.assertEqual(v2["identity_statuses"][40]["p0_l_wrist"], "ambiguous")
            self.assertEqual(v3["identity_statuses"][20]["p0_l_wrist"], "ambiguous")

            self.assertEqual(v2["based_on"], "v001")
            self.assertEqual(v3["based_on"], "v002")
            self.assertEqual(v2["identity_decisions"][0]["status"], "active")
            self.assertEqual(v3["identity_decisions"][0]["status"], "revoked")
            self.assertEqual(v2["parameters"]["recompute_window"], {"start_frame": 18, "end_frame": 31})
            self.assertEqual(v3["parameters"]["recompute_window"], {"start_frame": 18, "end_frame": 31})

            metrics = [item["metrics"] for item in report["comparison"]["versions"]]
            self.assertLess(metrics[1]["bone_length_rmse"], metrics[0]["bone_length_rmse"])
            self.assertAlmostEqual(
                metrics[2]["bone_length_rmse"], metrics[0]["bone_length_rmse"], places=9
            )
            self.assertEqual(metrics[0]["contact_frame_count"], metrics[1]["contact_frame_count"])
            self.assertEqual(metrics[1]["contact_frame_count"], metrics[2]["contact_frame_count"])
            self.assertEqual(metrics[0]["identity_mismatch_count"], 0)
            self.assertEqual(metrics[1]["identity_mismatch_count"], 0)
            self.assertEqual(metrics[2]["identity_mismatch_count"], 0)
            self.assertGreater(metrics[0]["ambiguous_identity_marker_frames"], metrics[1]["ambiguous_identity_marker_frames"])
            self.assertEqual(
                metrics[2]["ambiguous_identity_marker_frames"],
                metrics[0]["ambiguous_identity_marker_frames"],
            )

    def test_trajectory_reassignment_disputes_dependencies_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo(directory)
            store = PlatformStore(directory)
            action = "demo_multi_person_action"
            pipeline = "trajectory_reassignment_review"
            base = Path(directory, "processing", action, pipeline)

            def load(version: str) -> dict:
                return json.loads(
                    base.joinpath("versions", version, "reconstruction.json").read_text(
                        encoding="utf-8"
                    )
                )

            v2, v3, v4, v5, v6, v7 = (
                load(v) for v in ("v002", "v003", "v004", "v005", "v006", "v007")
            )
            truth = store.load_capture("demo_capture_0003")["ground_truth"]

            # A proposed reassignment previews impacts without changing frames.
            self.assertEqual(v3["frames"], v2["frames"])
            impact = v3["parameters"]["reassignment_impacts"][0]
            self.assertEqual(impact["reassignment_id"], "reassign_001_p1_right_arm_to_p0")
            self.assertEqual(impact["status"], "proposed")
            affected = impact["preview"]["affected_decisions"]
            self.assertEqual(len(affected), 1)
            self.assertEqual(affected[0]["decision_id"], "review_001_confirm_p0_left_arm")
            self.assertFalse(affected[0]["kept"])
            self.assertEqual(
                (affected[0]["start_frame"], affected[0]["end_frame"]), (20, 24)
            )
            self.assertTrue(impact["preview"]["affected_ambiguity_windows"])

            # Applying it reassigns only the selected window; kept decisions stay.
            self.assertEqual(v4["frames"][25]["p0_l_wrist"], truth[25]["p0_l_wrist"])
            self.assertEqual(v4["frames"][25]["p1_r_wrist"], truth[25]["p1_r_wrist"])
            self.assertEqual(v4["identity_statuses"][25]["p0_l_wrist"], "confirmed")
            applied_impact = v4["parameters"]["reassignment_impacts"][0]
            self.assertEqual(applied_impact["status"], "applied")
            self.assertEqual(
                applied_impact["kept_decision_ids"], ["review_001_confirm_p0_left_arm"]
            )
            decision_states = {
                item["decision_id"]: item["status"]
                for item in v4["parameters"]["decision_impacts"]
            }
            self.assertEqual(decision_states["review_001_confirm_p0_left_arm"], "applied")
            # Pose comparison covers the reassigned window before and after.
            pose = v4["parameters"]["pose_comparison"]
            self.assertEqual(len(pose), 1)
            self.assertEqual((pose[0]["start_frame"], pose[0]["end_frame"]), (20, 29))
            self.assertLess(
                pose[0]["bone_deviation_after"], pose[0]["bone_deviation_before"]
            )
            self.assertIn(
                {"start_frame": 40, "end_frame": 43},
                pose[0]["unresolved_intervals_after"],
            )

            # A second claim on the same trajectory is disputed, never shared.
            self.assertEqual(v6["frames"], v5["frames"])
            disputes = v6["parameters"]["reassignment_disputes"]
            self.assertEqual(len(disputes), 1)
            self.assertEqual(disputes[0]["reassignment_id"], "reassign_002_same_wrist_track")
            self.assertEqual(
                disputes[0]["disputes_with"], "reassign_001_p1_right_arm_to_p0"
            )
            self.assertEqual(disputes[0]["markers"], ["p1_r_wrist"])
            self.assertEqual(
                (disputes[0]["start_frame"], disputes[0]["end_frame"]), (22, 26)
            )
            self.assertTrue(disputes[0]["reassignment_trajectories"])
            self.assertTrue(disputes[0]["disputes_with_trajectories"])
            states = {
                item["reassignment_id"]: item["status"]
                for item in v6["parameters"]["reassignment_impacts"]
            }
            self.assertEqual(states["reassign_001_p1_right_arm_to_p0"], "applied")
            self.assertEqual(states["reassign_002_same_wrist_track"], "disputed")

            # Revoking the reassignment re-pends only dependent confirmations.
            self.assertEqual(v7["identity_statuses"][26]["p0_r_ankle"], "ambiguous")
            self.assertEqual(v7["identity_statuses"][22]["p0_l_wrist"], "confirmed")
            self.assertEqual(v7["identity_statuses"][41]["p0_l_wrist"], "ambiguous")
            pending = v7["parameters"]["pending_confirmations"]
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["decision_id"], "review_002_confirm_p0_right_ankle")
            self.assertEqual(pending[0]["reason"], "dependency_revoked")
            self.assertEqual(
                pending[0]["dependencies"], ["reassign_001_p1_right_arm_to_p0"]
            )
            final_states = {
                item["decision_id"]: item["status"]
                for item in v7["parameters"]["decision_impacts"]
            }
            self.assertEqual(final_states["review_001_confirm_p0_left_arm"], "applied")
            self.assertEqual(final_states["review_002_confirm_p0_right_ankle"], "pending")
            dependencies = {
                item["reassignment_id"]: item["depended_by"]
                for item in v7["parameters"]["reassignment_dependencies"]
            }
            self.assertEqual(
                dependencies["reassign_001_p1_right_arm_to_p0"],
                ["review_002_confirm_p0_right_ankle"],
            )

            # Reassignments persist immutably and reload for continued review.
            reassignments = store.load_reassignments(action, pipeline)
            self.assertEqual(len(reassignments), 2)
            statuses = {item["reassignment_id"]: item["status"] for item in reassignments}
            self.assertEqual(statuses["reassign_001_p1_right_arm_to_p0"], "revoked")
            self.assertEqual(statuses["reassign_002_same_wrist_track"], "revoked")
            from mocap_platform.models import TrajectoryReassignment

            with self.assertRaises(FileExistsError):
                store.save_reassignment(
                    action,
                    pipeline,
                    TrajectoryReassignment(
                        reassignment_id="reassign_001_p1_right_arm_to_p0",
                        source_person_id="p1",
                        target_person_id="p0",
                        joints=["r_wrist"],
                        start_frame=20,
                        end_frame=29,
                        reason="attempted overwrite",
                    ),
                )
            with self.assertRaises(FileExistsError):
                store.revoke_reassignment(
                    action, pipeline, "reassign_001_p1_right_arm_to_p0"
                )
            with self.assertRaises(FileNotFoundError):
                store.revoke_reassignment(action, pipeline, "reassign_999_missing")
            with self.assertRaises(FileExistsError):
                store.apply_reassignment(
                    action, pipeline, "reassign_001_p1_right_arm_to_p0"
                )

            # The export keeps the original capture, ranges, and provenance.
            export = json.loads(
                base.joinpath("review_export.json").read_text(encoding="utf-8")
            )
            self.assertEqual(export["capture_id"], "demo_capture_0003")
            self.assertEqual(
                export["original_capture"]["observations"],
                store.load_capture("demo_capture_0003")["observations"],
            )
            self.assertEqual(export["processing_version"], "v007")
            ranges = {
                item["reassignment_id"]: item for item in export["reassignment_ranges"]
            }
            self.assertEqual(
                ranges["reassign_001_p1_right_arm_to_p0"]["status"], "revoked"
            )
            self.assertEqual(
                (
                    ranges["reassign_001_p1_right_arm_to_p0"]["start_frame"],
                    ranges["reassign_001_p1_right_arm_to_p0"]["end_frame"],
                ),
                (20, 29),
            )
            provenance = {
                item["decision_id"]: item for item in export["decision_provenance"]
            }
            self.assertEqual(provenance["review_001_confirm_p0_left_arm"]["status"], "applied")
            self.assertEqual(
                provenance["review_002_confirm_p0_right_ankle"]["status"], "pending"
            )
            self.assertEqual(
                provenance["review_002_confirm_p0_right_ankle"]["depends_on"],
                ["reassign_001_p1_right_arm_to_p0"],
            )
            self.assertTrue(
                all(
                    item["source"] == "manual_review"
                    for item in provenance.values()
                )
            )

    def test_contact_evidence_schemes_simultaneous_contact_and_targeted_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo(directory)
            store = PlatformStore(directory)
            action = "demo_multi_person_action"
            pipeline = "contact_evidence_review"
            base = Path(directory, "processing", action, pipeline)

            def load(version: str) -> dict:
                return json.loads(
                    base.joinpath("versions", version, "reconstruction.json").read_text(
                        encoding="utf-8"
                    )
                )

            v1, v2, v3 = load("v001"), load("v002"), load("v003")
            schemes = {
                item["scheme_id"]: item
                for item in v2["parameters"]["scheme_reviews"]
            }
            results = {
                item["contact_id"]: item
                for item in v2["parameters"]["contact_review"]["contact_results"]
            }

            # Raw observed gaps stay missing and are explicitly itemized.
            self.assertEqual(
                v2["parameters"]["object_missing_intervals"]["ball_01"],
                [
                    {"start_frame": 22, "end_frame": 23},
                    {"start_frame": 27, "end_frame": 27},
                ],
            )
            short_contact = results[
                "contact_005_p1_hand_behind_ball_short_occlusion"
            ]
            self.assertEqual(short_contact["verdict"], "unobserved")
            self.assertEqual(short_contact["missing_observation_frames"], [22, 23])
            self.assertEqual(short_contact["observations_used"], [])

            # The swap interpretation follows the ball during the crossing.
            swap_contact = results[
                "contact_001_p0_left_hand_ball_during_crossing"
            ]
            self.assertEqual(swap_contact["verdict"], "still_plausible")
            self.assertEqual(swap_contact["observed_marker"], "p1_r_wrist")
            self.assertEqual(swap_contact["missing_observation_frames"], [27])
            self.assertTrue(swap_contact["supported_frames"])
            self.assertTrue(
                all(
                    item["observed_marker"] in {"p0_l_wrist", "p1_r_wrist"}
                    for item in swap_contact["observations_used"]
                )
            )

            # Independent schemes get independent geometric verdicts.
            swap_scheme = schemes["scheme_001_arm_swap"]
            keep_scheme = schemes["scheme_002_keep_labels"]
            self.assertEqual(swap_scheme["disputed_contact_intervals"], [])
            self.assertEqual(
                keep_scheme["disputed_contact_intervals"],
                [
                    {"start_frame": 25, "end_frame": 26},
                    {"start_frame": 28, "end_frame": 29},
                ],
            )
            keep_results = {
                item["contact_id"]: item
                for item in keep_scheme["contact_review"]["contact_results"]
            }
            self.assertEqual(
                keep_results[
                    "contact_001_p0_left_hand_ball_during_crossing"
                ]["verdict"],
                "geometrically_incompatible",
            )
            self.assertEqual(
                keep_results[
                    "contact_001_p0_left_hand_ball_during_crossing"
                ]["observed_marker"],
                "p0_l_wrist",
            )

            # Two people can touch the same object on the same frame.
            simultaneous = v2["parameters"]["contact_review"]["simultaneous_contacts"]
            self.assertEqual(len(simultaneous), 1)
            self.assertEqual(simultaneous[0]["frame"], 24)
            self.assertEqual(simultaneous[0]["object_id"], "ball_01")
            self.assertEqual(
                simultaneous[0]["contacts"],
                [
                    "contact_001_p0_left_hand_ball_during_crossing",
                    "contact_002_p1_left_hand_ball_during_grasp",
                ],
            )

            # The same body track still cannot be claimed by two identities.
            double_scheme = schemes["scheme_003_double_claim"]
            self.assertEqual(
                double_scheme["claim_conflict_intervals"],
                [{"start_frame": 22, "end_frame": 26}],
            )
            self.assertTrue(v2["parameters"]["contact_review"]["claim_conflicts"])

            # An automatic recommendation is retained as evidence but not confirmed.
            automatic = results["contact_004_automatic_near_object_hint"]
            self.assertEqual(automatic["source"], "automatic_recommendation")
            self.assertEqual(automatic["verdict"], "geometrically_incompatible")
            self.assertNotIn("contact_004_automatic_near_object_hint", v2["identity_decisions"])

            # Withdrawing one contact rechecks only dependent candidate/scheme.
            self.assertEqual(
                v3["parameters"]["contact_dependent_candidates"],
                ["candidate_no_swap_keep_observed_labels"],
            )
            self.assertEqual(
                v3["parameters"]["contact_dependent_schemes"],
                ["scheme_002_keep_labels"],
            )
            v3_ids = {
                item["contact_id"]
                for item in v3["parameters"]["contact_review"]["contact_results"]
            }
            self.assertNotIn(
                "contact_003_p0_left_hand_ball_after_separation", v3_ids
            )
            self.assertIn(
                "contact_001_p0_left_hand_ball_during_crossing", v3_ids
            )

            # Evidence and independent schemes persist for reopened review.
            contacts = store.load_contacts(action, pipeline)
            contact_statuses = {
                item["contact_id"]: item["status"] for item in contacts
            }
            self.assertEqual(
                contact_statuses[
                    "contact_003_p0_left_hand_ball_after_separation"
                ],
                "revoked",
            )
            self.assertEqual(
                contact_statuses[
                    "contact_001_p0_left_hand_ball_during_crossing"
                ],
                "active",
            )
            self.assertEqual(len(store.load_schemes(action, pipeline)), 4)
            self.assertIn("ball_01", store.load_object_tracks(action, pipeline))

            export = json.loads(
                base.joinpath("review_export.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                export["original_capture"]["observations"],
                store.load_capture("demo_capture_0003")["observations"],
            )
            self.assertEqual(
                [item["contact_id"] for item in export["manual_contact_notes"]],
                [item["contact_id"] for item in contacts],
            )
            exported_candidates = {
                item["candidate_id"]: item for item in export["identity_candidates"]
            }
            self.assertEqual(
                exported_candidates["candidate_swap_p0_left_is_p1_right_track"][
                    "decision"
                ],
                "adopted",
            )
            self.assertEqual(
                exported_candidates["candidate_no_swap_keep_observed_labels"][
                    "decision"
                ],
                "excluded",
            )
            self.assertIn(
                "geometrically incompatible",
                exported_candidates["candidate_no_swap_keep_observed_labels"][
                    "decision_reason"
                ],
            )
            self.assertEqual(len(export["identity_schemes"]), 4)
            self.assertEqual(
                export["object_tracks"]["ball_01"]["positions"][27], None
            )

            # Supplemental batches: pending v004, partial adoption v005,
            # one-batch revocation v006.
            def supplement_summary(version: str) -> dict:
                return load(version)["parameters"]["supplement_review"]

            v4_summary = supplement_summary("v004")
            self.assertEqual(v4_summary["supported_intervals"], [])
            self.assertEqual(
                v4_summary["still_missing_intervals"],
                [{"start_frame": 22, "end_frame": 23}, {"start_frame": 27, "end_frame": 27}],
            )
            self.assertTrue(v4_summary["pending_intervals"])

            v5_summary = supplement_summary("v005")
            self.assertEqual(
                v5_summary["supported_intervals"],
                [{"start_frame": 27, "end_frame": 27}],
            )
            self.assertEqual(
                v5_summary["conflicting_intervals"],
                [{"start_frame": 27, "end_frame": 27}],
            )
            self.assertEqual(v5_summary["still_missing_intervals"], [])
            self.assertIn(24, v5_summary["duplicate_frames"])

            v6_summary = supplement_summary("v006")
            self.assertEqual(v6_summary["conflicting_intervals"], [])
            self.assertEqual(
                v6_summary["supported_intervals"],
                [{"start_frame": 27, "end_frame": 27}],
            )
            v6_parameters = load("v006")["parameters"]
            self.assertIn(
                "candidate_supplement_reshoot_grip",
                v6_parameters["contact_dependent_candidates"],
            )
            self.assertIn(
                "scheme_004_reshoot_grip",
                v6_parameters["contact_dependent_schemes"],
            )

            # The export carries batches, adoptions and per-frame provenance.
            batch_status = {
                item["batch_id"]: item["status"]
                for item in export["supplement_batches"]
            }
            self.assertEqual(
                batch_status,
                {
                    "supplement_batch_001_long_overlap": "active",
                    "supplement_batch_002_short_partial_overlap": "revoked",
                },
            )
            self.assertEqual(len(export["supplement_adoptions"]), 2)
            provenance = export["frame_observation_provenance"][
                "contact_001_p0_left_hand_ball_during_crossing"
            ]
            self.assertEqual(
                provenance["27"]["object_batch_id"],
                "supplement_batch_001_long_overlap",
            )
            self.assertEqual(provenance["24"]["object_batch_id"], None)
            self.assertEqual(provenance["24"]["object_source"], "raw_observation")

    def test_contact_annotation_beyond_capture_length_stays_missing(self) -> None:
        # Five observed frames (0-4); annotations reach frame 7. Frames past
        # the capture length must remain missing evidence, never truncated and
        # never treated as separation.
        raw_observations = [
            {
                "p0_l_wrist": [0.0, 0.0, 0.0],
                "p1_r_wrist": [0.0, 0.0, 0.0],
            }
            for _ in range(4)
        ]
        # The p1 hand track is shorter than the capture: frame 4 lacks it.
        raw_observations.append({"p0_l_wrist": [0.0, 0.0, 0.0], "p1_r_wrist": None})
        # Both object tracks are shorter than the capture: only 3 frames long.
        objects = {
            "ball": [[0.0, 0.0, 0.0] for _ in range(3)],
            "box": [[0.0, 0.0, 0.0] for _ in range(3)],
        }

        single = ContactObservation(
            contact_id="contact_single_beyond_capture",
            person_id="p0",
            body_part="l_wrist",
            object_id="box",
            start_frame=1,
            end_frame=7,
            source="manual_review",
            reason="annotated past the end of the capture",
        )
        grasp_p0 = ContactObservation(
            contact_id="contact_grasp_p0",
            person_id="p0",
            body_part="l_wrist",
            object_id="ball",
            start_frame=2,
            end_frame=7,
            source="manual_review",
            reason="two-person grasp reaching past the capture length",
            simultaneous_with=["contact_grasp_p1"],
        )
        grasp_p1 = ContactObservation(
            contact_id="contact_grasp_p1",
            person_id="p1",
            body_part="r_wrist",
            object_id="ball",
            start_frame=2,
            end_frame=7,
            source="manual_review",
            reason="second hand on the same object past the capture length",
            simultaneous_with=["contact_grasp_p0"],
        )
        review = evaluate_contact_compatibility(
            [single, grasp_p0, grasp_p1], [], raw_observations, objects
        )
        results = {
            item["contact_id"]: item for item in review["contact_results"]
        }

        # Single-person contact: annotated range kept, tail frames missing.
        single_result = results["contact_single_beyond_capture"]
        self.assertEqual(
            (single_result["start_frame"], single_result["end_frame"]), (1, 7)
        )
        self.assertEqual(single_result["supported_frames"], [1, 2])
        self.assertEqual(single_result["missing_observation_frames"], [3, 4, 5, 6, 7])
        self.assertEqual(
            single_result["missing_intervals"], [{"start_frame": 3, "end_frame": 7}]
        )
        # Missing frames are not treated as separation: support still stands.
        self.assertEqual(single_result["verdict"], "still_plausible")
        # Only raw observed positions are used; nothing is interpolated.
        self.assertEqual(
            [item["frame"] for item in single_result["observations_used"]], [1, 2]
        )

        # Two-person grasp: the simultaneous check must not run past the
        # capture length, and per-contact results keep the missing tail.
        grasp_p0_result = results["contact_grasp_p0"]
        self.assertEqual(
            (grasp_p0_result["start_frame"], grasp_p0_result["end_frame"]), (2, 7)
        )
        self.assertEqual(grasp_p0_result["supported_frames"], [2])
        self.assertEqual(
            grasp_p0_result["missing_observation_frames"], [3, 4, 5, 6, 7]
        )
        grasp_p1_result = results["contact_grasp_p1"]
        self.assertEqual(grasp_p1_result["supported_frames"], [2])
        # Frame 4 misses the p1 hand track; frames 3 and 5-7 miss more.
        self.assertEqual(
            grasp_p1_result["missing_observation_frames"], [3, 4, 5, 6, 7]
        )
        self.assertEqual(
            review["simultaneous_contacts"],
            [
                {
                    "frame": 2,
                    "object_id": "ball",
                    "contacts": ["contact_grasp_p0", "contact_grasp_p1"],
                    "markers": ["p0_l_wrist", "p1_r_wrist"],
                    "body_batch_ids": [None, None],
                    "object_batch_id": None,
                }
            ],
        )

        # Save/reload keeps every manual decision; revoking one contact leaves
        # the others active, and the raw capture stays untouched.
        with tempfile.TemporaryDirectory() as directory:
            store = PlatformStore(directory)
            action = "demo_action"
            pipeline = "contact_beyond_capture"
            for contact in (single, grasp_p0, grasp_p1):
                store.save_contact(action, pipeline, contact)
            store.revoke_contact(
                action,
                pipeline,
                grasp_p1.contact_id,
                reason="withdrawn for recutting",
            )
            reloaded = {
                item["contact_id"]: item["status"]
                for item in store.load_contacts(action, pipeline)
            }
            self.assertEqual(
                reloaded,
                {
                    "contact_single_beyond_capture": "active",
                    "contact_grasp_p0": "active",
                    "contact_grasp_p1": "revoked",
                },
            )
            active_contacts = [
                ContactObservation(**item)
                for item in store.load_contacts(action, pipeline)
            ]
            recheck = evaluate_contact_compatibility(
                active_contacts, [], raw_observations, objects
            )
            recheck_ids = {
                item["contact_id"] for item in recheck["contact_results"]
            }
            self.assertEqual(
                recheck_ids,
                {"contact_single_beyond_capture", "contact_grasp_p0"},
            )
            recheck_single = recheck["contact_results"][0]
            self.assertEqual(
                recheck_single["missing_observation_frames"], [3, 4, 5, 6, 7]
            )
            # Raw observations were never overwritten by interpolated values.
            self.assertIsNone(raw_observations[4]["p1_r_wrist"])
            self.assertEqual(len(raw_observations), 5)


class SupplementalObservationTest(unittest.TestCase):
    def setUp(self) -> None:
        # Five frames. The object is missing on frames 1-2; each hand track is
        # missing on a different frame so the supplemental tracks handed in
        # later have different lengths and only partially overlap.
        self.observations = [
            {
                "p0_l_wrist": [0.0, 0.0, 0.0],
                "p1_r_wrist": [0.0, 0.0, 0.0],
            },
            {  # frame 1: p0 hand missing
                "p0_l_wrist": None,
                "p1_r_wrist": [0.0, 0.0, 0.0],
            },
            {  # frame 2: p1 hand missing
                "p0_l_wrist": [0.0, 0.0, 0.0],
                "p1_r_wrist": None,
            },
            {
                "p0_l_wrist": [0.0, 0.0, 0.0],
                "p1_r_wrist": [0.0, 0.0, 0.0],
            },
            {
                "p0_l_wrist": [0.0, 0.0, 0.0],
                "p1_r_wrist": [0.0, 0.0, 0.0],
            },
        ]
        self.objects = {
            "ball": [
                [0.0, 0.0, 0.0],
                None,
                None,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        }
        self.contacts = [
            ContactObservation(
                contact_id="contact_p0",
                person_id="p0",
                body_part="l_wrist",
                object_id="ball",
                start_frame=0,
                end_frame=2,
                source="manual_review",
                simultaneous_with=["contact_p1"],
            ),
            ContactObservation(
                contact_id="contact_p1",
                person_id="p1",
                body_part="r_wrist",
                object_id="ball",
                start_frame=0,
                end_frame=2,
                source="manual_review",
                simultaneous_with=["contact_p0"],
            ),
        ]

    def _batches(self) -> list[SupplementBatch]:
        long_batch = SupplementBatch(
            batch_id="batch_long",
            object_id="ball",
            source="reshoot_overhead",
            tracks=[
                SupplementTrack(
                    subject_id="ball",
                    subject_kind="object",
                    start_frame=1,
                    points=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ),
                SupplementTrack(
                    subject_id="p0_l_wrist",
                    subject_kind="body",
                    start_frame=1,
                    points=[[0.0, 0.0, 0.0]],
                ),
            ],
        )
        short_batch = SupplementBatch(
            batch_id="batch_short",
            object_id="ball",
            source="reshoot_side",
            tracks=[
                # Frame 2 duplicates the long ball point; a hypothetical extra
                # frame 3 would land on an already-observed frame.
                SupplementTrack(
                    subject_id="ball",
                    subject_kind="object",
                    start_frame=2,
                    points=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ),
                SupplementTrack(
                    subject_id="p1_r_wrist",
                    subject_kind="body",
                    start_frame=2,
                    points=[[0.0, 0.0, 0.0]],
                ),
            ],
        )
        return [long_batch, short_batch]

    def _adoptions(self) -> list[SupplementAdoption]:
        return [
            SupplementAdoption(
                adoption_id="adopt_long",
                batch_id="batch_long",
                object_id="ball",
                selections=[
                    SupplementSelection("ball", "object", 1, 1),
                    SupplementSelection("p0_l_wrist", "body", 1, 1),
                ],
            ),
            SupplementAdoption(
                adoption_id="adopt_short",
                batch_id="batch_short",
                object_id="ball",
                selections=[
                    SupplementSelection("ball", "object", 2, 3),
                    SupplementSelection("p1_r_wrist", "body", 2, 2),
                ],
            ),
        ]

    def test_pending_batches_leave_old_gaps_and_show_material_side_by_side(
        self,
    ) -> None:
        review = evaluate_contact_compatibility(
            self.contacts,
            [],
            self.observations,
            self.objects,
            supplement_index=resolve_supplement_points(self._batches(), []),
        )
        results = {item["contact_id"]: item for item in review["contact_results"]}
        # No adoption yet: frames 1-2 remain missing exactly as in v1.
        self.assertEqual(
            results["contact_p0"]["missing_observation_frames"], [1, 2]
        )
        self.assertEqual(results["contact_p0"]["supported_frames"], [0])
        self.assertEqual(results["contact_p0"]["verdict"], "still_plausible")
        # Pending material is listed per frame for reviewer selection.
        self.assertEqual(
            sorted(set(results["contact_p0"]["pending_supplement_frames"])),
            [1, 2],
        )
        pending = review["supplement_review"]["pending_intervals"]
        self.assertEqual(pending, [{"start_frame": 1, "end_frame": 3}])

    def test_partial_adoption_fills_only_chosen_gaps_with_provenance(self) -> None:
        review = evaluate_contact_compatibility(
            self.contacts,
            [],
            self.observations,
            self.objects,
            supplement_index=resolve_supplement_points(
                self._batches(), self._adoptions()
            ),
        )
        results = {item["contact_id"]: item for item in review["contact_results"]}
        # Both previously missing frames are supported by supplemental data.
        self.assertEqual(results["contact_p0"]["supported_frames"], [0, 1, 2])
        self.assertEqual(
            results["contact_p0"]["supported_by_supplement_frames"], [1, 2]
        )
        self.assertEqual(results["contact_p0"]["missing_observation_frames"], [])
        provenance = {
            item["frame"]: item
            for item in results["contact_p0"]["observations_used"]
        }
        self.assertIsNone(provenance[0]["body_batch_id"])
        self.assertEqual(provenance[0]["body_source"], "raw_observation")
        self.assertEqual(provenance[1]["body_batch_id"], "batch_long")
        self.assertEqual(provenance[1]["object_batch_id"], "batch_long")
        self.assertEqual(provenance[2]["object_batch_id"], "batch_short")
        # Frame 3's ball supplement repeats a frame the original analysis
        # already saw; it must not add evidence.
        self.assertIn(3, review["supplement_review"]["duplicate_frames"])
        # Both hands touch simultaneously on the supplemented frame 2.
        simultaneous = {
            item["frame"]: item
            for item in review["simultaneous_contacts"]
        }
        self.assertIn(2, simultaneous)
        self.assertEqual(
            simultaneous[2]["body_batch_ids"],
            # Frame 2's p0 hand was visible in the original capture (raw wins);
            # only p1's hand comes from the supplemental short batch.
            [None, "batch_short"],
        )

    def test_conflicting_adopted_ball_points_listed_side_by_side(self) -> None:
        batches = self._batches()
        # Make the long batch disagree with the short one at frame 2.
        batches[0].tracks[0].points[1] = [2.0, 0.0, 0.0]
        adoptions = self._adoptions()
        # Both batches must be adopted at frame 2 for the contradiction to be
        # visible (pending material is shown separately, not as a conflict).
        adoptions[0].selections[0].end_frame = 2
        review = evaluate_contact_compatibility(
            self.contacts,
            [],
            self.observations,
            self.objects,
            supplement_index=resolve_supplement_points(
                batches, adoptions
            ),
        )
        results = {item["contact_id"]: item for item in review["contact_results"]}
        p1_result = results["contact_p1"]
        # One adopted ball point (batch_short at origin) still supports contact;
        # the contradictory batch_long point is listed rather than chosen.
        self.assertEqual(p1_result["verdict"], "still_plausible")
        self.assertIn(2, p1_result["supported_by_supplement_frames"])
        conflict_frames = {
            item["frame"]: item
            for item in p1_result["conflicting_supplements"]
        }
        self.assertIn(2, conflict_frames)
        self.assertEqual(
            conflict_frames[2]["batches"], ["batch_long", "batch_short"]
        )
        self.assertEqual(
            review["supplement_review"]["conflicting_intervals"],
            [{"start_frame": 2, "end_frame": 2}],
        )

    def test_revoked_batch_only_removes_its_own_interpretations(self) -> None:
        batches = self._batches()
        batches[1].status = "revoked"
        review = evaluate_contact_compatibility(
            self.contacts,
            [],
            self.observations,
            self.objects,
            supplement_index=resolve_supplement_points(
                batches, self._adoptions()
            ),
        )
        results = {item["contact_id"]: item for item in review["contact_results"]}
        # Long-batch frame 1 support survives; frame 2 (short batch) is missing
        # again, and no short-batch duplicates remain.
        self.assertEqual(
            results["contact_p0"]["supported_by_supplement_frames"], [1]
        )
        self.assertEqual(results["contact_p1"]["missing_observation_frames"], [2])
        self.assertEqual(review["supplement_review"]["duplicate_frames"], [])
        # A candidate depending on the revoked batch is flagged for recheck.
        dependent = IdentityCandidate(
            candidate_id="candidate_needs_short",
            person_id="p1",
            joint_mapping={"r_wrist": "p1_r_wrist"},
            start_frame=0,
            end_frame=2,
            depends_on=["batch_short"],
        )
        self.assertEqual(
            candidates_dependent_on(
                [dependent], set(), set() | {"batch_short"}
            ),
            ["candidate_needs_short"],
        )

    def test_pending_save_reopen_partial_adopt_and_export_traceability(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = PlatformStore(directory)
            action = "demo_action"
            pipeline = "supplement_reopen_review"
            for batch in self._batches():
                store.save_supplement_batch(action, pipeline, batch)
            batches = [
                SupplementBatch.from_dict(data)
                for data in store.load_supplement_batches(action, pipeline)
            ]
            # Reopen: batches reload as active with no adoptions yet.
            self.assertEqual(
                [batch.batch_id for batch in batches],
                ["batch_long", "batch_short"],
            )
            adoption = self._adoptions()[0]
            store.save_supplement_adoption(action, pipeline, adoption)
            reloaded_adoptions = [
                SupplementAdoption.from_dict(data)
                for data in store.load_supplement_adoptions(action, pipeline)
            ]
            review = evaluate_contact_compatibility(
                self.contacts,
                [],
                self.observations,
                self.objects,
                supplement_index=resolve_supplement_points(
                    batches, reloaded_adoptions
                ),
            )
            results = {item["contact_id"]: item for item in review["contact_results"]}
            # Only the long batch is adopted: frame 1 filled, frame 2 remains
            # missing even though the short batch covers it (pending).
            self.assertEqual(
                results["contact_p0"]["supported_by_supplement_frames"], [1]
            )
            self.assertEqual(results["contact_p0"]["missing_observation_frames"], [2])
            self.assertEqual(
                results["contact_p0"]["pending_supplement_frames"], [2]
            )
            # Revoking the batch removes only its own evidence and adoption
            # effects; manual contact annotations stay active and untouched.
            store.revoke_supplement_batch(
                action, pipeline, "batch_long", reason="recut"
            )
            reloaded = {
                data["batch_id"]: data["status"]
                for data in store.load_supplement_batches(action, pipeline)
            }
            self.assertEqual(
                reloaded, {"batch_long": "revoked", "batch_short": "active"}
            )


if __name__ == "__main__":
    unittest.main()
