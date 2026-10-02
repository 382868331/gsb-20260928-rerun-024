from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mocap_platform.cli import demo
from mocap_platform.evaluation import evaluate
from mocap_platform.models import CaptureRecord, IdentityDecision, IdentityRevision
from mocap_platform.reconstruction import reconstruct
from mocap_platform.storage import PlatformStore
from mocap_platform.synthetic import generate_known_action, inject_occlusions_and_swaps
from mocap_platform.synthetic_multi import (
    generate_multi_person_action,
    inject_multi_person_crossing,
)


class PlatformIntegrationTest(unittest.TestCase):
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

    def test_contact_evidence_scenarios_recheck_and_export(self) -> None:
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

            v1, v2, v3, v4 = (load(v) for v in ("v001", "v002", "v003", "v004"))

            # Contact comparison separates incompatible, missing, and unobserved.
            baseline = {
                item["contact_id"]: item
                for item in v1["parameters"]["contact_review"]["assessments"]
            }
            wrist = baseline["contact_001_p0_left_wrist_crate"]
            self.assertEqual(wrist["status"], "incompatible")
            self.assertEqual(
                wrist["conflict_intervals"], [{"start_frame": 20, "end_frame": 20}]
            )
            self.assertEqual(wrist["missing_frames"], [21, 22])
            self.assertTrue(wrist["used_observations"])
            self.assertNotIn(
                21, [item["frame"] for item in wrist["used_observations"]]
            )
            self.assertEqual(
                baseline["contact_003_p0_right_ankle_crate"]["status"], "unobserved"
            )
            self.assertEqual(
                baseline["contact_003_p0_right_ankle_crate"]["missing_frames"],
                [26, 27, 28],
            )

            # The contact-consistent reassignment makes the contact compatible.
            reassigned = {
                item["contact_id"]: item
                for item in v2["parameters"]["contact_review"]["assessments"]
            }
            self.assertEqual(
                reassigned["contact_001_p0_left_wrist_crate"]["status"], "compatible"
            )
            self.assertEqual(
                reassigned["contact_002_p1_right_wrist_crate"]["missing_frames"],
                [26, 27, 28],
            )
            # Simultaneous contact with one object by two people is allowed.
            simultaneous = v2["parameters"]["contact_review"]["simultaneous_contacts"]
            shared = [
                item
                for item in simultaneous
                if item["contact_ids"]
                == [
                    "contact_001_p0_left_wrist_crate",
                    "contact_002_p1_right_wrist_crate",
                ]
            ]
            self.assertEqual(len(shared), 1)
            self.assertTrue(shared[0]["allowed"])
            self.assertEqual(
                (shared[0]["start_frame"], shared[0]["end_frame"]), (23, 24)
            )

            # Scenarios keep alternative explanations with distinct outcomes.
            scenarios = {
                item["scenario_id"]: item
                for item in v3["parameters"]["scenario_assessments"]
            }
            self.assertEqual(
                scenarios["scenario_a_reassigned_contact"]["automatic_assessment"],
                "compatible",
            )
            self.assertEqual(
                scenarios["scenario_b_no_reassignment"]["automatic_assessment"],
                "incompatible",
            )
            self.assertEqual(
                scenarios["scenario_c_duplicate_claim"]["automatic_assessment"],
                "contested",
            )
            claims = scenarios["scenario_c_duplicate_claim"][
                "trajectory_claim_conflicts"
            ]
            self.assertEqual(len(claims), 1)
            self.assertEqual(claims[0]["markers"], ["p1_r_wrist"])
            self.assertEqual(
                (claims[0]["start_frame"], claims[0]["end_frame"]), (22, 26)
            )
            self.assertIn("never counts", scenarios["scenario_a_reassigned_contact"]["note"])

            # Revoking one contact re-pends only dependent confirmations and
            # re-checks only scenarios that rely on that contact.
            pending = v4["parameters"]["pending_confirmations"]
            self.assertEqual(len(pending), 1)
            self.assertEqual(
                pending[0]["decision_id"], "contact_decision_001_confirm_p0_right_ankle"
            )
            self.assertEqual(pending[0]["reason"], "contact_revoked")
            self.assertEqual(
                pending[0]["dependencies"], ["contact_002_p1_right_wrist_crate"]
            )
            recheck = v4["parameters"]["review_annotations"]["contact_recheck"]
            self.assertEqual(
                recheck["rechecked_scenarios"], ["scenario_a_reassigned_contact"]
            )
            self.assertEqual(
                recheck["unaffected_scenarios"],
                ["scenario_b_no_reassignment", "scenario_c_duplicate_claim"],
            )
            reassessed = {
                item["scenario_id"]: item
                for item in v4["parameters"]["scenario_assessments"]
            }
            self.assertEqual(list(reassessed), ["scenario_a_reassigned_contact"])
            reassessed_contacts = {
                item["contact_id"]: item["status"]
                for item in reassessed["scenario_a_reassigned_contact"][
                    "contact_assessments"
                ]
            }
            self.assertNotIn("contact_002_p1_right_wrist_crate", reassessed_contacts)
            self.assertEqual(
                reassessed_contacts["contact_001_p0_left_wrist_crate"], "compatible"
            )

            # Contacts and scenarios persist immutably and reload with status.
            contacts = store.load_contacts(action, pipeline)
            self.assertEqual(len(contacts), 3)
            statuses = {item["contact_id"]: item["status"] for item in contacts}
            self.assertEqual(statuses["contact_002_p1_right_wrist_crate"], "revoked")
            self.assertEqual(statuses["contact_001_p0_left_wrist_crate"], "active")
            from mocap_platform.models import ContactEvidence, InterpretationScenario

            with self.assertRaises(FileExistsError):
                store.save_contact(
                    action,
                    pipeline,
                    ContactEvidence(
                        contact_id="contact_001_p0_left_wrist_crate",
                        person_id="p0",
                        joint="l_wrist",
                        object_id="obj_crate",
                        start_frame=18,
                        end_frame=24,
                        reason="attempted overwrite",
                    ),
                )
            with self.assertRaises(FileExistsError):
                store.revoke_contact(
                    action, pipeline, "contact_002_p1_right_wrist_crate"
                )
            with self.assertRaises(FileNotFoundError):
                store.revoke_contact(action, pipeline, "contact_999_missing")

            scenarios_loaded = store.load_scenarios(action, pipeline)
            scenario_statuses = {
                item["scenario_id"]: item["status"] for item in scenarios_loaded
            }
            self.assertEqual(scenario_statuses["scenario_a_reassigned_contact"], "adopted")
            self.assertEqual(scenario_statuses["scenario_b_no_reassignment"], "excluded")
            self.assertEqual(scenario_statuses["scenario_c_duplicate_claim"], "open")
            with self.assertRaises(FileExistsError):
                store.save_scenario(
                    action,
                    pipeline,
                    InterpretationScenario(
                        scenario_id="scenario_a_reassigned_contact",
                        label="attempted overwrite",
                        reason="attempted overwrite",
                    ),
                )
            with self.assertRaises(FileExistsError):
                store.adopt_scenario(action, pipeline, "scenario_a_reassigned_contact")
            with self.assertRaises(FileNotFoundError):
                store.adopt_scenario(action, pipeline, "scenario_999_missing")

            # The export keeps observations, contact annotations, and candidate
            # adoption or exclusion with reasons.
            export = json.loads(
                base.joinpath("review_export.json").read_text(encoding="utf-8")
            )
            self.assertEqual(export["capture_id"], "demo_capture_0004")
            capture = store.load_capture("demo_capture_0004")
            self.assertEqual(
                export["original_capture"]["observations"], capture["observations"]
            )
            self.assertEqual(
                export["original_capture"]["object_frames"], capture["object_frames"]
            )
            self.assertEqual(export["original_capture"]["objects"], ["obj_crate"])
            evidence = {
                item["contact_id"]: item for item in export["contact_evidence"]
            }
            self.assertEqual(
                evidence["contact_002_p1_right_wrist_crate"]["status"], "revoked"
            )
            self.assertTrue(
                all(item["source"] == "manual_review" for item in evidence.values())
            )
            exported_scenarios = {
                item["scenario_id"]: item for item in export["identity_scenarios"]
            }
            self.assertEqual(
                exported_scenarios["scenario_a_reassigned_contact"]["status"], "adopted"
            )
            self.assertEqual(
                exported_scenarios["scenario_b_no_reassignment"]["status"], "excluded"
            )
            self.assertEqual(
                exported_scenarios["scenario_c_duplicate_claim"]["status"], "open"
            )
            self.assertEqual(
                exported_scenarios["scenario_a_reassigned_contact"]["reassignment_ids"],
                ["contact_reassign_001_p1_right_arm_to_p0"],
            )


if __name__ == "__main__":
    unittest.main()
