from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mocap_platform.cli import demo
from mocap_platform.evaluation import evaluate
from mocap_platform.models import (
    CaptureRecord,
    IdentityDecision,
    IdentityRevision,
    TrackReassignment,
)
from mocap_platform.export_adapters import export_decision_csv, export_review_bundle
from mocap_platform.reconstruction import reconstruct
from mocap_platform.storage import PlatformStore
from mocap_platform.synthetic import generate_known_action, inject_occlusions_and_swaps
from mocap_platform.synthetic_multi import (
    generate_multi_person_action,
    inject_multi_person_crossing,
)


class PlatformIntegrationTest(unittest.TestCase):
    def test_scoped_track_reassignment_dispute_withdrawal_and_export(self) -> None:
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

            v6, v7, v8 = (load(version) for version in ("v006", "v007", "v008"))
            capture = store.load_capture("demo_capture_0003")

            self.assertEqual(v6["identity_statuses"][22]["p0_l_wrist"], "pending")
            self.assertEqual(v6["identity_statuses"][22]["p1_r_wrist"], "pending")
            self.assertEqual(v6["frames"][22]["p0_l_wrist"], capture["ground_truth"][22]["p0_l_wrist"])
            self.assertEqual(v8["identity_statuses"][22]["p0_l_wrist"], "pending")
            self.assertEqual(v8["identity_statuses"][22]["p1_r_wrist"], "pending")
            self.assertEqual(v8["identity_statuses"][41]["p0_l_wrist"], "ambiguous")

            preview = next(
                item
                for item in v7["parameters"]["reassignment_previews"]
                if item["reassignment_id"] == "reassignment_003_ankle_p0_to_p1"
            )
            arm_preview = v6["parameters"]["reassignment_previews"][0]
            self.assertEqual(
                arm_preview["related_changes"]["marker_confirmations"][0]["state"],
                "pending_recheck",
            )
            self.assertTrue(preview["related_changes"]["occlusion_judgments"])
            self.assertTrue(preview["related_changes"]["skeleton_correspondences"])
            self.assertEqual(
                arm_preview["dependencies"][0]["requires"],
                "reassignment_002_arm_p1_to_p0",
            )

            self.assertEqual(v7["identity_statuses"][25]["p0_r_ankle"], "disputed")
            self.assertEqual(v7["identity_statuses"][25]["p1_r_ankle"], "disputed")
            blockers = v7["parameters"]["occupancy_blockers"]
            self.assertEqual(len(blockers), 1)
            self.assertEqual(
                blockers[0]["reassignment_id"], "reassignment_004_ankle_p1_to_p0"
            )
            self.assertEqual(
                blockers[0]["blocked_by"], "reassignment_003_ankle_p0_to_p1"
            )
            self.assertEqual(
                blockers[0]["markers"],
                [
                    "p0_r_ankle",
                    "p0_r_hip",
                    "p0_r_knee",
                    "p1_r_ankle",
                    "p1_r_hip",
                    "p1_r_knee",
                ],
            )

            impacts = {
                item["reassignment_id"]: item["status"]
                for item in v8["parameters"]["reassignment_impacts"]
            }
            self.assertEqual(impacts["reassignment_001_arm_p0_to_p1"], "revoked")
            self.assertEqual(impacts["reassignment_002_arm_p1_to_p0"], "revoked")
            self.assertEqual(impacts["reassignment_003_ankle_p0_to_p1"], "applied")
            self.assertEqual(impacts["reassignment_004_ankle_p1_to_p0"], "disputed")

            stored = store.load_reassignments(action, pipeline)
            self.assertEqual(len(stored), 4)
            stored_statuses = {item["reassignment_id"]: item["status"] for item in stored}
            self.assertEqual(stored_statuses["reassignment_001_arm_p0_to_p1"], "revoked")
            self.assertEqual(stored_statuses["reassignment_003_ankle_p0_to_p1"], "active")
            self.assertEqual(len(store.load_decisions(action, pipeline)), 3)
            with self.assertRaises(FileExistsError):
                store.save_reassignment(
                    action,
                    pipeline,
                    TrackReassignment(
                        reassignment_id="reassignment_001_arm_p0_to_p1",
                        source_person_id="p0",
                        target_person_id="p1",
                        marker="l_wrist",
                        start_frame=20,
                        end_frame=24,
                    ),
                )
            with self.assertRaises(FileExistsError):
                store.withdraw_reassignment(
                    action, pipeline, "reassignment_001_arm_p0_to_p1"
                )
            with self.assertRaises(FileNotFoundError):
                store.withdraw_reassignment(action, pipeline, "reassignment_999_missing")

            bundle_path = export_review_bundle(
                store,
                Path(directory, "exports", "review.json"),
                action,
                pipeline,
                v8,
            )
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(
                bundle["capture"]["original_observations"], capture["observations"]
            )
            self.assertEqual(len(bundle["reassignment_scopes"]), 4)
            self.assertEqual(bundle["frame_provenance"][25]["frame"], 25)
            self.assertTrue(bundle["decision_sources"]["dependent_confirmations"])

            csv_path = export_decision_csv(v8, Path(directory, "exports", "sources.csv"))
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("track_reassignment,reassignment_004_ankle_p1_to_p0,active", csv_text)

            report = json.loads(base.joinpath("evaluation_report.json").read_text(encoding="utf-8"))
            review = report["reassignment_review"]
            self.assertEqual(review["versions"][0]["version"], "v005")
            self.assertEqual(review["versions"][-1]["version"], "v008")
            self.assertIn("pose_metrics_change", review["before_after"])

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


if __name__ == "__main__":
    unittest.main()
