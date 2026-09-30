from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mocap_platform.cli import demo
from mocap_platform.evaluation import evaluate
from mocap_platform.identity import decision_effects, initial_identity_statuses
from mocap_platform.models import CaptureRecord, IdentityDecision, IdentityRevision
from mocap_platform.models import ProcessingVersion
from mocap_platform.reconstruction import reconstruct
from mocap_platform.storage import PlatformStore
from mocap_platform.synthetic import generate_known_action, inject_occlusions_and_swaps
from mocap_platform.synthetic_multi import (
    generate_multi_person_action,
    inject_multi_person_crossing,
)


class PlatformIntegrationTest(unittest.TestCase):
    def test_decision_scope_conflicts_revocation_and_reopen_review(self) -> None:
        markers = [
            "p0_l_wrist",
            "p0_r_ankle",
            "p1_r_wrist",
            "p1_l_ankle",
        ]
        frames = [{marker: [0.0, 0.0, 0.0] for marker in markers} for _ in range(30)]
        ankle_window = {
            "id": "ankle-crossing",
            "start_frame": 10,
            "end_frame": 12,
            "markers": ["p0_r_ankle", "p1_l_ankle"],
        }
        wrist_window = {
            "id": "wrist-reappearance",
            "start_frame": 20,
            "end_frame": 22,
            "markers": ["p0_l_wrist", "p1_r_wrist"],
        }
        adjacent_window = {
            "id": "adjacent-ankle",
            "start_frame": 13,
            "end_frame": 14,
            "markers": ["p0_r_ankle", "p1_l_ankle"],
        }
        ankle_confirmation = IdentityDecision(
            decision_id="confirm-ankle",
            person_id="p0",
            marker="r_ankle",
            start_frame=10,
            end_frame=12,
            decision="confirm",
            alternative_person_id="p1",
            related_markers=["r_ankle"],
            alternative_related_markers=["l_ankle"],
            reason="right ankle segment only",
        )
        statuses = initial_identity_statuses(
            frames,
            [ankle_window, wrist_window, adjacent_window],
            [ankle_confirmation],
        )
        self.assertEqual(statuses[11]["p0_r_ankle"], "confirmed")
        self.assertEqual(statuses[11]["p1_l_ankle"], "confirmed")
        self.assertEqual(statuses[21]["p0_l_wrist"], "ambiguous")
        self.assertEqual(statuses[21]["p1_r_wrist"], "ambiguous")
        self.assertEqual(statuses[13]["p0_r_ankle"], "ambiguous")

        first_wrist = IdentityDecision(
            decision_id="confirm-wrist-p0",
            person_id="p0",
            marker="l_wrist",
            start_frame=20,
            end_frame=22,
            decision="confirm",
            alternative_person_id="p1",
            related_markers=["l_wrist"],
            alternative_related_markers=["r_wrist"],
            reason="first reviewer",
        )
        second_wrist = IdentityDecision(
            decision_id="confirm-wrist-p1",
            person_id="p0",
            marker="l_wrist",
            start_frame=20,
            end_frame=22,
            decision="deny",
            alternative_person_id="p1",
            related_markers=["l_wrist"],
            alternative_related_markers=["r_wrist"],
            reason="conflicting reviewer rejects p0 ownership",
        )
        statuses = initial_identity_statuses(
            frames,
            [ankle_window, wrist_window],
            [ankle_confirmation, first_wrist, second_wrist],
        )
        self.assertEqual(statuses[21]["p0_l_wrist"], "conflicted")
        self.assertEqual(statuses[11]["p0_r_ankle"], "confirmed")
        effects = decision_effects(
            [ankle_confirmation, first_wrist, second_wrist],
            [ankle_window, wrist_window],
            len(frames),
        )
        wrist_effects = [item for item in effects if item["decision_id"] == "confirm-wrist-p0"][0]
        self.assertTrue(wrist_effects["conflicted"])
        self.assertEqual(wrist_effects["frame_ranges"], [{"start_frame": 20, "end_frame": 22}])
        self.assertEqual(
            wrist_effects["affected_trajectories"],
            [
                {"marker": "p0_l_wrist", "owner_person_id": "p0"},
                {"marker": "p1_r_wrist", "owner_person_id": "p1"},
            ],
        )

        revoked_ankle = IdentityDecision.from_dict(
            {**ankle_confirmation.to_dict(), "status": "revoked"}
        )
        reopened_statuses = initial_identity_statuses(
            frames, [ankle_window, wrist_window], [revoked_ankle]
        )
        self.assertEqual(reopened_statuses[11]["p0_r_ankle"], "ambiguous")
        self.assertEqual(reopened_statuses[21]["p0_l_wrist"], "ambiguous")
        reopened_effect = decision_effects(
            [revoked_ankle], [ankle_window, wrist_window], len(frames)
        )[0]
        self.assertEqual(len(reopened_effect["reopened_ambiguity_keys"]), 6)

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

            reopened_v3 = store.processing_version(action, pipeline, "v003")
            recalibrated_decision = IdentityDecision.from_dict(
                {
                    **v2["identity_decisions"][0],
                    "status": "active",
                    "decision_id": "recrossing-confirmation-v004",
                }
            )
            multi_v4 = reconstruct(
                action,
                pipeline,
                capture["capture_id"],
                observed,
                store.load_calibration(capture["capture_id"]),
                decisions=[recalibrated_decision],
                ambiguity_windows=ambiguity_windows,
                based_on=reopened_v3,
            )
            self.assertEqual(multi_v4.version, "v004")
            self.assertEqual(multi_v4.based_on, "v003")
            self.assertEqual(len(multi_v4.decisions), 2)
            self.assertEqual(multi_v4.identity_statuses[20]["p0_l_wrist"], "confirmed")
            self.assertEqual(multi_v4.identity_statuses[40]["p0_l_wrist"], "ambiguous")
            store.save_processing(multi_v4)
            with self.assertRaises(FileExistsError):
                store.save_capture(
                    CaptureRecord(
                        capture_id=capture["capture_id"],
                        action_name=action,
                        frame_count=0,
                        frame_rate=30.0,
                        markers=[],
                        observed_frames=[],
                    )
                )


if __name__ == "__main__":
    unittest.main()
