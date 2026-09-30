from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mocap_platform.cli import demo
from mocap_platform.evaluation import evaluate
from mocap_platform.models import IdentityDecision, IdentityRevision
from mocap_platform.reconstruction import reconstruct
from mocap_platform.storage import PlatformStore
from mocap_platform.synthetic import generate_known_action, inject_occlusions_and_swaps
from mocap_platform.synthetic_multi import (
    generate_multi_person_action,
    inject_multi_person_crossing,
)


class PlatformIntegrationTest(unittest.TestCase):
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
