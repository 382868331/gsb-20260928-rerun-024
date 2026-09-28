from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mocap_platform.cli import demo
from mocap_platform.evaluation import evaluate
from mocap_platform.models import IdentityRevision
from mocap_platform.reconstruction import reconstruct
from mocap_platform.storage import PlatformStore
from mocap_platform.synthetic import generate_known_action, inject_occlusions_and_swaps


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


if __name__ == "__main__":
    unittest.main()
