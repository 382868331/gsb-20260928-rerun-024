from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mocap_platform.cli import demo
from mocap_platform.evaluation import evaluate
from mocap_platform.reconstruction import reconstruct
from mocap_platform.storage import PlatformStore
from mocap_platform.synthetic import (
    generate_two_person_action,
    generate_known_action,
    inject_occlusions_and_swaps,
)


class TwoPersonCrossingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        demo(self.tempdir.name)
        self.store = PlatformStore(self.tempdir.name)
        base = Path(
            self.tempdir.name,
            "processing",
            "demo_action",
            "two_person_crossing",
            "versions",
        )
        self.capture = json.loads(
            Path(self.tempdir.name, "captures", "demo_capture_0001", "v001_capture.json")
            .read_text(encoding="utf-8")
        )
        self.v1 = json.loads((base / "v001" / "reconstruction.json").read_text(encoding="utf-8"))
        self.v2 = json.loads((base / "v002" / "reconstruction.json").read_text(encoding="utf-8"))
        self.v3 = json.loads((base / "v003" / "reconstruction.json").read_text(encoding="utf-8"))
        self.report = json.loads(
            Path(
                self.tempdir.name,
                "processing",
                "demo_action",
                "two_person_crossing",
                "evaluation_report.json",
            ).read_text(encoding="utf-8")
        )
        self.truth = generate_two_person_action()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_all_three_versions_and_capture_are_reviewable(self) -> None:
        self.assertEqual(self.v1["version"], "v001")
        self.assertEqual(self.v2["version"], "v002")
        self.assertEqual(self.v3["version"], "v003")
        self.assertIsNone(self.v1["based_on"])
        self.assertEqual(self.v2["based_on"], "v001")
        self.assertEqual(self.v3["based_on"], "v002")
        lineage = Path(
            self.tempdir.name,
            "processing",
            "demo_action",
            "two_person_crossing",
            "versions",
            "v003",
            "lineage.json",
        )
        self.assertTrue(lineage.exists())

    def test_raw_capture_is_never_rewritten(self) -> None:
        # The injected swap remains exactly as collected in the immutable
        # capture record after all revisions.
        observed = self.capture["observations"]
        truth = self.capture["ground_truth"]
        self.assertNotEqual(observed[20]["pA_r_wrist"], truth[20]["pA_r_wrist"])
        self.assertEqual(observed[20]["pA_r_wrist"], truth[20]["pB_l_wrist"])
        self.assertIsNone(observed[25]["pA_r_wrist"])
        # But v2 must have corrected the derived reconstruction only.
        self.assertEqual(self.v2["frames"][20]["pA_r_wrist"], truth[20]["pA_r_wrist"])

    def test_manual_decision_propagates_to_entire_limb_group(self) -> None:
        for joint in ("pA_r_shoulder", "pA_r_elbow", "pA_r_wrist"):
            for frame_index in range(18, 24):
                self.assertAlmostEqual(
                    self.v2["frames"][frame_index][joint][0],
                    self.truth[frame_index][joint][0],
                    places=5,
                )
                self.assertAlmostEqual(
                    self.v2["frames"][frame_index][joint][1],
                    self.truth[frame_index][joint][1],
                    places=5,
                )
        revision = next(
            r for r in self.v2["identity_revisions"] if r["revision_id"] == "rev_deny_cross_01"
        )
        self.assertTrue(revision["propagated"])
        self.assertEqual(revision["group"], "r_arm")

    def test_short_simultaneous_occlusion_is_recovered(self) -> None:
        for frame_index in range(24, 28):
            self.assertEqual(self.v2["statuses"][frame_index]["pA_r_wrist"], "recovered")
            self.assertEqual(self.v2["statuses"][frame_index]["pB_l_wrist"], "recovered")

    def test_long_gap_stays_uncertain(self) -> None:
        for frame_index in range(40, 52):
            self.assertIsNone(self.v1["frames"][frame_index]["pA_head"])
            self.assertEqual(self.v1["statuses"][frame_index]["pA_head"], "uncertain")
            self.assertIsNone(self.v2["frames"][frame_index]["pA_head"])
            self.assertIsNone(self.v3["frames"][frame_index]["pA_head"])

    def test_equally_plausible_reappearance_is_uncertain_until_confirmed(self) -> None:
        for frame_index in range(28, 32):
            self.assertIsNone(self.v1["frames"][frame_index]["pA_r_wrist"])
            self.assertEqual(
                self.v1["statuses"][frame_index]["pA_r_wrist"], "uncertain"
            )
        ambiguity = self.v1["ambiguities"][0]
        self.assertEqual((ambiguity["start_frame"], ambiguity["end_frame"]), (28, 31))
        self.assertEqual(ambiguity["resolution"], "unresolved")
        self.assertAlmostEqual(
            ambiguity["current_score"], ambiguity["swapped_score"], delta=0.25
        )
        # Human confirmation commits the window to concrete identities.
        for frame_index in range(28, 32):
            self.assertIsNotNone(self.v2["frames"][frame_index]["pA_r_wrist"])
            self.assertEqual(
                self.v2["statuses"][frame_index]["pA_r_wrist"], "observed"
            )
        self.assertTrue(
            self.v2["ambiguities"][0]["resolution"].startswith("manual:")
        )

    def test_undo_regenerates_affected_window_but_keeps_other_decisions(self) -> None:
        # Revoked confirmation: reappearance goes back to uncertain.
        for frame_index in range(28, 32):
            self.assertIsNone(self.v3["frames"][frame_index]["pA_r_wrist"])
            self.assertEqual(
                self.v3["statuses"][frame_index]["pA_r_wrist"], "uncertain"
            )
        self.assertEqual(self.v3["ambiguities"][0]["resolution"], "unresolved")
        # The other manual decision (cross denial) survives.
        remaining = {r["revision_id"] for r in self.v3["identity_revisions"]}
        self.assertIn("rev_deny_cross_01", remaining)
        self.assertNotIn("rev_confirm_reappear_01", remaining)
        for frame_index in range(18, 24):
            self.assertAlmostEqual(
                self.v3["frames"][frame_index]["pA_r_wrist"][0],
                self.truth[frame_index]["pA_r_wrist"][0],
                places=5,
            )
        # Only the union of the surviving decision and the revoked window was
        # rebuilt (deny window 18-23 minus margin through 31 + margin).
        self.assertEqual(
            self.v3["parameters"]["recompute_window"]["start_frame"], 12
        )
        self.assertEqual(
            self.v3["parameters"]["recompute_window"]["end_frame"], 37
        )
        self.assertEqual(
            self.v3["parameters"]["revoked_revision_ids"], ["rev_confirm_reappear_01"]
        )

    def test_metrics_compare_before_and_after_revision(self) -> None:
        before = self.report["v1_before_revision"]
        after = self.report["v2_after_confirmation"]
        undone = self.report["v3_after_undo"]

        self.assertGreater(
            before["bone_length_errors"]["person_A_mean_error"],
            after["bone_length_errors"]["person_A_mean_error"],
        )
        self.assertGreater(
            before["bone_length_errors"]["person_B_mean_error"],
            after["bone_length_errors"]["person_B_mean_error"],
        )
        contact_key = "pA_r_wrist__pB_l_wrist"
        self.assertEqual(before["contacts"][contact_key]["recall"], 0.0)
        self.assertEqual(after["contacts"][contact_key]["recall"], 1.0)
        self.assertEqual(after["contacts"][contact_key]["precision"], 1.0)
        self.assertEqual(undone["contacts"][contact_key]["recall"], 0.0)
        self.assertLess(
            before["identity_continuity"]["continuity_ratio"],
            after["identity_continuity"]["continuity_ratio"],
        )


class SinglePersonBackwardCompatibilityTest(unittest.TestCase):
    def test_legacy_gap_recovery_still_works(self) -> None:
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
            all(statuses[index]["head"] == "uncertain" for index in range(13, 24))
        )
        self.assertGreater(evaluate(result.to_dict(), truth)["identity_mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
