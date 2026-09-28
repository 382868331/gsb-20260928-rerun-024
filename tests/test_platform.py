import unittest
import tempfile

from mocap.model import Observation

from mocap.calibration import calibrate
from mocap.demo import JOINTS, BONES, JOINT_LIMITS, make_observations, truth_motion
from mocap.geometry import Vec3, angle_between
from mocap.model import CaptureRecord, IdentityRevision, SkeletonDefinition
from mocap.processing import ProcessingConfig, recover_motion, rmse
from mocap.storage import WorkspaceStore


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.truth = truth_motion()
        self.observations = make_observations(self.truth)
        self.capture = CaptureRecord("test", 60.0, tuple(self.observations))
        self.clean = CaptureRecord(
            "clean",
            60.0,
            tuple(
                Observation(frame, joint, position)
                for frame, row in enumerate(self.truth)
                for joint, position in row.items()
            ),
        )

    def test_versions_are_separate_and_original_capture_remains_queryable(self):
        with self.subTest(), tempfile.TemporaryDirectory() as directory:
            store = WorkspaceStore(directory)
            capture_v1 = store.save_capture(self.capture)
            clean_v1 = store.save_capture(self.clean)
            calibration = calibrate(
                "cal", self.clean, clean_v1, JOINTS, "pelvis", BONES,
                0, len(self.truth) - 1, JOINT_LIMITS
            )
            calibration_v1 = store.save_calibration(calibration)
            config = ProcessingConfig(max_short_gap=2, bone_tolerance=0.05)
            process_v1 = store.create_processing(
                "baseline", capture_v1, calibration_v1, config=config
            )
            revision = IdentityRevision(
                "fix", 6, 8, {"l_elbow": "r_elbow", "r_elbow": "l_elbow",
                              "l_wrist": "r_wrist", "r_wrist": "l_wrist"},
            )
            process_v2 = store.create_processing(
                "fixed", capture_v1, calibration_v1, (revision,), config, process_v1
            )

            self.assertEqual((capture_v1, calibration_v1, process_v1, process_v2),
                             ("v001", "v001", "v001", "v002"))
            raw = store.get_capture(capture_v1)
            self.assertEqual(len(raw.observations), len(self.capture.observations))
            revised = store.get_processing(process_v2)
            self.assertEqual(revised.parent_processing_version, process_v1)
            self.assertIn("fix", revised.frames[6]["r_elbow"].sources)

    def test_long_gap_remains_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            store = WorkspaceStore(directory)
            capture_v1 = store.save_capture(self.capture)
            clean_v1 = store.save_capture(self.clean)
            calibration_v1 = store.save_calibration(
                calibrate(
                    "cal", self.clean, clean_v1, JOINTS, "pelvis", BONES,
                    0, 16, JOINT_LIMITS
                )
            )
            process_v1 = store.create_processing(
                "baseline",
                capture_v1,
                calibration_v1,
                config=ProcessingConfig(max_short_gap=2, bone_tolerance=0.05),
            )
            processed = store.get_processing(process_v1)
            self.assertEqual(processed.frames[3]["l_wrist"].status, "interpolated")
            self.assertEqual(processed.frames[16]["r_wrist"].status, "unknown")
            self.assertEqual(processed.unknown_frames[-1], 16)
            self.assertTrue(any("left undetermined" in warning for warning in processed.warnings))

    def test_identity_revision_improves_swap_region(self):
        with tempfile.TemporaryDirectory() as directory:
            store = WorkspaceStore(directory)
            capture_v1 = store.save_capture(self.capture)
            clean_v1 = store.save_capture(self.clean)
            calibration_v1 = store.save_calibration(
                calibrate(
                    "cal", self.clean, clean_v1, JOINTS, "pelvis", BONES,
                    0, 16, JOINT_LIMITS
                )
            )
            config = ProcessingConfig(max_short_gap=2, bone_tolerance=0.05)
            before = store.create_processing("before", capture_v1, calibration_v1, config=config)
            revision = IdentityRevision(
                "fix", 6, 8, {"l_elbow": "r_elbow", "r_elbow": "l_elbow",
                              "l_wrist": "r_wrist", "r_wrist": "l_wrist"},
            )
            after = store.create_processing(
                "after", capture_v1, calibration_v1, (revision,), config, before
            )
            before_record = store.get_processing(before)
            after_record = store.get_processing(after)
            self.assertLess(
                rmse(list(after_record.frames)[6:9], self.truth[6:9]),
                rmse(list(before_record.frames)[6:9], self.truth[6:9]),
            )

    def test_joint_range_is_enforced(self):
        skeleton = SkeletonDefinition(
            joints=("pelvis", "spine", "head"),
            root="pelvis",
            bones=(("pelvis", "spine"), ("spine", "head")),
            bone_lengths={("pelvis", "spine"): 1.0, ("spine", "head"): 1.0},
            joint_limits={("pelvis", "spine", "head"): (0.0, 0.1)},
        )
        observations = [
            Observation(0, "pelvis", Vec3(0, 0, 0)),
            Observation(0, "spine", Vec3(0, 1, 0)),
            Observation(0, "head", Vec3(1, 1, 0)),
        ]
        frames, unknown_frames, warnings = recover_motion(1, observations, skeleton)
        angle = angle_between(
            frames[0]["spine"].position - frames[0]["pelvis"].position,
            frames[0]["head"].position - frames[0]["spine"].position,
        )
        self.assertEqual(unknown_frames, [])
        self.assertEqual(warnings, [])
        self.assertLessEqual(angle, 0.100001)


if __name__ == "__main__":
    unittest.main()
