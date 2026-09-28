from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .geometry import Vec3
from .model import (
    CalibrationRecord,
    CaptureRecord,
    IdentityRevision,
    JointState,
    Observation,
    ProcessingRecord,
    SkeletonDefinition,
)
from .processing import ProcessingConfig, recover_motion


class WorkspaceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "captures").mkdir(parents=True, exist_ok=True)
        (self.root / "calibrations").mkdir(parents=True, exist_ok=True)
        (self.root / "processings").mkdir(parents=True, exist_ok=True)

    def _next_version(self, directory: Path) -> str:
        existing = sorted(path.name for path in directory.iterdir() if path.is_dir())
        return f"v{len(existing) + 1:03d}"

    def save_capture(self, record: CaptureRecord) -> str:
        version = self._next_version(self.root / "captures")
        path = self.root / "captures" / version / "record.json"
        path.parent.mkdir(parents=True)
        payload = {
            "capture_id": record.capture_id,
            "fps": record.fps,
            "metadata": record.metadata,
            "observations": [
                {
                    "frame": item.frame,
                    "marker": item.marker,
                    "position": item.position.to_list(),
                    "confidence": item.confidence,
                }
                for item in record.observations
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return version

    def get_capture(self, version: str) -> CaptureRecord:
        payload = json.loads(
            (self.root / "captures" / version / "record.json").read_text(encoding="utf-8")
        )
        observations = tuple(
            Observation(
                frame=item["frame"],
                marker=item["marker"],
                position=Vec3.from_list(item["position"]),
                confidence=item["confidence"],
            )
            for item in payload["observations"]
        )
        return CaptureRecord(payload["capture_id"], payload["fps"], observations, payload.get("metadata", {}))

    def save_calibration(self, record: CalibrationRecord) -> str:
        version = self._next_version(self.root / "calibrations")
        skeleton = record.skeleton
        payload = {
            "calibration_id": record.calibration_id,
            "capture_version": record.capture_version,
            "frame_start": record.frame_start,
            "frame_end": record.frame_end,
            "skeleton": {
                "joints": list(skeleton.joints),
                "root": skeleton.root,
                "bones": [list(bone) for bone in skeleton.bones],
                "bone_lengths": {"|".join(key): value for key, value in skeleton.bone_lengths.items()},
                "joint_limits": {
                    "|".join(key): list(value) for key, value in skeleton.joint_limits.items()
                },
            },
        }
        path = self.root / "calibrations" / version / "record.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return version

    def get_calibration(self, version: str) -> CalibrationRecord:
        payload = json.loads(
            (self.root / "calibrations" / version / "record.json").read_text(encoding="utf-8")
        )
        raw = payload["skeleton"]
        skeleton = SkeletonDefinition(
            joints=tuple(raw["joints"]),
            root=raw["root"],
            bones=tuple((bone[0], bone[1]) for bone in raw["bones"]),
            bone_lengths={
                tuple(key.split("|")): value for key, value in raw["bone_lengths"].items()
            },
            joint_limits={
                tuple(key.split("|")): tuple(value) for key, value in raw["joint_limits"].items()
            },
        )
        return CalibrationRecord(
            payload["calibration_id"],
            payload["capture_version"],
            skeleton,
            payload["frame_start"],
            payload["frame_end"],
        )

    def create_processing(
        self,
        processing_id: str,
        capture_version: str,
        calibration_version: str,
        revisions: tuple[IdentityRevision, ...] = (),
        config: ProcessingConfig | None = None,
        parent_processing_version: str | None = None,
    ) -> str:
        capture = self.get_capture(capture_version)
        calibration = self.get_calibration(calibration_version)
        frames, unknown_frames, warnings = recover_motion(
            capture.frame_count,
            list(capture.observations),
            calibration.skeleton,
            revisions,
            config,
        )
        record = ProcessingRecord(
            processing_id,
            capture_version,
            calibration_version,
            parent_processing_version,
            revisions,
            tuple(frames),
            tuple(unknown_frames),
            tuple(warnings),
        )
        version = self._next_version(self.root / "processings")
        path = self.root / "processings" / version / "record.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(self._processing_to_dict(record), ensure_ascii=False, indent=2), encoding="utf-8")
        return version

    def get_processing(self, version: str) -> ProcessingRecord:
        payload = json.loads(
            (self.root / "processings" / version / "record.json").read_text(encoding="utf-8")
        )
        revisions = tuple(
            IdentityRevision(
                revision_id=item["revision_id"],
                frame_start=item["frame_start"],
                frame_end=item["frame_end"],
                mapping=item["mapping"],
                reason=item.get("reason", ""),
            )
            for item in payload["revisions"]
        )
        frames = tuple(
            {
                joint: JointState(
                    Vec3.from_list(state["position"]) if state["position"] is not None else None,
                    state["status"],
                    tuple(state["sources"]),
                )
                for joint, state in row.items()
            }
            for row in payload["frames"]
        )
        return ProcessingRecord(
            payload["processing_id"],
            payload["capture_version"],
            payload["calibration_version"],
            payload.get("parent_processing_version"),
            revisions,
            frames,
            tuple(payload["unknown_frames"]),
            tuple(payload["warnings"]),
        )

    def _processing_to_dict(self, record: ProcessingRecord) -> dict:
        return {
            "processing_id": record.processing_id,
            "capture_version": record.capture_version,
            "calibration_version": record.calibration_version,
            "parent_processing_version": record.parent_processing_version,
            "revisions": [asdict(revision) for revision in record.revisions],
            "unknown_frames": list(record.unknown_frames),
            "warnings": list(record.warnings),
            "frames": [
                {
                    joint: {
                        "position": state.position.to_list() if state.position is not None else None,
                        "status": state.status,
                        "sources": list(state.sources),
                    }
                    for joint, state in row.items()
                }
                for row in record.frames
            ],
        }
