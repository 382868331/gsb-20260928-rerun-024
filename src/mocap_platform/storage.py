from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import Calibration, CaptureRecord, ProcessingVersion


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class PlatformStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.captures = self.root / "captures"
        self.calibrations = self.root / "calibrations"
        self.processing = self.root / "processing"

    def save_capture(self, record: CaptureRecord) -> Path:
        path = self.captures / record.capture_id / "v001_capture.json"
        if path.exists():
            raise FileExistsError(f"capture is immutable and already exists: {path}")
        atomic_write_json(path, record.to_dict())
        return path

    def load_capture(self, capture_id: str) -> dict[str, Any]:
        return read_json(self.captures / capture_id / "v001_capture.json")

    def save_calibration(self, calibration: Calibration) -> Path:
        directory = self.calibrations / calibration.capture_id
        path = directory / f"{calibration.source_version}_calibration.json"
        if path.exists():
            raise FileExistsError(f"calibration version already exists: {path}")
        atomic_write_json(path, calibration.to_dict())
        return path

    def load_calibration(self, capture_id: str, version: str = "v001") -> dict[str, Any]:
        return read_json(self.calibrations / capture_id / f"{version}_calibration.json")

    def processing_dir(self, action_name: str, pipeline_name: str) -> Path:
        return self.processing / action_name / pipeline_name

    def save_processing(self, result: ProcessingVersion) -> Path:
        directory = (
            self.processing
            / result.action_name
            / result.pipeline_name
            / "versions"
            / result.version
        )
        path = directory / "reconstruction.json"
        if path.exists():
            raise FileExistsError(f"processing version already exists: {path}")
        atomic_write_json(path, result.to_dict())
        atomic_write_json(directory / "lineage.json", self._lineage(result))
        return path

    def latest_processing(self, action_name: str, pipeline_name: str) -> dict[str, Any]:
        base = self.processing_dir(action_name, pipeline_name) / "versions"
        versions = sorted(base.glob("v*/reconstruction.json"))
        if not versions:
            raise FileNotFoundError(f"no processing versions for {action_name}/{pipeline_name}")
        return read_json(versions[-1])

    def save_report(self, action_name: str, pipeline_name: str, report: dict[str, Any]) -> Path:
        path = self.processing_dir(action_name, pipeline_name) / "evaluation_report.json"
        atomic_write_json(path, report)
        return path

    @staticmethod
    def _lineage(result: ProcessingVersion) -> dict[str, Any]:
        return {
            "version": result.version,
            "based_on": result.based_on,
            "capture_id": result.capture_id,
            "calibration_version": result.calibration_version,
            "created_at": result.created_at,
            "revisions": [revision.to_dict() for revision in result.revisions],
            "identity_decisions": [decision.to_dict() for decision in result.decisions],
        }
