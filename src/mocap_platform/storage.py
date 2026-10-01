from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import (
    Calibration,
    CaptureRecord,
    IdentityDecision,
    ProcessingVersion,
    TrackReassignment,
)


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

    def decisions_dir(self, action_name: str, pipeline_name: str) -> Path:
        return self.processing_dir(action_name, pipeline_name) / "decisions"

    def save_decision(
        self, action_name: str, pipeline_name: str, decision: IdentityDecision
    ) -> Path:
        path = self.decisions_dir(action_name, pipeline_name) / f"{decision.decision_id}.json"
        if path.exists():
            raise FileExistsError(f"decision is immutable and already exists: {path}")
        atomic_write_json(path, decision.to_dict())
        return path

    def revoke_decision(
        self,
        action_name: str,
        pipeline_name: str,
        decision_id: str,
        reason: str = "",
    ) -> Path:
        directory = self.decisions_dir(action_name, pipeline_name)
        decision_path = directory / f"{decision_id}.json"
        if not decision_path.exists():
            raise FileNotFoundError(f"cannot revoke unknown decision: {decision_path}")
        path = directory / f"{decision_id}.revocation.json"
        if path.exists():
            raise FileExistsError(f"decision already revoked: {path}")
        from .models import utc_now

        atomic_write_json(
            path,
            {
                "kind": "decision_revocation",
                "decision_id": decision_id,
                "reason": reason,
                "revoked_at": utc_now(),
            },
        )
        return path

    def load_decisions(self, action_name: str, pipeline_name: str) -> list[dict[str, Any]]:
        directory = self.decisions_dir(action_name, pipeline_name)
        if not directory.exists():
            return []
        revoked = {
            path.name[: -len(".revocation.json")]
            for path in directory.glob("*.revocation.json")
        }
        decisions: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".revocation.json"):
                continue
            if path.name.endswith(".reassignment.withdrawal.json"):
                continue
            data = read_json(path)
            if data.get("kind") == "track_reassignment":
                continue
            if data["decision_id"] in revoked:
                data["status"] = "revoked"
            decisions.append(data)
        return decisions

    def save_reassignment(
        self,
        action_name: str,
        pipeline_name: str,
        reassignment: TrackReassignment,
    ) -> Path:
        path = (
            self.decisions_dir(action_name, pipeline_name)
            / f"{reassignment.reassignment_id}.reassignment.json"
        )
        if path.exists():
            raise FileExistsError(f"reassignment is immutable and already exists: {path}")
        atomic_write_json(path, reassignment.to_dict())
        return path

    def withdraw_reassignment(
        self,
        action_name: str,
        pipeline_name: str,
        reassignment_id: str,
        reason: str = "",
    ) -> Path:
        directory = self.decisions_dir(action_name, pipeline_name)
        reassignment_path = directory / f"{reassignment_id}.reassignment.json"
        if not reassignment_path.exists():
            raise FileNotFoundError(
                f"cannot withdraw unknown reassignment: {reassignment_path}"
            )
        path = directory / f"{reassignment_id}.reassignment.withdrawal.json"
        if path.exists():
            raise FileExistsError(f"reassignment already withdrawn: {path}")
        from .models import utc_now

        atomic_write_json(
            path,
            {
                "kind": "track_reassignment_withdrawal",
                "reassignment_id": reassignment_id,
                "reason": reason,
                "withdrawn_at": utc_now(),
            },
        )
        return path

    def load_reassignments(
        self, action_name: str, pipeline_name: str
    ) -> list[dict[str, Any]]:
        directory = self.decisions_dir(action_name, pipeline_name)
        if not directory.exists():
            return []
        withdrawn = {
            path.name[: -len(".reassignment.withdrawal.json")]
            for path in directory.glob("*.reassignment.withdrawal.json")
        }
        records = []
        for path in sorted(directory.glob("*.reassignment.json")):
            data = read_json(path)
            if data["reassignment_id"] in withdrawn:
                data["status"] = "revoked"
            records.append(data)
        return records

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
            "track_reassignments": [item.to_dict() for item in result.reassignments],
        }
