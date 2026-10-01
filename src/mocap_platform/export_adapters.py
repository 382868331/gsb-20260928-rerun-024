from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .storage import PlatformStore


def supported_adapters() -> tuple[str, ...]:
    return ("review_bundle", "decision_csv")


def build_review_bundle(
    store: PlatformStore,
    action_name: str,
    pipeline_name: str,
    version: dict[str, Any],
) -> dict[str, Any]:
    capture = store.load_capture(version["capture_id"])
    return {
        "kind": "identity_review_export",
        "capture": {
            "capture_id": capture["capture_id"],
            "source": capture.get("source"),
            "frame_count": capture["frame_count"],
            "frame_rate": capture["frame_rate"],
            "original_observations": capture["observations"],
        },
        "processing": {
            "action_name": action_name,
            "pipeline_name": pipeline_name,
            "version": version["version"],
            "based_on": version.get("based_on"),
            "capture_id": version["capture_id"],
            "calibration_version": version["calibration_version"],
        },
        "reassignment_scopes": version.get("track_reassignments", []),
        "decision_sources": {
            "identity_decisions": version.get("identity_decisions", []),
            "decision_impacts": version["parameters"].get("decision_impacts", []),
            "reassignment_impacts": version["parameters"].get("reassignment_impacts", []),
            "occupancy_blockers": version["parameters"].get("occupancy_blockers", []),
            "dependent_confirmations": version["parameters"].get(
                "dependent_confirmations", []
            ),
        },
        "frames": version["frames"],
        "frame_provenance": [
            {
                "frame": frame_index,
                "markers": {
                    marker: {
                        "reconstruction_status": version["statuses"][frame_index].get(marker),
                        "identity_status": version["identity_statuses"][frame_index].get(marker)
                        if version.get("identity_statuses")
                        else None,
                    }
                    for marker in sorted(frame)
                },
            }
            for frame_index, frame in enumerate(version["frames"])
        ],
    }


def export_review_bundle(
    store: PlatformStore,
    output_path: str | Path,
    action_name: str,
    pipeline_name: str,
    version: dict[str, Any] | None = None,
) -> Path:
    selected = version or store.latest_processing(action_name, pipeline_name)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            build_review_bundle(store, action_name, pipeline_name, selected),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def export_decision_csv(
    version: dict[str, Any], output_path: str | Path
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kind",
                "decision_id",
                "status",
                "start_frame",
                "end_frame",
                "source_person_id",
                "target_person_id",
                "marker",
                "origin_version",
            ],
        )
        writer.writeheader()
        for decision in version.get("identity_decisions", []):
            writer.writerow(
                {
                    "kind": "identity_decision",
                    "decision_id": decision["decision_id"],
                    "status": decision["status"],
                    "start_frame": decision["start_frame"],
                    "end_frame": decision["end_frame"],
                    "source_person_id": decision["person_id"],
                    "target_person_id": decision["alternative_person_id"],
                    "marker": decision["marker"],
                    "origin_version": version["version"],
                }
            )
        for item in version.get("track_reassignments", []):
            writer.writerow(
                {
                    "kind": "track_reassignment",
                    "decision_id": item["reassignment_id"],
                    "status": item["status"],
                    "start_frame": item["start_frame"],
                    "end_frame": item["end_frame"],
                    "source_person_id": item["source_person_id"],
                    "target_person_id": item["target_person_id"],
                    "marker": item["marker"],
                    "origin_version": version["version"],
                }
            )
    return path
