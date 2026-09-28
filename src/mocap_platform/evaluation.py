from __future__ import annotations

import math
from typing import Any

from .geometry import distance


def evaluate(processing: dict[str, Any], ground_truth: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[float] = []
    swap_errors: list[float] = []
    observed_count = recovered_count = uncertain_count = 0
    uncertain_truth = 0

    for frame_index, (frame, truth) in enumerate(zip(processing["frames"], ground_truth)):
        for marker, reconstructed in frame.items():
            status = processing["statuses"][frame_index][marker]
            actual = truth.get(marker)
            if actual is None:
                continue
            if status == "observed":
                observed_count += 1
            elif status == "recovered":
                recovered_count += 1
                errors.append(distance(reconstructed, actual))
            elif status == "uncertain":
                uncertain_count += 1
                uncertain_truth += 1
            if status == "observed" and reconstructed is not None:
                error = distance(reconstructed, actual)
                if error > 0.02:
                    swap_errors.append(error)

    rmse = math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None
    return {
        "version": processing["version"],
        "observed_marker_frames": observed_count,
        "recovered_marker_frames": recovered_count,
        "uncertain_marker_frames": uncertain_count,
        "recovered_rmse": rmse,
        "recovered_max_error": max(errors, default=None),
        "identity_mismatch_count": len(swap_errors),
        "identity_mismatch_rmse": math.sqrt(
            sum(error * error for error in swap_errors) / len(swap_errors)
        )
        if swap_errors
        else 0.0,
        "uncertain_had_ground_truth_frames": uncertain_truth,
        "no_fabricated_long_gap": uncertain_truth >= 10,
    }
