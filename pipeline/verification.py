from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import D_CLASS_MAP, RuntimeConfig
from pipeline.integrator import OBSERVATION_COLUMNS
from pipeline.video_reader import get_video_metadata


def verify_run(
    config: RuntimeConfig,
    video_path: str | Path,
    observations: pd.DataFrame,
    segments: list[dict[str, Any]],
    output_paths: dict[str, str],
) -> dict[str, Any]:
    checks = []

    checks.append(_check("a_model_exists", (not config.a_group_enabled) or config.resolved_a_model_path().exists(), str(config.resolved_a_model_path())))
    checks.append(_check("d_model_exists", (not config.d_group_enabled) or config.resolved_d_model_path().exists(), str(config.resolved_d_model_path())))

    try:
        meta = get_video_metadata(video_path)
        video_ok = meta.frame_count > 0 or meta.duration_sec > 0
        checks.append(_check("video_open", video_ok, str(video_path)))
    except Exception as exc:
        checks.append(_check("video_open", False, str(exc)))

    checks.append(_check("sampled_frame_count", len(observations) > 0, str(len(observations))))
    checks.append(_check("a_inference_columns", {"a1_raw", "a2_raw", "a1_conf", "a2_conf"}.issubset(observations.columns), ""))
    checks.append(_check("d_inference_columns", {"d_present_any", "d_classes_present", "d_conf_by_class"}.issubset(observations.columns), ""))
    checks.append(_check("c1_result_column", "c1_raw" in observations.columns and "c1_conf" in observations.columns, ""))
    checks.append(_check("required_observation_columns", set(OBSERVATION_COLUMNS).issubset(observations.columns), ""))
    checks.append(_check("logical_variables_created", _has_logical_columns(observations), ""))
    checks.append(_check("crm_event_segment_generated", isinstance(segments, list), str(len(segments))))

    for key in ["observations_csv", "timeline_json", "summary_json"]:
        checks.append(_check(f"output_created:{key}", Path(output_paths.get(key, "")).exists(), output_paths.get(key, "")))
    if config.save_preview_video:
        checks.append(_check("output_created:preview_video", Path(output_paths.get("preview_video", "")).exists(), output_paths.get("preview_video", "")))

    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "d_class_count": len(D_CLASS_MAP),
    }


def _has_logical_columns(df: pd.DataFrame) -> bool:
    required = {"a1_raw", "a2_raw", "c1_raw", "d_present_any", "room_occupied_candidate", "simple_waiting_candidate"}
    return required.issubset(df.columns)


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}
