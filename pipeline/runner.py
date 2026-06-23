from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from config import RuntimeConfig
from inference.a_detector import ADetector
from inference.c_detector import CDetector
from inference.d_detector import DDetector
from pipeline.integrator import build_observation_row, empty_a_result, empty_d_result, output_schema
from pipeline.state_machine import CRMStateMachine
from pipeline.verification import verify_run
from pipeline.video_reader import get_video_metadata, iter_sampled_frames
from pipeline.visualizer import PreviewVideoWriter, annotate_frame, convert_to_h264
from utils.io import ensure_dir, make_output_dir, save_csv, save_json, zip_outputs

ProgressCallback = Callable[[int, int | None, str], None]

FACE_COVERING_DEVICE_CLASSES = {
    "D13_oxygen_dome",
    "D14_led_mask",
}


def process_video(
    video_path: str | Path,
    output_base_dir: str | Path,
    config: RuntimeConfig,
    run_name: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    video_path = Path(video_path)
    output_dir = make_output_dir(output_base_dir, run_name or video_path.stem)
    ensure_dir(output_dir)

    meta = get_video_metadata(video_path)
    estimated_total = int(meta.duration_sec * config.sampling_fps) + 1 if meta.duration_sec > 0 else None

    _validate_inputs(config, video_path)

    a_detector = ADetector(config) if config.a_group_enabled else None
    c_detector = CDetector(config) if config.c_group_enabled else None
    d_detector = DDetector(config) if config.d_group_enabled else None

    preview_writer: PreviewVideoWriter | None = None
    preview_path = output_dir / "annotated_preview.mp4"
    if config.save_preview_video:
        preview_writer = PreviewVideoWriter(preview_path, fps=config.preview_fps)

    observations: list[dict[str, Any]] = []
    processed_count = 0

    try:
        for sample in iter_sampled_frames(
            video_path,
            sampling_fps=config.sampling_fps,
            mode=config.sampling_mode,
            max_sampled_frames=config.max_sampled_frames,
        ):
            if progress_callback:
                progress_callback(processed_count, estimated_total, "sampling/inference")

            if a_detector is not None:
                a_detections = a_detector.predict(sample.frame_bgr)
                a_result = a_detector.logical(a_detections)
            else:
                a_result = empty_a_result()

            if d_detector is not None:
                d_detections = d_detector.predict(sample.frame_bgr)
                d_result = d_detector.logical(d_detections)
            else:
                d_result = empty_d_result()

            if c_detector is not None and a_result.get("a1_patient_bbox") is not None:
                c_result = c_detector.predict_from_patient_bbox(
                    sample.frame_bgr,
                    a_result.get("a1_patient_bbox"),
                )
                c_result = apply_c1_negative_guards(c_result, d_result)
            else:
                c_result = CDetector._empty("disabled_or_no_patient")

            row = build_observation_row(
                frame_idx=sample.frame_idx,
                source_frame_idx=sample.source_frame_idx,
                timestamp_sec=sample.timestamp_sec,
                a_result=a_result,
                c_result=c_result,
                d_result=d_result,
                config=config,
            )
            observations.append(row)

            if preview_writer is not None:
                annotated = annotate_frame(
                    sample.frame_bgr,
                    a_result.get("a_detections", []),
                    d_result.get("d_detections", []),
                    c_result,
                    row,
                )
                preview_writer.write(annotated)

            processed_count += 1
    finally:
        if preview_writer is not None:
            preview_writer.release()

    if progress_callback:
        progress_callback(processed_count, estimated_total, "state_machine")

    observations_df = pd.DataFrame(observations)
    state_machine = CRMStateMachine(config)
    segments = state_machine.build_segments(observations_df)
    video_meta_dict = meta.__dict__

    model_meta = {
        "a_model_path": str(config.resolved_a_model_path()) if config.a_group_enabled else None,
        "d_model_path": str(config.resolved_d_model_path()) if config.d_group_enabled else None,
        "c_group_mode": "a_patient_face_roi_rule" if config.c_group_enabled else "disabled",
        "b_group_enabled": config.b_group_enabled,
        "a3_enabled": config.a3_enabled,
    }

    summary = state_machine.build_summary(observations_df, segments, video_meta_dict, model_meta)

    csv_path = save_csv(observations_df, output_dir / "frame_observations.csv")
    timeline_path = save_json({"events": segments}, output_dir / "event_timeline.json")
    summary_path = save_json(summary, output_dir / "crm_summary.json")
    schema_path = save_json(output_schema(), output_dir / "output_schema.json")
    config_path = save_json(config.to_dict(), output_dir / "run_config.json")

    final_preview_path = preview_path
    if config.save_preview_video and preview_path.exists() and config.convert_preview_h264:
        final_preview_path = convert_to_h264(preview_path)

    output_paths = {
        "output_dir": str(output_dir),
        "observations_csv": str(csv_path),
        "timeline_json": str(timeline_path),
        "summary_json": str(summary_path),
        "schema_json": str(schema_path),
        "config_json": str(config_path),
        "preview_video": str(final_preview_path) if config.save_preview_video and final_preview_path.exists() else "",
    }

    verification = verify_run(config, video_path, observations_df, segments, output_paths)
    verification_path = save_json(verification, output_dir / "verification_report.json")
    output_paths["verification_json"] = str(verification_path)

    zip_path = zip_outputs(output_dir)
    output_paths["zip"] = str(zip_path)

    if progress_callback:
        progress_callback(processed_count, estimated_total, "done")

    return {
        "observations": observations_df,
        "segments": segments,
        "summary": summary,
        "verification": verification,
        "output_paths": output_paths,
        "video_metadata": video_meta_dict,
    }


def apply_c1_negative_guards(c_result: dict[str, Any], d_result: dict[str, Any]) -> dict[str, Any]:
    classes_present = set(d_result.get("d_classes_present", []))
    blocking_devices = classes_present.intersection(FACE_COVERING_DEVICE_CLASSES)

    if not blocking_devices:
        return c_result

    c_result = dict(c_result)
    c_result["c1_raw"] = False
    c_result["c1_conf"] = 0.0
    c_result["c1_bbox"] = None
    c_result["c1_area_ratio"] = 0.0
    c_result["c1_largest_component_ratio"] = 0.0
    c_result["c1_reason"] = "suppressed_by_face_covering_device"
    c_result["c1_suppressed_by"] = sorted(blocking_devices)

    debug = dict(c_result.get("c1_debug", {}))
    debug["suppressed_by_face_covering_device"] = True
    debug["blocking_devices"] = sorted(blocking_devices)
    c_result["c1_debug"] = debug

    return c_result


def _validate_inputs(config: RuntimeConfig, video_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"video_not_found: {video_path}")
    if config.a_group_enabled and not config.resolved_a_model_path().exists():
        raise FileNotFoundError(f"a_model_not_found: {config.resolved_a_model_path()}")
    if config.d_group_enabled and not config.resolved_d_model_path().exists():
        raise FileNotFoundError(f"d_model_not_found: {config.resolved_d_model_path()}")