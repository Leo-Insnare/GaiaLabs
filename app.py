from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import RuntimeConfig
from pipeline.runner import process_video
from utils.io import read_bytes


st.set_page_config(page_title="Theia CCTV AI", layout="wide")


def main() -> None:
    st.title("Theia Clinic CCTV AI")
    st.caption("A/C/D 관측 변수 기반 CRM 출력변수 테스트")

    with st.sidebar:
        st.header("Input")
        video_file = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])

        st.subheader("Model")
        model_mode = st.radio("Model source", ["Path", "Upload"], horizontal=True)
        a_model_upload = None
        d_model_upload = None
        a_model_path = "models/a_group_yolov8n_best.pt"
        d_model_path = "models/best.pt"

        if model_mode == "Path":
            a_model_path = st.text_input("A model path", a_model_path)
            d_model_path = st.text_input("D model path", d_model_path)
        else:
            a_model_upload = st.file_uploader("A Group .pt", type=["pt"], key="a_pt")
            d_model_upload = st.file_uploader("D Group .pt", type=["pt"], key="d_pt")

        st.subheader("Sampling")
        sampling_fps = st.number_input("Sampling FPS", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        sampling_mode = st.selectbox("Sampling mode", ["seek", "sequential"], index=0)
        max_sampled_frames = st.number_input("Max sampled frames (0 = no limit)", min_value=0, value=0, step=50)

        st.subheader("Threshold")
        a_conf = st.slider("A confidence", 0.05, 0.95, 0.35, 0.01)
        d_conf = st.slider("D confidence", 0.05, 0.95, 0.35, 0.01)
        a2_black_uniform_conf = st.sidebar.slider(
            "A2 black uniform threshold",
            0.0,
            1.0,
            0.60,
            0.01,
        )
        c1_conf = st.sidebar.slider("C1 confidence threshold", 0.0, 1.0, 0.55, 0.01)
        c1_min_area_ratio = st.sidebar.slider("C1 min area ratio", 0.001, 0.20, 0.05, 0.001)
        st.subheader("Debounce")
        s1_start = st.number_input("S1 start", min_value=0.0, value=5.0, step=1.0)
        s1_end = st.number_input("S1 end", min_value=0.0, value=60.0, step=1.0)
        s2_start = st.number_input("S2 start", min_value=0.0, value=5.0, step=1.0)
        s2_end = st.number_input("S2 end", min_value=0.0, value=10.0, step=1.0)
        s3_start = st.number_input("S3 start", min_value=0.0, value=30.0, step=1.0)
        s3_end = st.number_input("S3 end", min_value=0.0, value=5.0, step=1.0)
        s5_start = st.number_input("S5 start", min_value=0.0, value=5.0, step=1.0)
        s5_end = st.number_input("S5 end", min_value=0.0, value=3.0, step=1.0)
        s6_start = st.number_input("S6 start", min_value=0.0, value=60.0, step=1.0)
        s6_end = st.number_input("S6 end", min_value=0.0, value=5.0, step=1.0)

        st.subheader("Output")
        save_preview = st.checkbox("Annotated preview video", value=True)
        convert_h264 = st.checkbox("H.264 conversion", value=True)
        run_button = st.button("Run analysis", type="primary", use_container_width=True)

    if not run_button:
        render_guide()
        return

    if video_file is None:
        st.warning("분석할 영상을 업로드해 주세요.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        video_path = temp_path / video_file.name
        video_path.write_bytes(video_file.getbuffer())

        config = build_config(
            a_model_path=a_model_path,
            d_model_path=d_model_path,
            sampling_fps=sampling_fps,
            sampling_mode=sampling_mode,
            max_sampled_frames=max_sampled_frames,
            a_conf=a_conf,
            a2_black_uniform_conf=a2_black_uniform_conf,
            d_conf=d_conf,
            c1_conf=c1_conf,
            c1_area=c1_min_area_ratio,
            s1_start=s1_start,
            s1_end=s1_end,
            s2_start=s2_start,
            s2_end=s2_end,
            s3_start=s3_start,
            s3_end=s3_end,
            s5_start=s5_start,
            s5_end=s5_end,
            s6_start=s6_start,
            s6_end=s6_end,
            save_preview=save_preview,
            convert_h264=convert_h264,
        )

        if model_mode == "Upload":
            if a_model_upload is None or d_model_upload is None:
                st.warning("A/D 모델 파일을 모두 업로드해 주세요.")
                return
            a_path = temp_path / a_model_upload.name
            d_path = temp_path / d_model_upload.name
            a_path.write_bytes(a_model_upload.getbuffer())
            d_path.write_bytes(d_model_upload.getbuffer())
            config.model_paths.a_model_path = str(a_path)
            config.model_paths.d_model_path = str(d_path)

        progress_bar = st.progress(0)
        progress_text = st.empty()

        def progress_callback(current: int, total: int | None, stage: str) -> None:
            if total and total > 0:
                progress_bar.progress(min(current / total, 1.0))
                progress_text.write(f"{stage}: {current}/{total}")
            else:
                progress_text.write(f"{stage}: {current}")

        try:
            result = process_video(
                video_path=video_path,
                output_base_dir=temp_path / "outputs",
                config=config,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            st.error(str(exc))
            return

        progress_bar.progress(1.0)
        progress_text.write("done")
        render_results(result)


def build_config(**kwargs: Any) -> RuntimeConfig:
    config = RuntimeConfig()
    config.model_paths.a_model_path = kwargs["a_model_path"]
    config.model_paths.d_model_path = kwargs["d_model_path"]
    config.sampling_fps = float(kwargs["sampling_fps"])
    config.sampling_mode = kwargs["sampling_mode"]
    config.max_sampled_frames = int(kwargs["max_sampled_frames"]) or None
    config.thresholds.a_conf = float(kwargs["a_conf"])
    config.thresholds.a2_black_uniform_conf = float(kwargs["a2_black_uniform_conf"])
    config.thresholds.d_conf = float(kwargs["d_conf"])
    config.thresholds.c1_conf = float(kwargs["c1_conf"])
    config.thresholds.c1_min_area_ratio = float(kwargs["c1_area"])
    config.debounce.s1_start_sec = float(kwargs["s1_start"])
    config.debounce.s1_end_sec = float(kwargs["s1_end"])
    config.debounce.s2_start_sec = float(kwargs["s2_start"])
    config.debounce.s2_end_sec = float(kwargs["s2_end"])
    config.debounce.s3_start_sec = float(kwargs["s3_start"])
    config.debounce.s3_end_sec = float(kwargs["s3_end"])
    config.debounce.s5_start_sec = float(kwargs["s5_start"])
    config.debounce.s5_end_sec = float(kwargs["s5_end"])
    config.debounce.s6_start_sec = float(kwargs["s6_start"])
    config.debounce.s6_end_sec = float(kwargs["s6_end"])
    config.save_preview_video = bool(kwargs["save_preview"])
    config.convert_preview_h264 = bool(kwargs["convert_h264"])
    config.b_group_enabled = False
    config.a3_enabled = False
    config.pose_enabled = False
    config.tracking_enabled = False
    return config


def render_guide() -> None:
    st.info("영상과 A/D 모델을 입력한 뒤 Run analysis를 실행하면 CSV, JSON, preview video가 생성됩니다.")
    st.markdown(
        """
        - A1: 시술면 위 환자 lying/sitting
        - A2: 시술면 밖 스탭/인력 whitecoat/gray scrub/black uniform
        - C1: A1 환자 bbox 기반 얼굴/상단 crop rule
        - D: D0~D14 의료기기 bbox
        - B Group, A3, pose, tracking은 이번 CRM 출력변수 계산에서 제외
        """
    )


def render_results(result: dict[str, Any]) -> None:
    output_paths = result["output_paths"]
    observations: pd.DataFrame = result["observations"]
    segments = result["segments"]
    summary = result["summary"]
    verification = result["verification"]

    st.success("Analysis completed")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames", summary.get("processed_frame_count", 0))
    c2.metric("S1 occupied sec", summary.get("room_occupied_total_sec", 0))
    c3.metric("S2 waiting sec", summary.get("simple_waiting_total_sec", 0))
    c4.metric("S3 anesthesia sec", summary.get("anesthesia_total_sec", 0))

    if verification.get("passed"):
        st.success("Verification passed")
    else:
        st.warning("Verification check required")

    tabs = st.tabs(["Annotated video", "Frame observations", "CRM timeline", "Summary JSON", "Download outputs"])

    with tabs[0]:
        preview = output_paths.get("preview_video")
        if preview and Path(preview).exists():
            st.video(read_bytes(preview))
        else:
            st.write("Preview video was not generated.")

    with tabs[1]:
        st.dataframe(observations, use_container_width=True, height=520)

    with tabs[2]:
        timeline_df = pd.DataFrame(segments)
        st.dataframe(timeline_df, use_container_width=True, height=520)

    with tabs[3]:
        st.json(summary)
        with st.expander("Verification"):
            st.json(verification)

    with tabs[4]:
        download_button("frame_observations.csv", output_paths.get("observations_csv"), "text/csv")
        download_button("event_timeline.json", output_paths.get("timeline_json"), "application/json")
        download_button("crm_summary.json", output_paths.get("summary_json"), "application/json")
        download_button("verification_report.json", output_paths.get("verification_json"), "application/json")
        download_button("outputs.zip", output_paths.get("zip"), "application/zip")


def download_button(label: str, path: str | None, mime: str) -> None:
    if not path or not Path(path).exists():
        return
    st.download_button(label=label, data=read_bytes(path), file_name=Path(path).name, mime=mime, use_container_width=True)


if __name__ == "__main__":
    main()
