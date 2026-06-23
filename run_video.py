from __future__ import annotations

import argparse
from pathlib import Path

from config import RuntimeConfig
from pipeline.runner import process_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--a-model", default=None)
    parser.add_argument("--d-model", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--sampling-fps", type=float, default=1.0)
    parser.add_argument("--sampling-mode", default="seek", choices=["seek", "sequential"])
    parser.add_argument("--a-conf", type=float, default=0.35)
    parser.add_argument("--d-conf", type=float, default=0.35)
    parser.add_argument("--c1-conf", type=float, default=0.35)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-sampled-frames", type=int, default=None)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-h264", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RuntimeConfig()
    if args.a_model:
        config.model_paths.a_model_path = args.a_model
    if args.d_model:
        config.model_paths.d_model_path = args.d_model
    config.sampling_fps = args.sampling_fps
    config.sampling_mode = args.sampling_mode
    config.thresholds.a_conf = args.a_conf
    config.thresholds.d_conf = args.d_conf
    config.thresholds.c1_conf = args.c1_conf
    config.device = args.device
    config.max_sampled_frames = args.max_sampled_frames
    config.save_preview_video = not args.no_preview
    config.convert_preview_h264 = not args.no_h264

    def progress(current: int, total: int | None, stage: str) -> None:
        total_text = str(total) if total is not None else "?"
        print(f"[{stage}] {current}/{total_text}")

    result = process_video(
        video_path=Path(args.video),
        output_base_dir=Path(args.output_dir),
        config=config,
        progress_callback=progress,
    )
    print("output_dir:", result["output_paths"]["output_dir"])
    print("zip:", result["output_paths"]["zip"])
    print("verification_passed:", result["verification"]["passed"])


if __name__ == "__main__":
    main()
