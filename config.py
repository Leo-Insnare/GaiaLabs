from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


A_CLASS_MAP = {
    0: "a1_lying_on_surface",
    1: "a1_sitting_on_surface",
    2: "a2_medical_whitecoat",
    3: "a2_medical_gray_scrub",
    4: "a2_coordinator_black_uniform",
}

A1_CLASSES = {
    "a1_lying_on_surface",
    "a1_sitting_on_surface",
}

A2_CLASSES = {
    "a2_medical_whitecoat",
    "a2_medical_gray_scrub",
    "a2_coordinator_black_uniform",
}

D_CLASS_MAP = {
    0: "D0_erbium",
    1: "D1_titanium",
    2: "D2_thermage",
    3: "D3_potenza",
    4: "D4_inmode",
    5: "D5_density",
    6: "D6_dermashine",
    7: "D7_onda",
    8: "D8_shurink",
    9: "D9_ulthera",
    10: "D10_sofwave",
    11: "D11_ldm",
    12: "D12_bbl",
    13: "D13_oxygen_dome",
    14: "D14_led_mask",
}

CRM_STATUS_MAP = {
    "room_occupied": "S1_room_occupied",
    "simple_waiting": "S2_simple_waiting",
    "anesthesia": "S3_anesthesia",
    "treatment_or_procedure_candidate": "S4_treatment_or_procedure_candidate",
    "device_present": "S5_device_present",
    "room_empty": "S6_room_empty",
}

DEFAULT_A_MODEL_PATH = os.getenv(
    "THEIA_A_MODEL_PATH",
    "/content/drive/MyDrive/GaiaLabs/A_group_model/a_group_yolov8n_best.pt",
)
DEFAULT_D_MODEL_PATH = os.getenv(
    "THEIA_D_MODEL_PATH",
    "/content/drive/MyDrive/GaiaLabs/D_group_model/best.pt",
)


@dataclass
class ModelPaths:
    a_model_path: str = DEFAULT_A_MODEL_PATH
    d_model_path: str = DEFAULT_D_MODEL_PATH


@dataclass
class ThresholdConfig:
    a_conf: float = 0.30
    a1_conf: float = 0.35
    a2_whitecoat_conf: float = 0.35
    a2_gray_scrub_conf: float = 0.35
    a2_black_uniform_conf: float = 0.60
    d_conf: float = 0.35
    yolo_iou: float = 0.45
    c1_conf: float = 0.55
    c1_min_area_ratio: float = 0.05

@dataclass
class DebounceConfig:
    s1_start_sec: float = 5.0
    s1_end_sec: float = 60.0
    s2_start_sec: float = 5.0
    s2_end_sec: float = 10.0
    s3_start_sec: float = 30.0
    s3_end_sec: float = 5.0
    s5_start_sec: float = 5.0
    s5_end_sec: float = 3.0
    s6_start_sec: float = 60.0
    s6_end_sec: float = 5.0


@dataclass
class RuntimeConfig:
    model_paths: ModelPaths = field(default_factory=ModelPaths)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    debounce: DebounceConfig = field(default_factory=DebounceConfig)
    sampling_fps: float = 1.0
    sampling_mode: str = "seek"
    device: str | None = None
    a_group_enabled: bool = True
    c_group_enabled: bool = True
    d_group_enabled: bool = True
    b_group_enabled: bool = False
    a3_enabled: bool = False
    pose_enabled: bool = False
    tracking_enabled: bool = False
    s4_candidate_enabled: bool = True
    save_preview_video: bool = True
    preview_fps: float = 4.0
    convert_preview_h264: bool = True
    max_sampled_frames: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_a_model_path(self) -> Path:
        return Path(self.model_paths.a_model_path).expanduser()

    def resolved_d_model_path(self) -> Path:
        return Path(self.model_paths.d_model_path).expanduser()