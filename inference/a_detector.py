from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics import YOLO

from config import A1_CLASSES, A2_CLASSES, A_CLASS_MAP, RuntimeConfig


@dataclass
class Detection:
    class_id: int
    class_name: str
    conf: float
    bbox_xyxy: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "conf": self.conf,
            "bbox_xyxy": self.bbox_xyxy,
        }


class ADetector:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.model = YOLO(str(config.resolved_a_model_path()))

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=frame_bgr,
            conf=self.config.thresholds.a_conf,
            iou=self.config.thresholds.yolo_iou,
            verbose=False,
            device=self.config.device,
        )[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = A_CLASS_MAP.get(class_id, str(class_id))

            if class_name not in A1_CLASSES and class_name not in A2_CLASSES:
                continue

            conf = float(box.conf.item())

            if conf < self._class_threshold(class_name):
                continue

            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(Detection(class_id, class_name, conf, xyxy))
        return detections
        
    def _class_threshold(self, class_name: str) -> float:
        t = self.config.thresholds

        if class_name in A1_CLASSES:
            return float(t.a1_conf)

        if class_name == "a2_medical_whitecoat":
            return float(t.a2_whitecoat_conf)

        if class_name == "a2_medical_gray_scrub":
            return float(t.a2_gray_scrub_conf)

        if class_name == "a2_coordinator_black_uniform":
            return float(t.a2_black_uniform_conf)

        return float(t.a_conf)
    @staticmethod
    def logical(detections: list[Detection]) -> dict[str, Any]:
        patient = [d for d in detections if d.class_name in A1_CLASSES]
        staff = [d for d in detections if d.class_name in A2_CLASSES]
        best_patient = max(patient, key=lambda d: d.conf, default=None)
        best_staff = max(staff, key=lambda d: d.conf, default=None)
        return {
            "a1_raw": best_patient is not None,
            "a1_conf": float(best_patient.conf) if best_patient else 0.0,
            "a1_source_class": best_patient.class_name if best_patient else "",
            "a1_patient_bbox": best_patient.bbox_xyxy if best_patient else None,
            "a2_raw": best_staff is not None,
            "a2_conf": float(best_staff.conf) if best_staff else 0.0,
            "a2_source_class": best_staff.class_name if best_staff else "",
            "a_detections": [d.to_dict() for d in detections],
        }