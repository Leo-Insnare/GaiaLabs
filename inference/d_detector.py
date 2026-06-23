from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics import YOLO

from config import D_CLASS_MAP, RuntimeConfig


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


class DDetector:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.model = YOLO(str(config.resolved_d_model_path()))

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            source=frame_bgr,
            conf=self.config.thresholds.d_conf,
            iou=self.config.thresholds.yolo_iou,
            verbose=False,
            device=self.config.device,
        )[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = D_CLASS_MAP.get(class_id, str(class_id))
            conf = float(box.conf.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(Detection(class_id, class_name, conf, xyxy))
        return detections

    @staticmethod
    def logical(detections: list[Detection]) -> dict[str, Any]:
        best_by_class: dict[str, Detection] = {}
        for det in detections:
            prev = best_by_class.get(det.class_name)
            if prev is None or det.conf > prev.conf:
                best_by_class[det.class_name] = det
        classes_present = sorted(best_by_class.keys())
        conf_by_class = {name: float(det.conf) for name, det in best_by_class.items()}
        return {
            "d_present_any": len(classes_present) > 0,
            "d_classes_present": classes_present,
            "d_conf_by_class": conf_by_class,
            "d_detections": [d.to_dict() for d in detections],
        }