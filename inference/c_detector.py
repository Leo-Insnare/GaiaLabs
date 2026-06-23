from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from config import RuntimeConfig


class CDetector:
    def __init__(self, config: RuntimeConfig):
        self.config = config

    def predict_from_patient_bbox(self, frame_bgr: np.ndarray, patient_bbox: list[float] | None) -> dict[str, Any]:
        if patient_bbox is None:
            return self._empty("no_patient_bbox")

        frame_h, frame_w = frame_bgr.shape[:2]
        px1, py1, px2, py2 = self._clip_bbox(patient_bbox, frame_w, frame_h)

        if px2 <= px1 or py2 <= py1:
            return self._empty("invalid_patient_bbox")

        patient_crop = frame_bgr[py1:py2, px1:px2]

        if patient_crop.size == 0:
            return self._empty("empty_patient_crop")

        face_roi = self._find_face_roi(patient_crop, px1, py1, frame_w, frame_h)

        if face_roi is None:
            return self._empty("no_face_roi")

        fx1, fy1, fx2, fy2, face_meta = face_roi
        face_crop = frame_bgr[fy1:fy2, fx1:fx2]

        if face_crop.size == 0:
            return self._empty("empty_face_crop")

        score = self._score_anesthesia(face_crop)

        min_ratio = max(float(self.config.thresholds.c1_min_area_ratio), 0.035)
        min_component = max(min_ratio * 0.35, 0.010)

        raw = (
            score["white_ratio"] >= min_ratio
            and score["largest_component_ratio"] >= min_component
            and score["conf"] >= float(self.config.thresholds.c1_conf)
            and not score["uniform_cover_like"]
        )

        final_conf = float(score["conf"]) if raw else 0.0

        if raw:
            reason = "ok"
        elif score["uniform_cover_like"]:
            reason = "uniform_cover_like"
        else:
            reason = "face_found_no_anesthesia"

        return {
            "c1_raw": bool(raw),
            "c1_conf": final_conf,
            "c1_source": "a_patient_face_roi_rule",
            "c1_bbox": [float(fx1), float(fy1), float(fx2), float(fy2)] if raw else None,
            "c1_face_bbox": [float(fx1), float(fy1), float(fx2), float(fy2)],
            "c1_area_ratio": float(score["white_ratio"]) if raw else 0.0,
            "c1_largest_component_ratio": float(score["largest_component_ratio"]) if raw else 0.0,
            "c1_reason": reason,
            "c1_debug": {
                "face_score": float(face_meta["score"]),
                "face_area_ratio": float(face_meta["area_ratio"]),
                "raw_rule_score": float(score["conf"]),
                "white_ratio": float(score["white_ratio"]),
                "largest_component_ratio": float(score["largest_component_ratio"]),
                "largest_fill_ratio": float(score["largest_fill_ratio"]),
                "largest_bbox_area_ratio": float(score["largest_bbox_area_ratio"]),
                "skin_ratio_in_face": float(score["skin_ratio"]),
                "uniform_cover_like": bool(score["uniform_cover_like"]),
            },
        }

    def _find_face_roi(
        self,
        patient_crop: np.ndarray,
        offset_x: int,
        offset_y: int,
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int, int, int, dict[str, float]] | None:
        crop_h, crop_w = patient_crop.shape[:2]

        if crop_h < 20 or crop_w < 20:
            return None

        skin_mask = self._skin_mask(patient_crop)
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crop_area = float(crop_h * crop_w)
        candidates: list[tuple[float, int, int, int, int, float, float]] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < max(80.0, crop_area * 0.001):
                continue
            if area > crop_area * 0.18:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            if w <= 4 or h <= 4:
                continue

            aspect = float(w / max(h, 1))
            if aspect < 0.35 or aspect > 2.3:
                continue

            rect_area = float(w * h)
            extent = area / max(rect_area, 1.0)

            if extent < 0.18:
                continue

            cx = (x + w * 0.5) / crop_w
            cy = (y + h * 0.5) / crop_h
            edge_dist = min(cx, 1.0 - cx, cy, 1.0 - cy)

            area_score = min(area / max(crop_area * 0.025, 1.0), 1.0)
            aspect_score = 1.0 - min(abs(np.log(max(aspect, 1e-6))) / np.log(2.5), 1.0)
            edge_score = 1.0 - min(edge_dist / 0.45, 1.0)
            extent_score = min(extent / 0.75, 1.0)

            score = area_score * 0.50 + aspect_score * 0.20 + edge_score * 0.20 + extent_score * 0.10
            candidates.append((score, x, y, w, h, area / crop_area, extent))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        score, x, y, w, h, area_ratio, extent = candidates[0]

        pad_x = int(w * 0.65)
        pad_y_top = int(h * 0.65)
        pad_y_bottom = int(h * 0.75)

        rx1 = offset_x + x - pad_x
        ry1 = offset_y + y - pad_y_top
        rx2 = offset_x + x + w + pad_x
        ry2 = offset_y + y + h + pad_y_bottom

        rx1, ry1, rx2, ry2 = self._clip_bbox([rx1, ry1, rx2, ry2], frame_w, frame_h)

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        return (
            rx1,
            ry1,
            rx2,
            ry2,
            {
                "score": float(score),
                "area_ratio": float(area_ratio),
                "extent": float(extent),
            },
        )

    def _score_anesthesia(self, face_crop: np.ndarray) -> dict[str, float]:
        hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)

        b = face_crop[:, :, 0].astype(np.int16)
        g = face_crop[:, :, 1].astype(np.int16)
        r = face_crop[:, :, 2].astype(np.int16)

        max_ch = np.maximum(np.maximum(b, g), r)
        min_ch = np.minimum(np.minimum(b, g), r)
        channel_gap = max_ch - min_ch

        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        skin_mask = self._skin_mask(face_crop).astype(bool)

        neutral = channel_gap < 38
        bright = min_ch > 135
        low_sat = sat < 48
        very_low_sat = sat < 32
        high_val = val > 165
        mid_val = val > 145

        white_mask = ((neutral & bright & low_sat & high_val) | (neutral & very_low_sat & mid_val))
        white_mask = white_mask & (~skin_mask)

        kernel = np.ones((3, 3), np.uint8)
        white_uint8 = white_mask.astype(np.uint8) * 255
        white_uint8 = cv2.morphologyEx(white_uint8, cv2.MORPH_OPEN, kernel)
        white_uint8 = cv2.morphologyEx(white_uint8, cv2.MORPH_CLOSE, kernel)

        white_bool = white_uint8 > 0
        white_ratio = float(white_bool.mean())
        skin_ratio = float(skin_mask.mean())

        contours, _ = cv2.findContours(white_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        face_area = float(face_crop.shape[0] * face_crop.shape[1])

        largest_component_ratio = 0.0
        largest_fill_ratio = 0.0
        largest_bbox_area_ratio = 0.0

        if contours:
            largest = max(contours, key=cv2.contourArea)
            largest_area = float(cv2.contourArea(largest))
            x, y, w, h = cv2.boundingRect(largest)
            bbox_area = float(max(w * h, 1))

            largest_component_ratio = largest_area / max(face_area, 1.0)
            largest_fill_ratio = largest_area / bbox_area
            largest_bbox_area_ratio = bbox_area / max(face_area, 1.0)

        uniform_cover_like = (
            white_ratio >= 0.18
            and largest_component_ratio >= 0.10
            and largest_fill_ratio >= 0.62
            and largest_bbox_area_ratio >= 0.18
        )

        min_ratio = max(float(self.config.thresholds.c1_min_area_ratio), 0.035)
        min_component = max(min_ratio * 0.35, 0.010)

        ratio_score = min(white_ratio / max(min_ratio, 1e-6), 1.0)
        component_score = min(largest_component_ratio / max(min_component, 1e-6), 1.0)
        conf = ratio_score * 0.65 + component_score * 0.35

        if skin_ratio < 0.015 and white_ratio < 0.10:
            conf *= 0.5

        if uniform_cover_like:
            conf *= 0.25

        return {
            "conf": float(min(conf, 1.0)),
            "white_ratio": float(white_ratio),
            "largest_component_ratio": float(largest_component_ratio),
            "largest_fill_ratio": float(largest_fill_ratio),
            "largest_bbox_area_ratio": float(largest_bbox_area_ratio),
            "skin_ratio": float(skin_ratio),
            "uniform_cover_like": bool(uniform_cover_like),
        }

    @staticmethod
    def _skin_mask(bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        y = ycrcb[:, :, 0]
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        mask_ycrcb = (
            (y > 45)
            & (cr > 132)
            & (cr < 178)
            & (cb > 72)
            & (cb < 135)
        )

        mask_hsv = (
            (((h < 25) | (h > 160)))
            & (s > 18)
            & (s < 185)
            & (v > 55)
        )

        mask = mask_ycrcb & mask_hsv
        return mask.astype(np.uint8) * 255

    @staticmethod
    def _clip_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(int(round(x1)), width - 1))
        y1 = max(0, min(int(round(y1)), height - 1))
        x2 = max(0, min(int(round(x2)), width))
        y2 = max(0, min(int(round(y2)), height))

        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)

        return x1, y1, x2, y2

    @staticmethod
    def _empty(reason: str) -> dict[str, Any]:
        return {
            "c1_raw": False,
            "c1_conf": 0.0,
            "c1_source": "a_patient_face_roi_rule",
            "c1_bbox": None,
            "c1_face_bbox": None,
            "c1_area_ratio": 0.0,
            "c1_largest_component_ratio": 0.0,
            "c1_reason": reason,
            "c1_debug": {
                "face_score": 0.0,
                "face_area_ratio": 0.0,
                "white_ratio": 0.0,
                "largest_component_ratio": 0.0,
                "skin_ratio_in_face": 0.0,
            },
        }