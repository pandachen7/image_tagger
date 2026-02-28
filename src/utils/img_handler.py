import time
from typing import Optional

import cv2
import numpy as np

from src.utils.dynamic_settings import settings
from src.utils.loglo import getUniqueLogger
from src.utils.model import Bbox, ModelType, Polygon

log = getUniqueLogger(__file__)


def mask_to_polygon(contours, tolerance=0.01):
    """
    tolerance: 越小越精密, 越大越粗糙
    - 0.001 ~ 0.005: 精密
    - 0.01 ~ 0.02: 中等
    - 0.05 ~ 0.1: 粗糙
    """
    polygons = []
    for cnt in contours:
        epsilon = tolerance * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) >= 3:
            polygons.append(approx.squeeze().tolist())
    return polygons


class Inferencer:
    """Manages model instances and runs inference."""

    def __init__(self):
        self.active_model_type: str = ModelType.NONE
        self.model_path: Optional[str] = None
        self.sam_model_path: Optional[str] = None
        self._yolo_model = None
        self._sam_predictor = None

    def set_active_model(self, model_type: str, model_path: str = None):
        self.active_model_type = model_type
        if model_path:
            if model_type == ModelType.YOLO:
                self.model_path = model_path
            elif model_type == ModelType.SAM3:
                self.sam_model_path = model_path

    def ensure_loaded(self, model_type: str = None) -> bool:
        """Lazy-load the given model type. Returns True if ready."""
        if model_type is None:
            model_type = self.active_model_type
        if model_type == ModelType.YOLO:
            if self._yolo_model is None and self.model_path:
                from ultralytics import YOLO

                self._yolo_model = YOLO(self.model_path)
            return self._yolo_model is not None
        elif model_type == ModelType.SAM3:
            if self._sam_predictor is None and self.sam_model_path:
                from ultralytics.models.sam import SAM3SemanticPredictor

                overrides = dict(
                    conf=0.25,
                    imgsz=630,  # 設愈高, VRAM容易不夠, 建議14倍數的630
                    task="segment",
                    mode="predict",
                    model=self.sam_model_path,
                    half=True,
                    verbose=False,
                )
                self._sam_predictor = SAM3SemanticPredictor(overrides=overrides)
            return self._sam_predictor is not None
        return False

    def is_loaded(self, model_type: str) -> bool:
        if model_type == ModelType.YOLO:
            return self._yolo_model is not None
        elif model_type == ModelType.SAM3:
            return self._sam_predictor is not None
        return False

    def infer_yolo(self, cv_img) -> list:
        """YOLO inference. Returns list of Bbox."""
        results = self._yolo_model.predict(cv_img, verbose=False)
        bboxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    b = box.xyxy[0]
                    label = self._yolo_model.names[int(box.cls)]
                    bboxes.append(
                        Bbox(
                            int(b[0]),
                            int(b[1]),
                            int(b[2] - b[0]),
                            int(b[3] - b[1]),
                            label,
                            float(box.conf),
                        )
                    )
        return bboxes

    def infer_sam3(self, image_path, src_shape) -> tuple[list, list]:
        """SAM3 inference. Returns (list of Bbox, list of Polygon)."""
        self._sam_predictor.set_image(image_path)
        labels = list(set(settings.text_prompts or []))
        bboxes, polygons = [], []
        t1 = time.time()
        masks, boxes = self._sam_predictor.inference_features(
            self._sam_predictor.features, src_shape=src_shape, text=labels
        )
        if masks is not None:
            masks_np = masks.cpu().numpy()
            for i, mask in enumerate(masks_np):
                label = labels[i] if i < len(labels) else labels[-1]
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    tolerance = settings.polygon_tolerance or 0.002
                    for poly_pts in mask_to_polygon(contours, tolerance):
                        points = [(float(x), float(y)) for x, y in poly_pts]
                        polygons.append(Polygon(points, label, -1.0))
        if boxes is not None:
            boxes_np = boxes.cpu().numpy()
            for i, box in enumerate(boxes_np):
                label = labels[i] if i < len(labels) else labels[-1]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    bboxes.append(Bbox(x1, y1, x2 - x1, y2 - y1, label, -1.0))
        log.d(f"SAM3 inference time: {time.time() - t1}")
        return bboxes, polygons


inferencer = Inferencer()
