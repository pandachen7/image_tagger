# 管理 YOLO/SAM3 模型推論, 包含 mask 轉 polygon 功能
# updated: 2026-08-20
import time
from typing import Optional

import cv2
import numpy as np

from src.utils.dynamic_settings import settings
from src.utils.logger import getUniqueLogger
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


def sam3_label_conf(boxes_np, idx: int, labels: list[str]) -> tuple[str, float]:
    """
    由 SAM3 的 pred_boxes 取出第 idx 個偵測的 (label, confidence)。
    boxes_np: (N, 6) = xyxy + score + cls, 其中 cls 為 text prompt 的索引。
    """
    fallback = labels[-1] if labels else "object"
    if boxes_np is None or idx >= len(boxes_np):
        return fallback, -1.0
    cls_idx = int(boxes_np[idx][5])
    label = labels[cls_idx] if 0 <= cls_idx < len(labels) else fallback
    return label, float(boxes_np[idx][4])


class Inferencer:
    """Manages model instances and runs inference."""

    def __init__(self):
        self.active_model_type: str = ModelType.NONE
        self.model_path: Optional[str] = None
        self.sam_model_path: Optional[str] = None
        self._yolo_model = None
        self._sam_predictor = None
        self._loading = False

    @property
    def is_loading(self) -> bool:
        return self._loading

    def set_active_model(self, model_type: str, model_path: str = None):
        """設定啟用的模型類型與路徑，路徑變更時清除已載入的舊模型以便重新載入"""
        self.active_model_type = model_type
        if model_path:
            if model_type == ModelType.YOLO:
                if model_path != self.model_path:
                    self._yolo_model = None
                self.model_path = model_path
            elif model_type == ModelType.SAM3:
                if model_path != self.sam_model_path:
                    self._sam_predictor = None
                self.sam_model_path = model_path

    def ensure_loaded(self, model_type: str = None) -> bool:
        """Lazy-load the given model type. Returns True if ready."""
        if self._loading:
            return False
        if model_type is None:
            model_type = self.active_model_type
        self._loading = True
        try:
            if model_type == ModelType.YOLO:
                if self._yolo_model is None and self.model_path:
                    from ultralytics import YOLO

                    self._yolo_model = YOLO(self.model_path)
                return self._yolo_model is not None
            elif model_type == ModelType.SAM3:
                if self._sam_predictor is None and self.sam_model_path:
                    from ultralytics.models.sam import SAM3SemanticPredictor

                    overrides = dict(
                        conf=settings.models.sam3_conf or 0.25,
                        imgsz=630,  # 設愈高, VRAM容易不夠, 建議14倍數的630
                        task="segment",
                        mode="predict",
                        model=self.sam_model_path,
                        quantize=16,  # 16 = FP16, 取代已 deprecated 的 half=True
                        verbose=False,
                    )
                    self._sam_predictor = SAM3SemanticPredictor(overrides=overrides)
                return self._sam_predictor is not None
            return False
        finally:
            self._loading = False

    def is_loaded(self, model_type: str) -> bool:
        if model_type == ModelType.YOLO:
            return self._yolo_model is not None
        elif model_type == ModelType.SAM3:
            return self._sam_predictor is not None
        return False

    def infer_yolo(self, cv_img) -> tuple[list[Bbox], list[Polygon]]:
        """YOLO inference. 依 model task 與 yolo_label_mode 回傳 bbox / polygon / all。"""
        conf = settings.models.yolo_conf or 0.25
        results = self._yolo_model.predict(cv_img, conf=conf, verbose=False)
        is_seg = self._yolo_model.task == "segment"
        bboxes, polygons = [], []

        for result in results:
            # Bbox
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
            # Polygon（僅 segment model，masks.xy 已轉換至原圖座標）
            if is_seg and result.masks is not None:
                tolerance = settings.models.yolo_polygon_tolerance or 0.01
                for i, poly_xy in enumerate(result.masks.xy):
                    if len(poly_xy) < 3:
                        continue
                    label = self._yolo_model.names[int(result.boxes[i].cls)]
                    conf = float(result.boxes[i].conf)
                    contour = poly_xy.reshape(-1, 1, 2).astype(np.float32)
                    epsilon = tolerance * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) >= 3:
                        points = [(float(x), float(y)) for x, y in approx.squeeze()]
                        polygons.append(Polygon(points, label, conf))

        return bboxes, polygons

    def infer_sam3(self, image, src_shape) -> tuple[list, list]:
        """SAM3 inference. image: np.ndarray (BGR) or file path str.
        Returns (list of Bbox, list of Polygon)."""
        self._sam_predictor.set_image(image)
        # 直接改 predictor.args.conf, 調門檻就不必重建 predictor 重載整個 SAM3
        self._sam_predictor.args.conf = settings.models.sam3_conf or 0.25
        # 使用 dict.fromkeys 保序去重, 避免 set() 順序不確定導致標籤錯亂
        labels = list(dict.fromkeys(settings.class_names.text_prompts or []))
        bboxes, polygons = [], []
        t1 = time.time()
        masks, boxes = self._sam_predictor.inference_features(
            self._sam_predictor.features, src_shape=src_shape, text=labels
        )
        # boxes 為 (N, 6) = xyxy + score + cls, cls 是 text prompt 的索引 (非偵測序號);
        # masks 與 boxes 經過同一組 conf 過濾與 NMS, 兩者索引一一對應
        boxes_np = boxes.cpu().numpy() if boxes is not None else None
        if masks is not None:
            masks_np = masks.cpu().numpy()
            for i, mask in enumerate(masks_np):
                label, conf = sam3_label_conf(boxes_np, i, labels)
                mask_uint8 = (np.squeeze(mask) * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    tolerance = settings.models.sam3_polygon_tolerance or 0.01
                    for poly_pts in mask_to_polygon(contours, tolerance):
                        points = [(float(x), float(y)) for x, y in poly_pts]
                        polygons.append(Polygon(points, label, conf))
        if boxes_np is not None:
            for i, box in enumerate(boxes_np):
                label, conf = sam3_label_conf(boxes_np, i, labels)
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    bboxes.append(Bbox(x1, y1, x2 - x1, y2 - y1, label, conf))
        log.d(f"SAM3 inference time: {time.time() - t1}")
        return bboxes, polygons


inferencer = Inferencer()
