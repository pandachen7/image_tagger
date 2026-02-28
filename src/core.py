"""
Core module for managing application state and business logic.
This module decouples the ImageWidget from MainWindow by providing
a centralized state management system.
"""

from typing import Callable, Optional

from src.utils.model import Bbox, ModelType, Polygon


class AppState:
    """Central state management for the image tagger application."""

    def __init__(self):
        # Auto save/detect flags
        self.auto_save = False
        self.auto_detect = False

        # Unified model management
        self.active_model_type: str = ModelType.NONE
        self.model_path: Optional[str] = None
        self.sam_model_path: Optional[str] = None
        self._yolo_model = None
        self._sam_predictor = None
        self._sam_inference = None

        # Labels management
        self.preset_labels: dict[str, str] = {}
        self.last_used_label = "object"

        # Convert settings
        self.convert_format = "yolo"
        self.yolo_obb_format = False
        self.yolo_seg_format = False

        # Callbacks for UI updates
        self._callbacks: dict[str, list[Callable]] = {
            "auto_save_changed": [],
            "auto_detect_changed": [],
            "model_changed": [],
            "inference_completed": [],
            "status_message": [],
        }

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for a specific event."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _trigger_callback(self, event: str, *args, **kwargs):
        """Trigger all callbacks for a specific event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                callback(*args, **kwargs)

    def toggle_auto_save(self):
        """Toggle auto save mode."""
        self.auto_save = not self.auto_save
        self._trigger_callback("auto_save_changed", self.auto_save)
        self._trigger_callback(
            "status_message", f"Auto save: {'on' if self.auto_save else 'off'}"
        )

    def toggle_auto_detect(self):
        """Toggle auto detect mode."""
        self.auto_detect = not self.auto_detect
        self._trigger_callback("auto_detect_changed", self.auto_detect)
        self._trigger_callback(
            "status_message", f"Auto detect: {'on' if self.auto_detect else 'off'}"
        )

    def set_active_model(self, model_type: str, model_path: str = None):
        """Set the active model type and optionally its path."""
        self.active_model_type = model_type
        if model_path:
            if model_type == ModelType.YOLO:
                self.model_path = model_path
            elif model_type == ModelType.SAM3:
                self.sam_model_path = model_path
        self._trigger_callback("model_changed")
        self._trigger_callback("status_message", f"Active model: {model_type}")

    def ensure_loaded(self) -> bool:
        """Lazy-load the active model. Returns True if ready."""
        if self.active_model_type == ModelType.YOLO:
            if self._yolo_model is None and self.model_path:
                from ultralytics import YOLO

                self._yolo_model = YOLO(self.model_path)
            return self._yolo_model is not None
        elif self.active_model_type == ModelType.SAM3:
            if self._sam_predictor is None and self.sam_model_path:
                from ultralytics.models.sam import SAM3SemanticPredictor

                overrides = dict(
                    conf=0.50,
                    task="segment",
                    mode="predict",
                    model=self.sam_model_path,
                    verbose=False,
                )
                self._sam_predictor = SAM3SemanticPredictor(overrides=overrides)
                self._sam_inference = SAM3SemanticPredictor(overrides=overrides)
                self._sam_inference.setup_model()
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
        import cv2
        import numpy as np

        self._sam_predictor.set_image(image_path)
        labels = list(set(self.preset_labels.values()))
        bboxes, polygons = [], []
        for label in labels:
            masks, boxes = self._sam_inference.inference_features(
                self._sam_predictor.features, src_shape=src_shape, text=[label]
            )
            if masks is not None:
                masks_np = masks.cpu().numpy()
                for mask in masks_np:
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        points = [(float(pt[0][0]), float(pt[0][1])) for pt in largest]
                        if len(points) >= 3:
                            polygons.append(Polygon(points, label, -1.0))
            if boxes is not None:
                boxes_np = boxes.cpu().numpy()
                for box in boxes_np:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    if (x2 - x1) > 0 and (y2 - y1) > 0:
                        bboxes.append(Bbox(x1, y1, x2 - x1, y2 - y1, label, -1.0))
        return bboxes, polygons

    def set_last_used_label(self, label: str):
        """Set the last used label."""
        self.last_used_label = label.strip()

    def get_label_by_key(self, key: str) -> str:
        """Get a label by its key from preset labels."""
        return self.preset_labels.get(key, self.last_used_label)

    def is_model_loaded(self) -> bool:
        """Check if the active model is loaded."""
        if self.active_model_type == ModelType.YOLO:
            return self._yolo_model is not None
        elif self.active_model_type == ModelType.SAM3:
            return self._sam_predictor is not None
        return False
