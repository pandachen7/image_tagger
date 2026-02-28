"""
Core module for managing application state and business logic.
This module decouples the ImageWidget from MainWindow by providing
a centralized state management system.
"""

from typing import Callable, Optional

from ultralytics import YOLO


class AppState:
    """Central state management for the image tagger application."""

    def __init__(self):
        # Auto save/detect flags
        self.auto_save = False
        self.auto_detect = False

        # Model management
        self.model: Optional[YOLO] = None
        self.use_model = False

        # Labels management
        self.preset_labels: dict[str, str] = {}
        self.last_used_label = "object"

        # Convert settings
        self.convert_format = "yolo"  # 轉換格式，目前只有 "yolo"
        self.yolo_obb_format = False  # 是否使用 YOLO OBB 格式（包含旋轉角度）
        self.yolo_seg_format = False  # 是否使用 YOLO Segmentation 格式

        # SAM model management (SAM3SemanticPredictor)
        self.sam_predictor = None  # feature extractor
        self.sam_inference = None  # inference predictor
        self.use_sam = False

        # Callbacks for UI updates
        self._callbacks: dict[str, list[Callable]] = {
            "auto_save_changed": [],
            "auto_detect_changed": [],
            "model_loaded": [],
            "detection_completed": [],
            "status_message": [],
            "sam_loaded": [],
            "sam_completed": [],
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
        if self.auto_detect:
            self._trigger_callback("detection_completed")

    def load_model(self, model_path: str) -> tuple[bool, str]:
        """
        Load a YOLO model.

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            if self.model:
                del self.model
            self.model = YOLO(model_path)
            self.use_model = True
            self.auto_detect = True
            message = f"Model loaded: {model_path}"
            self._trigger_callback("model_loaded", model_path)
            self._trigger_callback("auto_detect_changed", self.auto_detect)
            self._trigger_callback("status_message", message)
            self._trigger_callback("detection_completed")
            return True, message
        except Exception as e:
            message = f"Failed to load model: {e}"
            return False, message

    def load_sam_model(self, model_path: str) -> tuple[bool, str]:
        """
        Load a SAM3 model using SAM3SemanticPredictor.

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor

            if self.sam_predictor:
                del self.sam_predictor
            if self.sam_inference:
                del self.sam_inference

            overrides = dict(
                conf=0.50, task="segment", mode="predict", model=model_path, verbose=False
            )
            self.sam_predictor = SAM3SemanticPredictor(overrides=overrides)
            self.sam_inference = SAM3SemanticPredictor(overrides=overrides)
            self.sam_inference.setup_model()
            self.use_sam = True
            message = f"SAM model loaded: {model_path}"
            self._trigger_callback("sam_loaded", model_path)
            self._trigger_callback("status_message", message)
            return True, message
        except Exception as e:
            
            message = f"Failed to load SAM model ({model_path}): {e}"
            return False, message

    def set_last_used_label(self, label: str):
        """Set the last used label."""
        self.last_used_label = label.strip()

    def get_label_by_key(self, key: str) -> str:
        """Get a label by its key from preset labels."""
        return self.preset_labels.get(key, self.last_used_label)

    def is_model_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self.model is not None
