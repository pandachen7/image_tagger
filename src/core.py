"""
Core module for managing application state and business logic.
This module decouples the ImageWidget from MainWindow by providing
a centralized state management system.

更新日期: 2026-08-06
"""

from typing import Callable


class AppState:
    """Central state management for the image tagger application."""

    def __init__(self):
        # Auto save/detect flags
        self.auto_save = False
        self.auto_detect = False

        # Labels management
        self.preset_labels: dict[str, str] = {}
        self.last_used_label = "object"

        # Multi-digit key input buffer
        self._key_buffer = ""

        # Convert settings
        self.convert_format = "yolo"
        self.yolo_output_mode = "bbox"  # "bbox", "seg", "obb"

        # Callbacks for UI updates
        self._callbacks: dict[str, list[Callable]] = {
            "auto_save_changed": [],
            "auto_detect_changed": [],
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
        """Toggle auto save mode.

        auto_save 專門負責「自動產生的標註」落檔 (auto_detect 的推論結果、
        影片定時抽幀), 因此依附於 auto_detect; 未開啟 auto_detect 時不可切換。
        手動畫的框由 g_param.user_labeling 負責, 不受此開關影響。
        """
        if not self.auto_detect:
            self._trigger_callback(
                "status_message", "Auto save 需先開啟 Auto Detect"
            )
            return
        self.auto_save = not self.auto_save
        self._trigger_callback("auto_save_changed", self.auto_save)
        self._trigger_callback(
            "status_message", f"Auto save: {'on' if self.auto_save else 'off'}"
        )

    def toggle_auto_detect(self):
        """Toggle auto detect mode. 關閉時一併關閉 auto_save。"""
        self.auto_detect = not self.auto_detect
        # auto_save 依附於 auto_detect, 否則會出現選項灰掉卻仍打勾且持續存檔的狀態
        if not self.auto_detect and self.auto_save:
            self.auto_save = False
            self._trigger_callback("auto_save_changed", False)
        self._trigger_callback("auto_detect_changed", self.auto_detect)
        self._trigger_callback(
            "status_message", f"Auto detect: {'on' if self.auto_detect else 'off'}"
        )

    def set_last_used_label(self, label: str):
        """Set the last used label."""
        self.last_used_label = label.strip()

    def get_label_by_key(self, key: str) -> str:
        """Get a label by its key from preset labels."""
        return self.preset_labels.get(key, self.last_used_label)

    def append_key_buffer(self, digit: str) -> str:
        """Append a digit to the key buffer and return current buffer."""
        self._key_buffer += digit
        return self._key_buffer

    def clear_key_buffer(self):
        self._key_buffer = ""

    @property
    def key_buffer(self) -> str:
        return self._key_buffer

    def resolve_key_buffer(self) -> str | None:
        """Try to resolve the current buffer to a label.
        Returns the label if matched, None otherwise.
        """
        label = self.preset_labels.get(self._key_buffer)
        self._key_buffer = ""
        return label

    def is_unique_prefix(self) -> bool:
        """Check if current buffer exactly matches one key and
        no other key starts with this prefix (so we can apply immediately).
        """
        buf = self._key_buffer
        if not buf:
            return False
        if buf not in self.preset_labels:
            return False
        # Check no other key has this as a strict prefix
        return not any(k != buf and k.startswith(buf) for k in self.preset_labels)

    def has_any_prefix_match(self) -> bool:
        """Check if any preset key starts with the current buffer."""
        buf = self._key_buffer
        return any(k.startswith(buf) for k in self.preset_labels)

    def get_prefix_matches(self) -> list[tuple[str, str]]:
        """Return all (key, label) pairs that start with the current buffer."""
        buf = self._key_buffer
        return [
            (k, v) for k, v in self.preset_labels.items() if k.startswith(buf)
        ]
