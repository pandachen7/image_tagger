"""
Core module for managing application state and business logic.
This module decouples the ImageWidget from MainWindow by providing
a centralized state management system.
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
