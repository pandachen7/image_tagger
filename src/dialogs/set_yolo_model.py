# Set YOLO Model 對話框：設定 YOLO 模型路徑
# 更新日期: 2026-04-12
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from src.utils.dynamic_settings import settings


class SetYoloModelDialog(QDialog):
    """設定 YOLO 模型路徑"""

    DEFAULT_MODEL = "yolo26s.pt"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set YOLO Model")
        self.setMinimumWidth(450)

        main_layout = QVBoxLayout(self)

        hint = QLabel("選擇用於偵測的 YOLO 模型檔案 (.pt)")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        main_layout.addWidget(hint)

        # --- Model 路徑 ---
        model_row = QHBoxLayout()
        self.model_edit = QLineEdit(settings.models.model_path or "")
        self.model_edit.setPlaceholderText("YOLO model 路徑 (.pt)")
        model_browse = QPushButton("瀏覽...")
        model_browse.setFixedWidth(80)
        model_browse.clicked.connect(self._browse)
        model_reset = QPushButton("Reset")
        model_reset.setFixedWidth(60)
        model_reset.setToolTip(f"重設為預設模型 ({self.DEFAULT_MODEL})")
        model_reset.clicked.connect(lambda: self.model_edit.setText(self.DEFAULT_MODEL))
        model_row.addWidget(QLabel("Model:"))
        model_row.addWidget(self.model_edit)
        model_row.addWidget(model_browse)
        model_row.addWidget(model_reset)
        main_layout.addLayout(model_row)

        # --- 按鈕 ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        save_btn = QPushButton("確定")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        main_layout.addLayout(btn_layout)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇 YOLO Model", self.model_edit.text(), "Model Files (*.pt)"
        )
        if path:
            self.model_edit.setText(path)

    def _save(self):
        self.model_path = self.model_edit.text().strip()
        if self.model_path:
            settings.models.model_path = self.model_path
        self.accept()
