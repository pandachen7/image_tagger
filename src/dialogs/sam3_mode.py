# SAM3 Output Mode 對話框：選擇 SAM3 偵測後產生的標註類型
# 更新日期: 2026-04-12
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from src.utils.dynamic_settings import settings


class Sam3ModeDialog(QDialog):
    """SAM3 輸出模式選擇對話框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM3 Output Mode")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("選擇 SAM3 偵測後要產生的標註類型："))

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("BBox — 只產生 Bounding Box", "bbox")
        self.mode_combo.addItem("Seg — 只產生 Polygon", "seg")
        self.mode_combo.addItem("All — 同時產生 Polygon 和 BBox", "all")
        layout.addWidget(self.mode_combo)

        hint = QLabel("Seg用於訓練 segment 模型，BBox用於訓練 object detect 模型")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(hint)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        save_btn = QPushButton("確定")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        # 載入目前設定
        current = settings.models.sam3_label_mode or "seg"
        idx = self.mode_combo.findData(current)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

    def _save(self):
        settings.models.sam3_label_mode = self.mode_combo.currentData()
        self.accept()
