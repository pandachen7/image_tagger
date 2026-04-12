# Set YOLO Model 對話框：設定 YOLO 模型路徑、輸出模式、Polygon Tolerance
# 更新日期: 2026-04-12
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from src.utils.dynamic_settings import settings


class SetYoloModelDialog(QDialog):
    """設定 YOLO 模型路徑與輸出模式"""

    DEFAULT_MODEL = "yolo26s.pt"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set YOLO Model")
        self.setMinimumWidth(480)

        main_layout = QVBoxLayout(self)

        # --- Model 路徑 ---
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout()
        self.model_edit = QLineEdit(settings.models.model_path or "")
        self.model_edit.setPlaceholderText("YOLO model 路徑 (.pt)")
        model_browse = QPushButton("瀏覽...")
        model_browse.setFixedWidth(80)
        model_browse.clicked.connect(self._browse)
        model_reset = QPushButton("Reset")
        model_reset.setFixedWidth(60)
        model_reset.setToolTip(f"重設為預設模型 ({self.DEFAULT_MODEL})")
        model_reset.clicked.connect(lambda: self.model_edit.setText(self.DEFAULT_MODEL))
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(model_browse)
        model_layout.addWidget(model_reset)
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # --- 參數 ---
        param_group = QGroupBox("參數")
        param_layout = QFormLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("BBox — 只產生 Bounding Box", "bbox")
        self.mode_combo.addItem("Seg — 只產生 Polygon（需 seg model）", "seg")
        self.mode_combo.addItem("All — 同時產生 BBox 和 Polygon（需 seg model）", "all")
        current_mode = settings.models.yolo_label_mode or "bbox"
        idx = self.mode_combo.findData(current_mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        param_layout.addRow("Output Mode:", self.mode_combo)

        mode_hint = QLabel("Seg / All 模式需使用 segment model (如 yolo26m-seg.pt)")
        mode_hint.setStyleSheet("color: gray; font-size: 11px;")
        param_layout.addRow(mode_hint)

        tolerance_row = QHBoxLayout()
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.001, 0.1)
        self.tolerance_spin.setDecimals(3)
        self.tolerance_spin.setSingleStep(0.001)
        self.tolerance_spin.setValue(settings.models.yolo_polygon_tolerance or 0.002)
        tolerance_info = QLabel("\u2139")
        tolerance_info.setToolTip(
            "Polygon 簡化程度（僅 seg model 有效）\n"
            "越小越精密, 越大越粗糙\n"
            "0.001~0.005: 精密\n"
            "0.01~0.02: 中等\n"
            "0.05~0.1: 粗糙"
        )
        tolerance_row.addWidget(self.tolerance_spin)
        tolerance_row.addWidget(tolerance_info)
        param_layout.addRow("Polygon Tolerance:", tolerance_row)

        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)

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
        """儲存 YOLO 模型路徑與參數"""
        model_path = self.model_edit.text().strip()
        if model_path:
            settings.models.model_path = model_path
        settings.models.yolo_label_mode = self.mode_combo.currentData()
        settings.models.yolo_polygon_tolerance = self.tolerance_spin.value()
        self.accept()
