# Param 對話框：調整模型輸出的後處理參數
# 更新日期: 2026-04-12
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
)

from src.utils.dynamic_settings import settings


class ParamDialog(QDialog):
    """編輯參數的對話框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Param")
        self.setMinimumWidth(300)

        layout = QFormLayout(self)

        hint = QLabel("調整模型輸出的後處理參數，影響標註精度與效能")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)
        layout.addRow(hint)

        self.polygon_tolerance_spin = QDoubleSpinBox()
        self.polygon_tolerance_spin.setRange(0.001, 0.1)
        self.polygon_tolerance_spin.setDecimals(3)
        self.polygon_tolerance_spin.setSingleStep(0.001)
        self.polygon_tolerance_spin.setValue(settings.models.sam3_polygon_tolerance or 0.002)

        tolerance_tip = (
            "越小越精密, 越大越粗糙\n"
            "0.001~0.005: 精密\n"
            "0.01~0.02: 中等\n"
            "0.05~0.1: 粗糙"
        )
        info_label = QLabel("\u2139")
        info_label.setToolTip(tolerance_tip)

        row_layout = QHBoxLayout()
        row_layout.addWidget(self.polygon_tolerance_spin)
        row_layout.addWidget(info_label)
        layout.addRow("Polygon Tolerance:", row_layout)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addRow(btn_layout)

    def save(self):
        settings.models.sam3_polygon_tolerance = self.polygon_tolerance_spin.value()
        self.accept()
