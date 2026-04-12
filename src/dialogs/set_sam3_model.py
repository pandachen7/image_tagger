# Set SAM3 Model 對話框：設定 SAM3 模型路徑、輸出模式、Polygon Tolerance、Text Prompts
# 更新日期: 2026-04-12
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.utils.dynamic_settings import settings


class SetSam3ModelDialog(QDialog):
    """設定 SAM3 模型路徑與相關參數（輸出模式、Polygon Tolerance、Text Prompts）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set SAM3 Model")
        self.setMinimumWidth(480)

        main_layout = QVBoxLayout(self)

        # --- Model 路徑 ---
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout()
        self.model_edit = QLineEdit(settings.models.sam3_model_path or "")
        self.model_edit.setPlaceholderText("SAM3 model 路徑 (.pt)")
        model_browse = QPushButton("瀏覽...")
        model_browse.setFixedWidth(80)
        model_browse.clicked.connect(self._browse_model)
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(model_browse)
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # --- SAM3 Output Mode + Polygon Tolerance ---
        param_group = QGroupBox("參數")
        param_layout = QFormLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Seg — 只產生 Polygon", "seg")
        self.mode_combo.addItem("BBox — 只產生 Bounding Box", "bbox")
        self.mode_combo.addItem("All — 同時產生 Polygon 和 BBox", "all")
        current_mode = settings.models.sam3_label_mode or "seg"
        idx = self.mode_combo.findData(current_mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        param_layout.addRow("Output Mode:", self.mode_combo)

        mode_hint = QLabel("Seg 用於訓練 segment 模型，BBox 用於訓練 object detect 模型")
        mode_hint.setStyleSheet("color: gray; font-size: 11px;")
        param_layout.addRow(mode_hint)

        tolerance_row = QHBoxLayout()
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.001, 0.1)
        self.tolerance_spin.setDecimals(3)
        self.tolerance_spin.setSingleStep(0.001)
        self.tolerance_spin.setValue(settings.models.polygon_tolerance or 0.002)
        tolerance_info = QLabel("\u2139")
        tolerance_info.setToolTip(
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

        # --- Text Prompts ---
        prompts_group = QGroupBox("Text Prompts")
        prompts_layout = QVBoxLayout()

        prompts_hint = QLabel("提供給 SAM3 的文字提示，用於引導分割目標")
        prompts_hint.setStyleSheet("color: gray; font-size: 11px;")
        prompts_layout.addWidget(prompts_hint)

        content_layout = QHBoxLayout()
        self.prompts_table = QTableWidget()
        self.prompts_table.setColumnCount(1)
        self.prompts_table.setHorizontalHeaderLabels(["Prompt"])
        self.prompts_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        content_layout.addWidget(self.prompts_table)

        prompt_btn_layout = QVBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._add_prompt)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_prompt)
        prompt_btn_layout.addWidget(add_btn)
        prompt_btn_layout.addWidget(delete_btn)
        prompt_btn_layout.addStretch()
        content_layout.addLayout(prompt_btn_layout)

        prompts_layout.addLayout(content_layout)
        prompts_group.setLayout(prompts_layout)
        main_layout.addWidget(prompts_group)

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

        self._load_prompts()

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇 SAM3 Model", self.model_edit.text(), "Model Files (*.pt)"
        )
        if path:
            self.model_edit.setText(path)

    # --- Text Prompts ---

    def _load_prompts(self):
        prompts = settings.class_names.text_prompts or []
        self.prompts_table.setRowCount(0)
        for prompt in prompts:
            row = self.prompts_table.rowCount()
            self.prompts_table.insertRow(row)
            self.prompts_table.setItem(row, 0, QTableWidgetItem(prompt))

    def _add_prompt(self):
        row = self.prompts_table.rowCount()
        self.prompts_table.insertRow(row)
        self.prompts_table.setItem(row, 0, QTableWidgetItem(""))
        self.prompts_table.editItem(self.prompts_table.item(row, 0))

    def _delete_prompt(self):
        selected = self.prompts_table.currentRow()
        if selected >= 0:
            self.prompts_table.removeRow(selected)

    def _save(self):
        """儲存所有 SAM3 相關設定"""
        # Model path
        model_path = self.model_edit.text().strip()
        if model_path:
            settings.models.sam3_model_path = model_path
        self.model_path = model_path

        # Output mode
        settings.models.sam3_label_mode = self.mode_combo.currentData()

        # Polygon tolerance
        settings.models.polygon_tolerance = self.tolerance_spin.value()

        # Text prompts
        prompts = []
        for row in range(self.prompts_table.rowCount()):
            item = self.prompts_table.item(row, 0)
            if item and item.text().strip():
                prompts.append(item.text().strip())
        settings.class_names.text_prompts = prompts

        self.accept()
