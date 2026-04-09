# 各種設定用的 Dialog (Categories, Param, TextPrompts, Sam3Mode, ConvertSettings)
# Updated: 2026-04-09
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.core import AppState
from src.utils.dynamic_settings import settings


class CategorySettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Categories")
        self.categories = settings.class_names.categories

        hint = QLabel("設定 VOC → YOLO 轉換時 class name 與 class id 的對應關係")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["class_name", "class_id"])
        self.table_widget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_category)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_category)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_categories)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(hint)
        main_layout.addWidget(self.table_widget)
        main_layout.addLayout(button_layout)

        self.load_categories()

    def load_categories(self):
        self.table_widget.setRowCount(0)
        for name, index in self.categories.items():
            self.add_row(name, str(index))

    def add_row(self, name="", index=""):
        row_count = self.table_widget.rowCount()
        self.table_widget.insertRow(row_count)
        name_item = QTableWidgetItem(name)
        index_item = QTableWidgetItem(index)
        self.table_widget.setItem(row_count, 0, name_item)
        self.table_widget.setItem(row_count, 1, index_item)

    def _next_class_id(self) -> int:
        """找出目前表格中最大的 class_id + 1"""
        max_id = -1
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 1)
            if item and item.text().isdigit():
                max_id = max(max_id, int(item.text()))
        return max_id + 1

    def add_category(self):
        self.add_row("", str(self._next_class_id()))

    def delete_category(self):
        selected_row = self.table_widget.currentRow()
        if selected_row >= 0:
            self.table_widget.removeRow(selected_row)

    def save_categories(self):
        categories = {}
        for row in range(self.table_widget.rowCount()):
            name_item = self.table_widget.item(row, 0)
            index_item = self.table_widget.item(row, 1)
            if name_item and index_item:
                name = name_item.text()
                index_str = index_item.text()
                if name and index_str.isdigit():
                    categories[name] = int(index_str)
        settings.class_names.categories = categories
        self.accept()


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
        self.polygon_tolerance_spin.setValue(settings.models.polygon_tolerance or 0.002)

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
        settings.models.polygon_tolerance = self.polygon_tolerance_spin.value()
        self.accept()


class TextPromptsDialog(QDialog):
    """編輯 SAM3 Text Prompts 的對話框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Text Prompts")
        self.setMinimumWidth(350)

        self.prompts = list(settings.class_names.text_prompts or [])

        hint = QLabel("提供給 SAM3 / Segmentation model 的文字提示，用於引導分割目標")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)

        self.list_widget = QTableWidget()
        self.list_widget.setColumnCount(1)
        self.list_widget.setHorizontalHeaderLabels(["Prompt"])
        self.list_widget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_prompt)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_prompt)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_prompts)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)

        content_layout = QHBoxLayout()
        content_layout.addWidget(self.list_widget)
        content_layout.addLayout(button_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(hint)
        main_layout.addLayout(content_layout)

        self.load_prompts()

    def load_prompts(self):
        self.list_widget.setRowCount(0)
        for prompt in self.prompts:
            row = self.list_widget.rowCount()
            self.list_widget.insertRow(row)
            self.list_widget.setItem(row, 0, QTableWidgetItem(prompt))

    def add_prompt(self):
        row = self.list_widget.rowCount()
        self.list_widget.insertRow(row)
        self.list_widget.setItem(row, 0, QTableWidgetItem(""))
        self.list_widget.editItem(self.list_widget.item(row, 0))

    def delete_prompt(self):
        selected_row = self.list_widget.currentRow()
        if selected_row >= 0:
            self.list_widget.removeRow(selected_row)

    def save_prompts(self):
        prompts = []
        for row in range(self.list_widget.rowCount()):
            item = self.list_widget.item(row, 0)
            if item and item.text().strip():
                prompts.append(item.text().strip())
        settings.class_names.text_prompts = prompts
        self.accept()


class Sam3ModeDialog(QDialog):
    """SAM3 輸出模式選擇對話框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM3 Output Mode")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("選擇 SAM3 偵測後要產生的標註類型："))

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Seg — 只產生 Polygon", "seg")
        self.mode_combo.addItem("BBox — 只產生 Bounding Box", "bbox")
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


class ConvertSettingsDialog(QDialog):
    """VOC → YOLO 轉換設定對話框（含 dataset split 比例）"""

    def __init__(self, parent=None, app_state: AppState = None):
        super().__init__(parent)
        self.app_state = app_state
        self.setWindowTitle("轉換設定 (Convert Settings)")
        self.setMinimumWidth(420)

        main_layout = QVBoxLayout(self)

        # --- YOLO 輸出模式 ---
        mode_group = QGroupBox("YOLO 輸出模式")
        mode_layout = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(
            "BBox - 標準邊界框 (class_id cx cy w h)", "bbox"
        )
        self.mode_combo.addItem(
            "Segmentation - 多邊形分割 (class_id x1 y1 ... xN yN)", "seg"
        )
        self.mode_combo.addItem(
            "OBB - 旋轉邊界框 (class_id x1 y1 x2 y2 x3 y3 x4 y4)", "obb"
        )
        self.mode_combo.model().item(2).setEnabled(False)
        mode_layout.addRow("輸出模式:", self.mode_combo)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)

        # --- Train / Val split ---
        split_group = QGroupBox("Train / Val 比例")
        split_layout = QFormLayout()

        self.train_spin = QSpinBox()
        self.train_spin.setRange(50, 100)
        self.train_spin.setSuffix(" %")
        self.train_spin.setSingleStep(5)
        self.train_spin.setValue(100)
        self.train_spin.valueChanged.connect(self._on_train_changed)

        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 50)
        self.val_spin.setSuffix(" %")
        self.val_spin.setValue(0)
        self.val_spin.setReadOnly(True)

        ratio_row = QHBoxLayout()
        ratio_row.addWidget(QLabel("Train:"))
        ratio_row.addWidget(self.train_spin)
        ratio_row.addWidget(QLabel("Val:"))
        ratio_row.addWidget(self.val_spin)
        split_layout.addRow(ratio_row)

        hint = QLabel(
            "圖片會依比例移動到 images/train 和 images/val\n"
            "Train 最少 50%，設為 100% 則不產生 val set"
        )
        hint.setStyleSheet("color: gray; font-size: 11px;")
        split_layout.addRow(hint)

        split_group.setLayout(split_layout)
        main_layout.addWidget(split_group)

        # --- 圖片處理方式（複製 / 搬移）---
        img_group = QGroupBox("圖片處理方式")
        img_layout = QVBoxLayout()

        self.radio_copy = QRadioButton("複製 (複製圖片到 images/，保留目前的圖片)")
        self.radio_move = QRadioButton("搬移 (將目前的圖片移到 images/，減少硬碟使用空間)")
        self.radio_copy.setChecked(True)

        self.img_mode_group = QButtonGroup(self)
        self.img_mode_group.addButton(self.radio_copy)
        self.img_mode_group.addButton(self.radio_move)

        img_layout.addWidget(self.radio_copy)
        img_layout.addWidget(self.radio_move)

        img_group.setLayout(img_layout)
        main_layout.addWidget(img_group)

        # --- 按鈕 ---
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("確定 (OK)")
        self.save_button.clicked.connect(self.save_settings)
        self.cancel_button = QPushButton("取消 (Cancel)")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        self.load_settings()

    def _on_train_changed(self, value: int):
        self.val_spin.setValue(100 - value)

    def load_settings(self):
        if self.app_state:
            index = self.mode_combo.findData(self.app_state.yolo_output_mode)
            if index >= 0:
                self.mode_combo.setCurrentIndex(index)

    def save_settings(self):
        if self.app_state:
            self.app_state.convert_format = "yolo"
            self.app_state.yolo_output_mode = self.mode_combo.currentData()
        self.accept()

    @property
    def train_ratio(self) -> float:
        return self.train_spin.value() / 100.0

    @property
    def val_ratio(self) -> float:
        return self.val_spin.value() / 100.0

    @property
    def copy_images(self) -> bool:
        """True = 複製圖片, False = 搬移圖片"""
        return self.radio_copy.isChecked()
