# VOC → YOLO 轉換設定對話框：含 class mapping、資料夾選擇、dataset split 比例
# 更新日期: 2026-04-12
from pathlib import Path

from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from src.core import AppState
from src.dialogs.class_mapping import ClassMappingDialog
from src.utils.const import IMAGE_EXTS
from src.utils.dynamic_settings import save_settings, settings


class ConvertSettingsDialog(QDialog):
    """VOC → YOLO 轉換設定對話框（含 class mapping、資料夾選擇、dataset split 比例）"""

    def __init__(
        self, parent=None, app_state: AppState = None, default_folder: str = ""
    ):
        super().__init__(parent)
        self.app_state = app_state
        self.setWindowTitle("轉換設定 (Convert Settings)")
        self.setMinimumWidth(480)

        main_layout = QVBoxLayout(self)

        # --- Class Mapping ---
        mapping_group = QGroupBox("Class Mapping")
        mapping_layout = QHBoxLayout()
        self.mapping_info = QLabel()
        self.mapping_info.setStyleSheet("color: gray; font-size: 11px;")
        mapping_edit_btn = QPushButton("編輯 Mapping")
        mapping_edit_btn.clicked.connect(self._open_class_mapping)
        mapping_layout.addWidget(self.mapping_info, 1)
        mapping_layout.addWidget(mapping_edit_btn)
        mapping_group.setLayout(mapping_layout)
        main_layout.addWidget(mapping_group)

        # --- 資料夾 ---
        folder_group = QGroupBox("包含圖片與VOC標籤檔的資料夾")
        folder_layout = QVBoxLayout()
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit(default_folder)
        self.folder_edit.setPlaceholderText("輸入或選擇包含 VOC XML 的資料夾路徑")
        self.folder_edit.editingFinished.connect(self._update_folder_info)
        folder_browse = QPushButton("瀏覽...")
        folder_browse.setFixedWidth(80)
        folder_browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(folder_browse)
        folder_layout.addLayout(folder_row)
        self.folder_info = QLabel()
        self.folder_info.setStyleSheet("color: gray; font-size: 11px;")
        folder_layout.addWidget(self.folder_info)
        folder_group.setLayout(folder_layout)
        main_layout.addWidget(folder_group)

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
        self.train_spin.setValue(80)
        self.train_spin.valueChanged.connect(self._on_train_changed)

        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 50)
        self.val_spin.setSuffix(" %")
        self.val_spin.setValue(20)
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
        self._update_mapping_info()
        self._update_folder_info()

    def _on_train_changed(self, value: int):
        self.val_spin.setValue(100 - value)

    def load_settings(self):
        if self.app_state:
            index = self.mode_combo.findData(self.app_state.yolo_output_mode)
            if index >= 0:
                self.mode_combo.setCurrentIndex(index)

    def save_settings(self):
        if not self.folder_path or not Path(self.folder_path).is_dir():
            QMessageBox.warning(self, "Warning", "請選擇有效的資料夾")
            return
        if self.app_state:
            self.app_state.convert_format = "yolo"
            self.app_state.yolo_output_mode = self.mode_combo.currentData()
        self.accept()

    # --- Class Mapping ---

    def _update_mapping_info(self):
        """更新 Class Mapping 摘要資訊"""
        categories = settings.class_names.categories
        count = len(categories)
        if count == 0:
            self.mapping_info.setText("尚未設定 class mapping")
            return
        items = list(categories.items())
        preview = ", ".join(f"{name}→{cid}" for name, cid in items[:3])
        if count > 3:
            preview += " ..."
        self.mapping_info.setText(f"目前 {count} 組: {preview}")

    def _open_class_mapping(self):
        """開啟 Class Mapping 編輯對話框"""
        dialog = ClassMappingDialog(self)
        if dialog.exec():
            save_settings()
            self._update_mapping_info()

    # --- 資料夾 ---

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "選擇包含 VOC XML 的資料夾", self.folder_edit.text()
        )
        if path:
            self.folder_edit.setText(path)
            self._update_folder_info()

    def _update_folder_info(self):
        """更新資料夾中的圖片數量"""
        folder = self.folder_edit.text().strip()
        if not folder or not Path(folder).is_dir():
            self.folder_info.setText("")
            return
        base = Path(folder)
        img_count = sum(
            1 for f in base.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        self.folder_info.setText(f"圖片: {img_count} 張")

    @property
    def folder_path(self) -> str:
        return self.folder_edit.text().strip()

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
