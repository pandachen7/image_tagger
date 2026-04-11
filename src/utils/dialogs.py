# 各種設定用的 Dialog (Categories, Param, TextPrompts, Sam3Mode, ConvertSettings, CategorizeMedia)
# Updated: 2026-04-11
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import cv2
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
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
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.core import AppState
from src.utils.const import ALL_EXTS, IMAGE_EXTS, VIDEO_EXTS
from src.utils.dynamic_settings import settings
from src.utils.logger import getUniqueLogger

log = getUniqueLogger(__file__)


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


class CategorizeMediaDialog(QDialog):
    """依 YOLO 偵測結果將媒體檔案分類到子資料夾"""

    DEFAULT_MODEL = "yolo26s.pt"
    NOT_DETECTED_FOLDER = "not_detected"
    VIDEO_SAMPLE_FRAMES = 5

    def __init__(
        self, parent=None, default_folder: str = "", default_model: str = ""
    ):
        super().__init__(parent)
        self.setWindowTitle("Categorize Media")
        self.setMinimumWidth(500)
        self._canceled = False

        main_layout = QVBoxLayout(self)

        # 說明
        hint = QLabel(
            "使用 YOLO 模型偵測資料夾中的圖片與影片，\n"
            "依偵測到最多次的物件名稱，將檔案分類到對應的子資料夾\n"
            "（也可使用 SAM3 model，但分類效果通常不如 YOLO）"
        )
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)
        main_layout.addWidget(hint)

        # --- 資料夾選擇 ---
        form = QFormLayout()
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit(default_folder)
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setPlaceholderText("選擇要分類的資料夾")
        folder_browse = QPushButton("瀏覽...")
        folder_browse.setFixedWidth(80)
        folder_browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(folder_browse)
        form.addRow("資料夾:", folder_row)

        # --- Model 選擇 ---
        model_row = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItem("YOLO", "yolo")
        self.type_combo.addItem("YOLO-Seg", "yolo-seg")
        self.type_combo.addItem("SAM3", "sam3")
        self.type_combo.setFixedWidth(100)
        self.model_edit = QLineEdit(default_model)
        self.model_edit.setReadOnly(True)
        self.model_edit.setPlaceholderText("選擇用於分類的 model (.pt)")
        model_browse = QPushButton("瀏覽...")
        model_browse.setFixedWidth(80)
        model_browse.clicked.connect(self._browse_model)
        model_reset = QPushButton("Reset")
        model_reset.setFixedWidth(60)
        model_reset.setToolTip(f"重設為預設模型 ({self.DEFAULT_MODEL})")
        model_reset.clicked.connect(self._reset_model)
        model_row.addWidget(self.type_combo)
        model_row.addWidget(self.model_edit)
        model_row.addWidget(model_browse)
        model_row.addWidget(model_reset)
        form.addRow("Model:", model_row)
        main_layout.addLayout(form)

        # --- 按鈕 ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.start_btn = QPushButton("開始偵測")
        self.start_btn.clicked.connect(self._run)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(btn_layout)

        # --- 進度區域（類似狀態列）---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-size: 11px;")
        main_layout.addWidget(self.status_label)

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "選擇資料夾", self.folder_edit.text()
        )
        if path:
            self.folder_edit.setText(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇 Model", self.model_edit.text(), "Model Files (*.pt)"
        )
        if path:
            self.model_edit.setText(path)
            detected = self._detect_model_type(path)
            if detected:
                idx = self.type_combo.findData(detected)
                if idx >= 0:
                    self.type_combo.setCurrentIndex(idx)

    def _reset_model(self):
        """重設為預設 YOLO model"""
        self.model_edit.setText(self.DEFAULT_MODEL)
        self.type_combo.setCurrentIndex(0)  # YOLO

    @staticmethod
    def _detect_model_type(model_path: str) -> str | None:
        """偵測 .pt 模型的類型 (yolo / yolo-seg / sam3)"""
        try:
            import torch
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "model" in ckpt:
                cls_name = type(ckpt["model"]).__name__.lower()
                if "sam" in cls_name:
                    return "sam3"
                task = getattr(ckpt["model"], "task", "") or ""
                if task == "segment":
                    return "yolo-seg"
                return "yolo"
        except Exception:
            log.w(f"無法偵測模型類型: {model_path}")
        return None

    def _on_cancel(self):
        """取消按鈕：偵測中則中斷，否則關閉"""
        self._canceled = True
        self.reject()

    def _run(self):
        """開始偵測並分類"""
        folder = self.folder_edit.text()
        model_path = self.model_edit.text()

        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "Warning", "請選擇有效的資料夾")
            return
        if not model_path or not Path(model_path).is_file():
            QMessageBox.warning(self, "Warning", "請選擇有效的 Model 檔案")
            return

        # 收集媒體檔案（不含子資料夾）
        base = Path(folder)
        media_files = sorted(
            f for f in base.iterdir()
            if f.is_file() and f.suffix.lower() in ALL_EXTS
        )
        if not media_files:
            QMessageBox.warning(self, "Warning", "資料夾中沒有找到圖片或影片檔案")
            return

        # 載入 model
        model_type = self.type_combo.currentData()
        self.start_btn.setEnabled(False)
        self._canceled = False
        self.status_label.setText("正在載入模型...")
        QApplication.processEvents()

        sam3_labels: list[str] = []
        try:
            if model_type == "sam3":
                from ultralytics.models.sam import SAM3SemanticPredictor

                sam3_labels = list(
                    dict.fromkeys(settings.class_names.text_prompts or [])
                )
                if not sam3_labels:
                    QMessageBox.warning(
                        self, "Warning",
                        "SAM3 需要 Text Prompts 才能偵測，\n"
                        "請先在 Edit → Text Prompts 中設定",
                    )
                    self.start_btn.setEnabled(True)
                    return
                overrides = dict(
                    conf=0.25, imgsz=630, task="segment",
                    mode="predict", model=model_path, half=True, verbose=False,
                )
                model = SAM3SemanticPredictor(overrides=overrides)
            else:
                from ultralytics import YOLO
                model = YOLO(model_path)
        except Exception:
            log.e(f"無法載入模型: {model_path}")
            QMessageBox.critical(self, "Error", "模型載入失敗，請確認檔案是否正確")
            self.start_btn.setEnabled(True)
            return

        # 偵測每個檔案
        total = len(media_files)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)

        # {file_path: subfolder_name}
        file_to_subfolder: dict[Path, str] = {}

        for i, file_path in enumerate(media_files):
            if self._canceled:
                break

            self.status_label.setText(
                f"偵測中: {file_path.name} ({i + 1}/{total})"
            )
            self.progress_bar.setValue(i)
            QApplication.processEvents()

            try:
                if model_type == "sam3":
                    class_counts = self._detect_file_sam3(
                        model, file_path, sam3_labels
                    )
                else:
                    class_counts = self._detect_file(model, file_path)
            except Exception:
                log.e(f"偵測失敗: {file_path.name}")
                class_counts = {}

            if not class_counts:
                subfolder = self.NOT_DETECTED_FOLDER
            else:
                max_count = max(class_counts.values())
                top_classes = sorted(
                    name for name, cnt in class_counts.items() if cnt == max_count
                )
                subfolder = "+".join(top_classes)

            file_to_subfolder[file_path] = subfolder

        if self._canceled:
            self.status_label.setText("已取消")
            self.progress_bar.setVisible(False)
            self.start_btn.setEnabled(True)
            return

        # 搬移檔案
        self.status_label.setText("正在搬移檔案...")
        QApplication.processEvents()

        moved_counts: dict[str, int] = {}
        for file_path, subfolder in file_to_subfolder.items():
            dest_dir = base / subfolder
            dest_dir.mkdir(exist_ok=True)
            shutil.move(str(file_path), str(dest_dir / file_path.name))
            moved_counts[subfolder] = moved_counts.get(subfolder, 0) + 1

        self.progress_bar.setValue(total)
        self.status_label.setText("完成")

        # 結果摘要
        lines = ["分類完成\n"]
        for subfolder in sorted(moved_counts.keys()):
            lines.append(f"  {subfolder}/: {moved_counts[subfolder]} 個檔案")
        lines.append(f"\n共處理 {total} 個檔案")
        QMessageBox.information(self, "Categorize Media 結果", "\n".join(lines))
        self.start_btn.setEnabled(True)

    def _detect_file(self, model, file_path: Path) -> dict[str, int]:
        """偵測單一檔案，回傳 {class_name: count}"""
        counts: Counter = Counter()
        suffix = file_path.suffix.lower()

        if suffix in IMAGE_EXTS:
            img = cv2.imread(str(file_path))
            if img is not None:
                self._count_detections(model, img, counts)
        elif suffix in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(file_path))
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    for idx in self._sample_frame_indices(total_frames):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            self._count_detections(model, frame, counts)
            finally:
                cap.release()

        return dict(counts)

    @staticmethod
    def _count_detections(model, img, counts: Counter):
        """對單一影像跑 YOLO 推論並累加 class_name 計數"""
        results = model.predict(img, verbose=False)
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    name = model.names[int(box.cls)]
                    counts[name] += 1

    def _detect_file_sam3(
        self, predictor, file_path: Path, labels: list[str]
    ) -> dict[str, int]:
        """SAM3 偵測單一檔案，回傳 {class_name: count}"""
        counts: Counter = Counter()
        suffix = file_path.suffix.lower()

        if suffix in IMAGE_EXTS:
            img = cv2.imread(str(file_path))
            if img is not None:
                self._count_sam3(predictor, img, labels, counts)
        elif suffix in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(file_path))
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    for idx in self._sample_frame_indices(total_frames):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            self._count_sam3(predictor, frame, labels, counts)
            finally:
                cap.release()

        return dict(counts)

    @staticmethod
    def _count_sam3(predictor, img, labels: list[str], counts: Counter):
        """對單一影像跑 SAM3 推論並累加 class_name 計數"""
        predictor.set_image(img)
        src_shape = img.shape[:2]
        masks, boxes = predictor.inference_features(
            predictor.features, src_shape=src_shape, text=labels
        )
        if boxes is not None:
            for i in range(boxes.shape[0]):
                box = boxes[i].cpu().numpy() if hasattr(boxes[i], "cpu") else boxes[i]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    label = labels[i] if i < len(labels) else labels[-1]
                    counts[label] += 1

    def _sample_frame_indices(self, total: int) -> list[int]:
        """從影片中均勻取樣 frame indices"""
        n = min(self.VIDEO_SAMPLE_FRAMES, total)
        if n <= 1:
            return [0]
        return [int(i * (total - 1) / (n - 1)) for i in range(n)]
