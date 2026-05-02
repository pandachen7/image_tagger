# Categorize Media 對話框：依 YOLO/SAM3 偵測結果將媒體檔案分類到子資料夾
# 更新日期: 2026-04-12
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import cv2
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from src.utils.const import ALL_EXTS, IMAGE_EXTS, VIDEO_EXTS
from src.utils.dynamic_settings import settings
from src.utils.func import imread_unicode
from src.utils.logger import getUniqueLogger

log = getUniqueLogger(__file__)


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
        self.close_dialog_btn = QPushButton("關閉")
        self.close_dialog_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.close_dialog_btn)
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
            img = imread_unicode(file_path)
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
            img = imread_unicode(file_path)
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
