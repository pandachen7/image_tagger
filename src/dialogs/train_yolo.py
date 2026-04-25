# Train YOLO 對話框：選擇 dataset.yaml、設定訓練參數、執行 ultralytics 訓練並顯示進度與結果
# 更新日期: 2026-04-25
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from src.dialogs.train_yolo_advanced import TrainYoloAdvancedDialog
from src.utils.dynamic_settings import save_settings, settings
from src.utils.logger import getUniqueLogger

log = getUniqueLogger(__file__)


def _resolve_cache(cache_str: str | None):
    """把 settings 中的 cache 字串轉成 ultralytics 接受的型別 (False / 'ram' / 'disk')"""
    val = (cache_str or "false").lower()
    if val in ("false", "0", ""):
        return False
    if val in ("true", "ram"):
        return "ram"
    if val == "disk":
        return "disk"
    return False


def _build_train_kwargs(name: str) -> dict:
    """依 settings.training 組合 ultralytics model.train() 所需的 kwargs"""
    t = settings.training
    return {
        # 資料與輸出
        "data": t.last_data_yaml,
        "name": name,
        "exist_ok": False,
        "plots": True,
        # 訓練核心
        "epochs": t.epochs,
        "patience": t.patience,
        "batch": t.batch,
        "imgsz": t.imgsz,
        "device": _parse_device(t.device or "0"),
        "seed": 42,
        # 儲存
        "save": True,
        "save_period": t.save_period,
        # 驗證
        "val": True,
        # 優化器
        "optimizer": t.optimizer,
        "lr0": t.lr0,
        "lrf": t.lrf,
        "weight_decay": t.weight_decay,
        "warmup_epochs": t.warmup_epochs,
        "warmup_momentum": t.warmup_momentum,
        # 幾何
        "degrees": t.degrees,
        "translate": t.translate,
        "scale": t.scale,
        "perspective": t.perspective,
        # 翻轉
        "flipud": t.flipud,
        "fliplr": t.fliplr,
        # 色彩
        "hsv_h": t.hsv_h,
        "hsv_s": t.hsv_s,
        "hsv_v": t.hsv_v,
        # 混合增強
        "mosaic": t.mosaic,
        "close_mosaic": t.close_mosaic,
        "mixup": t.mixup,
        "copy_paste": t.copy_paste,
        # 系統
        "workers": t.workers,
        "cache": _resolve_cache(t.cache),
        "rect": bool(t.rect),
        "amp": bool(t.amp),
        "fraction": t.fraction,
        "freeze": t.freeze if (t.freeze or 0) > 0 else None,
    }


def _parse_device(text: str):
    """解析 device 字串: '0' -> 0, 'cpu' -> 'cpu', '0,1' -> [0, 1]"""
    if not text:
        return 0
    if text.lower() == "cpu":
        return "cpu"
    if "," in text:
        try:
            return [int(x.strip()) for x in text.split(",") if x.strip()]
        except ValueError:
            log.w(f"無法解析 device: {text}")
            return text
    try:
        return int(text)
    except ValueError:
        return text


class _TrainerThread(QThread):
    """背景執行 YOLO 訓練的 worker thread"""

    progress = pyqtSignal(int, int, str)  # epoch, total_epochs, message
    finished_train = pyqtSignal(bool, str, dict)  # success, msg, info

    def __init__(self, model_info: str, train_kwargs: dict):
        """初始化 trainer thread

        Args:
            model_info: 模型權重檔名 (例如 yolo26s.pt)
            train_kwargs: model.train() 的全部 kwargs
        """
        super().__init__()
        self.model_info = model_info
        self.train_kwargs = train_kwargs
        self._stop = False
        self._save_dir: str = ""

    def stop(self) -> None:
        """請求中止訓練 (在下一個 epoch 結束後生效)"""
        self._stop = True

    def run(self) -> None:
        """執行訓練主流程，透過 ultralytics callback 回報進度"""
        try:
            from ultralytics import YOLO
        except Exception as e:
            log.e(f"ultralytics 匯入失敗: {e}")
            self.finished_train.emit(False, "ultralytics 未安裝", {})
            return

        try:
            model = YOLO(self.model_info)
            start = time.time()

            def on_train_start(trainer):
                self._save_dir = str(getattr(trainer, "save_dir", ""))
                total = int(
                    getattr(trainer, "epochs", self.train_kwargs.get("epochs", 0))
                )
                self.progress.emit(
                    0, total, f"訓練開始，輸出資料夾: {self._save_dir}"
                )

            def on_epoch_end(trainer):
                # 使用者要求停止：設定 ultralytics 內建 stop flag
                if self._stop:
                    try:
                        trainer.stop = True
                    except Exception:
                        pass
                    return
                epoch = int(getattr(trainer, "epoch", 0)) + 1
                total = int(
                    getattr(trainer, "epochs", self.train_kwargs.get("epochs", 0))
                )
                msg = f"Epoch {epoch}/{total}"
                # 嘗試讀取 metrics (epoch end 後的驗證結果)
                try:
                    metrics = getattr(trainer, "metrics", None) or {}
                    map50 = (
                        metrics.get("metrics/mAP50(B)")
                        or metrics.get("metrics/mAP50(M)")
                    )
                    if map50 is not None:
                        msg += f"  mAP50={float(map50):.3f}"
                except Exception as e:
                    log.w(f"讀取 epoch metrics 失敗: {e}")
                self.progress.emit(epoch, total, msg)

            model.add_callback("on_train_start", on_train_start)
            model.add_callback("on_train_epoch_end", on_epoch_end)

            results = model.train(**self.train_kwargs)

            elapsed = timedelta(seconds=int(time.time() - start))
            save_dir = str(getattr(results, "save_dir", self._save_dir))
            info: dict = {"save_dir": save_dir, "elapsed": str(elapsed)}
            box = getattr(results, "box", None)
            if box is not None:
                info["map50"] = float(getattr(box, "map50", 0) or 0)
                info["map"] = float(getattr(box, "map", 0) or 0)
            seg = getattr(results, "seg", None)
            if seg is not None:
                info["seg_map50"] = float(getattr(seg, "map50", 0) or 0)
                info["seg_map"] = float(getattr(seg, "map", 0) or 0)

            msg = "訓練已中止 (使用者停止)" if self._stop else "訓練完成"
            self.finished_train.emit(True, msg, info)
        except Exception as e:
            log.e(f"訓練錯誤: {e}")
            info = {"save_dir": self._save_dir} if self._save_dir else {}
            self.finished_train.emit(False, "訓練失敗，請查看 log", info)


class TrainYoloDialog(QDialog):
    """設定 YOLO 訓練參數並執行 ultralytics 訓練。
    基本參數會持久化到 settings.training，進階參數透過 TrainYoloAdvancedDialog 設定。
    """

    # (size_code, 顯示說明)
    MODEL_SIZES = [
        ("n", "Nano - 最快, 最輕量"),
        ("s", "Small - 預設, 速度與精度兼顧"),
        ("m", "Medium - 平衡選擇"),
        ("l", "Large - 較準較慢"),
        ("x", "Xlarge - 最準最慢"),
    ]
    DEFAULT_VERSION = "yolo26"

    def __init__(self, parent=None, default_folder: str = ""):
        """初始化對話框

        Args:
            parent: 父視窗
            default_folder: 預設資料夾，用於 dataset.yaml 自動搜尋與檔案瀏覽起點
        """
        super().__init__(parent)
        self.setWindowTitle("Train YOLO")
        self.setMinimumWidth(580)
        self._default_folder = default_folder
        self._thread: _TrainerThread | None = None
        self._save_dir: str = ""

        main_layout = QVBoxLayout(self)

        # 全域說明
        hint = QLabel(
            "使用 ultralytics 訓練 YOLO 模型。\n"
            "請先準備好 dataset.yaml (可由 Train → VOC to YOLO 產生)。\n"
            "訓練結果預設儲存在執行目錄下的 runs/<task>/<name>/"
        )
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)
        main_layout.addWidget(hint)

        # === Dataset YAML ===
        ds_group = QGroupBox("Dataset")
        ds_layout = QVBoxLayout()
        ds_row = QHBoxLayout()
        # 預先用 settings 的紀錄、自動搜尋、default_folder 三選一作為初值
        initial_yaml = (
            settings.training.last_data_yaml
            if settings.training.last_data_yaml
            and Path(settings.training.last_data_yaml).is_file()
            else self._autodiscover_yaml(default_folder)
        )
        self.yaml_edit = QLineEdit(initial_yaml)
        self.yaml_edit.setPlaceholderText("dataset.yaml 路徑")
        yaml_browse = QPushButton("瀏覽...")
        yaml_browse.setFixedWidth(80)
        yaml_browse.clicked.connect(self._browse_yaml)
        ds_row.addWidget(self.yaml_edit)
        ds_row.addWidget(yaml_browse)
        ds_layout.addLayout(ds_row)
        ds_hint = QLabel(
            "dataset.yaml 內須定義 train/val 路徑、nc (類別數) 與 names"
        )
        ds_hint.setStyleSheet("color: gray; font-size: 11px;")
        ds_layout.addWidget(ds_hint)
        ds_group.setLayout(ds_layout)
        main_layout.addWidget(ds_group)

        # === Model 設定 ===
        model_group = QGroupBox("Model 設定")
        model_layout = QFormLayout()

        self.task_combo = QComboBox()
        self.task_combo.addItem("Object Detection — bbox 偵測", "detect")
        self.task_combo.addItem("Segmentation — 多邊形分割 (-seg.pt)", "segment")
        self.task_combo.setToolTip(
            "依 dataset.yaml 內標註類型選擇\n"
            "Detect: 一般物件偵測 (bbox)\n"
            "Segment: 多邊形分割 (需要 seg 模型權重)"
        )
        model_layout.addRow("Task:", self.task_combo)

        self.size_combo = QComboBox()
        for size, desc in self.MODEL_SIZES:
            self.size_combo.addItem(f"{size} — {desc}", size)
        self.size_combo.setToolTip("模型規模越大越準確但越慢、越吃 VRAM")
        model_layout.addRow("Model Size:", self.size_combo)

        self.version_edit = QLineEdit()
        self.version_edit.setToolTip(
            "YOLO 版本前綴 (例如 yolo26 / yolov8 / yolo12)\n"
            "會組合成 <version><size>[-seg].pt"
        )
        model_layout.addRow("Version:", self.version_edit)

        self.model_info_label = QLabel()
        self.model_info_label.setStyleSheet("color: gray; font-size: 11px;")
        model_layout.addRow(self.model_info_label)
        # 自動更新顯示的最終模型檔名
        self.task_combo.currentIndexChanged.connect(self._update_model_info_label)
        self.size_combo.currentIndexChanged.connect(self._update_model_info_label)
        self.version_edit.textChanged.connect(self._update_model_info_label)

        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # === 訓練參數 ===
        param_group = QGroupBox("訓練參數")
        param_layout = QFormLayout()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setToolTip(
            "最大訓練輪數，搭配 Patience 提前停止。一般 100~600"
        )
        param_layout.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(-1, 512)
        self.batch_spin.setToolTip(
            "每批次圖片數。VRAM 不足時降低；-1 = 自動偵測最大可用 batch"
        )
        param_layout.addRow("Batch:", self.batch_spin)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(160, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setToolTip(
            "輸入影像解析度 (px)。常見 320 / 640 / 1280，越大越準但越慢"
        )
        param_layout.addRow("Image Size:", self.imgsz_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 1000)
        self.patience_spin.setToolTip(
            "early stopping: 連續 N 個 epoch 無改善則停止；0 = 關閉"
        )
        param_layout.addRow("Patience:", self.patience_spin)

        self.device_edit = QLineEdit()
        self.device_edit.setToolTip(
            "訓練裝置: 0 = 第一張 GPU；cpu = CPU；0,1 = 多 GPU"
        )
        param_layout.addRow("Device:", self.device_edit)

        self.save_period_spin = QSpinBox()
        self.save_period_spin.setRange(-1, 1000)
        self.save_period_spin.setToolTip(
            "每 N 個 epoch 額外存一次 checkpoint；-1 = 關閉。長時間訓練建議開啟"
        )
        param_layout.addRow("Save Period:", self.save_period_spin)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(
            f"預設: train_{datetime.now().strftime('%Y_%m%d_%H%M%S')}"
        )
        self.name_edit.setToolTip(
            "輸出資料夾名稱，結果存於 runs/<task>/<name>/（每次訓練不會持久化）"
        )
        param_layout.addRow("Name:", self.name_edit)

        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)

        # === 訓練狀態 ===
        status_group = QGroupBox("訓練狀態")
        status_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        status_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("尚未開始")
        self.status_label.setStyleSheet("color: gray;")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setVisible(False)
        self.result_text.setMaximumHeight(140)
        status_layout.addWidget(self.result_text)

        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)

        # === 按鈕 ===
        btn_layout = QHBoxLayout()
        self.advanced_btn = QPushButton("進階參數...")
        self.advanced_btn.setToolTip("設定優化器、增強、cache 等詳細訓練參數")
        self.advanced_btn.clicked.connect(self._open_advanced)
        self.start_btn = QPushButton("開始訓練")
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("等待當前 epoch 結束後優雅地停止")
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_folder_btn = QPushButton("開啟訓練資料夾")
        self.open_folder_btn.setToolTip("開啟 runs/<task>/<name>/ 或 runs/<task>/")
        self.open_folder_btn.clicked.connect(self._open_folder)
        self.close_btn = QPushButton("關閉")
        self.close_btn.clicked.connect(self._on_close)
        btn_layout.addWidget(self.advanced_btn)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.open_folder_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.close_btn)
        main_layout.addLayout(btn_layout)

        # 從 settings 載入基本參數初值
        self._load_basic_from_settings()
        self._update_model_info_label()

    # === Helpers ===

    @staticmethod
    def _autodiscover_yaml(folder: str) -> str:
        """搜尋資料夾中最新的 dataset*.yaml 作為預設值"""
        if not folder or not Path(folder).is_dir():
            return ""
        try:
            candidates = sorted(
                Path(folder).glob("dataset*.yaml"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            return str(candidates[0]) if candidates else ""
        except Exception as e:
            log.w(f"搜尋 dataset yaml 失敗: {e}")
            return ""

    def _build_model_info(self) -> str:
        """組合模型權重檔名 (例如 yolo26s.pt 或 yolo26s-seg.pt)"""
        version = self.version_edit.text().strip() or self.DEFAULT_VERSION
        size = self.size_combo.currentData() or "s"
        task = self.task_combo.currentData() or "detect"
        suffix = "-seg" if task == "segment" else ""
        return f"{version}{size}{suffix}.pt"

    def _update_model_info_label(self) -> None:
        """更新顯示最終模型檔名的提示 label"""
        self.model_info_label.setText(
            f"最終使用模型: {self._build_model_info()} "
            f"(若本地不存在，ultralytics 會自動下載)"
        )

    def _browse_yaml(self) -> None:
        """瀏覽選擇 dataset.yaml"""
        start = self.yaml_edit.text() or self._default_folder or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇 dataset.yaml", start, "YAML Files (*.yaml *.yml)"
        )
        if path:
            self.yaml_edit.setText(path)

    def _load_basic_from_settings(self) -> None:
        """從 settings.training 把基本參數值灌到 UI"""
        t = settings.training

        idx = self.task_combo.findData(t.task or "detect")
        self.task_combo.setCurrentIndex(idx if idx >= 0 else 0)

        idx = self.size_combo.findData(t.model_size or "s")
        self.size_combo.setCurrentIndex(idx if idx >= 0 else 1)

        self.version_edit.setText(t.version or self.DEFAULT_VERSION)
        self.epochs_spin.setValue(t.epochs or 100)
        self.batch_spin.setValue(t.batch if t.batch is not None else 16)
        self.imgsz_spin.setValue(t.imgsz or 640)
        self.patience_spin.setValue(t.patience if t.patience is not None else 50)
        self.device_edit.setText(t.device or "0")
        self.save_period_spin.setValue(
            t.save_period if t.save_period is not None else -1
        )

    def _save_basic_to_settings(self) -> None:
        """把基本參數寫回 settings.training (不含 name)"""
        t = settings.training
        t.last_data_yaml = self.yaml_edit.text().strip()
        t.task = self.task_combo.currentData()
        t.model_size = self.size_combo.currentData()
        t.version = self.version_edit.text().strip() or self.DEFAULT_VERSION
        t.epochs = self.epochs_spin.value()
        t.batch = self.batch_spin.value()
        t.imgsz = self.imgsz_spin.value()
        t.patience = self.patience_spin.value()
        t.device = self.device_edit.text().strip() or "0"
        t.save_period = self.save_period_spin.value()

    # === 訓練控制 ===

    def _open_advanced(self) -> None:
        """開啟詳細參數對話框"""
        dialog = TrainYoloAdvancedDialog(self)
        dialog.exec()

    def _on_start(self) -> None:
        """檢查參數並啟動訓練 thread"""
        yaml_path = self.yaml_edit.text().strip()
        if not yaml_path or not Path(yaml_path).is_file():
            QMessageBox.warning(self, "Warning", "請選擇有效的 dataset.yaml")
            return

        # 把基本參數寫回 settings 並持久化
        self._save_basic_to_settings()
        try:
            save_settings()
        except Exception as e:
            log.e(f"儲存 settings 失敗: {e}")

        name = (
            self.name_edit.text().strip()
            or f"train_{datetime.now().strftime('%Y_%m%d_%H%M%S')}"
        )
        train_kwargs = _build_train_kwargs(name)
        model_info = self._build_model_info()

        # UI 狀態切換
        self._set_running(True)
        self.progress_bar.setRange(0, train_kwargs["epochs"])
        self.progress_bar.setValue(0)
        self.status_label.setText(f"準備載入模型: {model_info} ...")
        self.result_text.setVisible(False)
        self.result_text.clear()
        self._save_dir = ""

        self._thread = _TrainerThread(model_info, train_kwargs)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished_train.connect(self._on_finished)
        self._thread.start()

    def _on_progress(self, epoch: int, total: int, message: str) -> None:
        """每個 epoch 完成的進度更新"""
        if epoch == 0:
            self.status_label.setText(message)
            return
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(epoch)
        self.status_label.setText(message)

    def _on_finished(self, success: bool, message: str, info: dict) -> None:
        """訓練完成或失敗的回呼"""
        self._set_running(False)
        save_dir = info.get("save_dir", "")
        if save_dir:
            self._save_dir = save_dir

        self.status_label.setText(message)

        if success:
            lines = [message]
            if save_dir:
                lines.append(f"  輸出資料夾: {save_dir}")
            if "elapsed" in info:
                lines.append(f"  訓練時間: {info['elapsed']}")
            if "map50" in info:
                lines.append(f"  Box mAP@0.5    : {info['map50']:.4f}")
                lines.append(f"  Box mAP@0.5:0.95: {info['map']:.4f}")
            if "seg_map50" in info:
                lines.append(f"  Seg mAP@0.5    : {info['seg_map50']:.4f}")
                lines.append(f"  Seg mAP@0.5:0.95: {info['seg_map']:.4f}")
            self.result_text.setPlainText("\n".join(lines))
            self.result_text.setVisible(True)
        else:
            QMessageBox.warning(
                self, "Warning", "訓練未完成，請查看 console log 取得詳細資訊"
            )

    def _on_stop(self) -> None:
        """請求中止訓練"""
        if self._thread and self._thread.isRunning():
            self.status_label.setText("正在停止訓練 (將在當前 epoch 結束後停止)...")
            self.stop_btn.setEnabled(False)
            self._thread.stop()

    def _open_folder(self) -> None:
        """開啟訓練輸出資料夾 (若尚未開始或無 save_dir 則退回 runs/<task>/)"""
        target = self._save_dir
        if not target or not Path(target).is_dir():
            task = self.task_combo.currentData() or "detect"
            target = str(Path("runs") / task)
            try:
                Path(target).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                log.e(f"建立 runs 資料夾失敗: {e}")
                QMessageBox.warning(self, "Warning", "無法建立或開啟訓練資料夾")
                return
        try:
            os.startfile(target)  # Windows: 用檔案總管開啟
        except Exception as e:
            log.e(f"開啟資料夾失敗: {e}")
            QMessageBox.warning(self, "Warning", f"無法開啟資料夾: {target}")

    def _on_close(self) -> None:
        """關閉前若仍在訓練則確認"""
        if self._thread and self._thread.isRunning():
            reply = QMessageBox.question(
                self,
                "確認",
                "訓練尚未結束，確定要中止並關閉?\n"
                "(目前 epoch 結束後才會真正停止)",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._thread.stop()
            self._thread.wait(2000)
        self.accept()

    def _set_running(self, running: bool) -> None:
        """切換 UI 為訓練中/閒置狀態"""
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.advanced_btn.setEnabled(not running)
        self.yaml_edit.setEnabled(not running)
        self.task_combo.setEnabled(not running)
        self.size_combo.setEnabled(not running)
        self.version_edit.setEnabled(not running)
        self.epochs_spin.setEnabled(not running)
        self.batch_spin.setEnabled(not running)
        self.imgsz_spin.setEnabled(not running)
        self.patience_spin.setEnabled(not running)
        self.device_edit.setEnabled(not running)
        self.save_period_spin.setEnabled(not running)
        self.name_edit.setEnabled(not running)
