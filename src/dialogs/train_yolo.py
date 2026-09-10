# Train YOLO 對話框：選擇 dataset.yaml、設定訓練參數、執行 ultralytics 訓練並顯示進度與結果
# 支援指定既有 .pt 來再訓練（fine-tune）或從中斷處續訓（resume）
# 更新日期: 2026-09-10
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
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
from ruamel.yaml import YAML

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


def _build_train_kwargs(name: str, resume: bool = False) -> dict:
    """依 settings.training 組合 ultralytics model.train() 所需的 kwargs

    Args:
        name: 輸出資料夾名稱（runs/<task>/<name>）
        resume: True 時加上 resume=True，從原訓練的 epoch / optimizer / scheduler 接續
    """
    t = settings.training
    task = t.task or "detect"
    # 強制 project 指向工作目錄下的 runs/<task>，避免 ultralytics 全域 settings.json
    # 把 runs_dir 記在其他磁碟造成輸出位置跑掉。
    # 這裡的 task 只是預設值: 指定 .pt 再訓練時真正的 task 在權重裡, 由
    # _TrainerThread._align_task() 載入模型後修正。
    project = str(Path.cwd() / "runs" / task)
    epochs = t.epochs or 500
    # close_mosaic >= epochs 時 ultralytics 會在第 0 個 epoch 就關掉 mosaic
    # (trainer 的判斷是 epoch == epochs - close_mosaic), 等於整場都沒有 mosaic 增強,
    # 而且不會有任何警告。夾成 epochs-1, 至少保留一輪。
    close_mosaic = min(t.close_mosaic or 0, max(0, epochs - 1))
    if close_mosaic != (t.close_mosaic or 0):
        log.w(f"close_mosaic ({t.close_mosaic}) 不可 >= epochs ({epochs}), 已夾成 {close_mosaic}")
    kwargs = {
        # 資料與輸出
        "data": t.last_data_yaml,
        "project": project,
        "name": name,
        "exist_ok": False,
        "plots": True,
        # 訓練核心
        "epochs": epochs,
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
        "close_mosaic": close_mosaic,
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
    if resume:
        # ultralytics resume 會讀取 last.pt 旁邊的 args.yaml 接續訓練, 並且整份
        # self.args 都以 args.yaml 為準; 只有 BaseTrainer.check_resume() 白名單內的
        # 參數 (imgsz / batch / device / close_mosaic / save_period / workers /
        # cache / patience / val / plots / freeze ...) 允許覆寫。
        # epochs 與 optimizer / lr / 增強參數都不在白名單, 這裡傳了也會被丟掉。
        kwargs["resume"] = True
    return kwargs


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


def _detect_dataset_label_type(yaml_path: str) -> str | None:
    """讀 dataset.yaml 的 train 標籤，判斷這份 dataset 是 bbox 還是 seg 格式

    YOLO 標籤一行 5 欄是 bbox (class cx cy w h), 更多欄則是 polygon。同一份
    dataset 不會混格式, 取第一筆有內容的即可; 空標籤 (背景圖) 會跳過。

    Args:
        yaml_path: dataset.yaml 路徑

    Returns:
        "bbox" / "seg"; 讀不到或全是空標籤時回 None (呼叫端就不做檢查)
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = YAML().load(f) or {}
        train = data.get("train") or "images/train"
        if isinstance(train, (list, tuple)):
            train = train[0]
        img_dir = Path(train)
        if not img_dir.is_absolute():
            img_dir = Path(data.get("path") or Path(yaml_path).parent) / img_dir

        # 與 ultralytics 的 img2label_paths 同規則: 換掉路徑中最後一個 images
        parts = list(img_dir.parts)
        if "images" in parts:
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        lbl_dir = Path(*parts)
        if not lbl_dir.is_dir():
            return None

        for txt in sorted(lbl_dir.glob("*.txt"))[:50]:
            for line in txt.read_text(encoding="utf-8").splitlines():
                cols = line.split()
                if len(cols) >= 5:
                    return "bbox" if len(cols) == 5 else "seg"
        return None
    except Exception as e:
        log.w(f"判斷 dataset 標籤格式失敗 ({yaml_path}): {e}")
        return None


class _TrainerThread(QThread):
    """背景執行 YOLO 訓練的 worker thread"""

    progress = pyqtSignal(int, int, str)  # epoch, total_epochs, message
    finished_train = pyqtSignal(bool, str, dict)  # success, msg, info

    def __init__(
        self, model_info: str, train_kwargs: dict, label_type: str | None = None
    ):
        """初始化 trainer thread

        Args:
            model_info: 模型權重檔名 (例如 yolo26s.pt)
            train_kwargs: model.train() 的全部 kwargs
            label_type: dataset 標籤格式 ("bbox" / "seg"), 用來與模型 task 交叉比對
        """
        super().__init__()
        self.model_info = model_info
        self.train_kwargs = train_kwargs
        self.label_type = label_type
        self._stop = False
        self._save_dir: str = ""

    def stop(self) -> None:
        """請求中止訓練 (在下一個 epoch 結束後生效)"""
        self._stop = True

    def _align_task(self, model) -> None:
        """以實際載入的模型 task 修正輸出目錄，並比對 dataset 標籤格式

        task 的真值在權重裡: 指定 .pt 再訓練時, 對話框的 Task 欄位是鎖住的舊值,
        拿它決定 runs/<task>/ 會把 seg 的訓練結果丟進 runs/detect/。

        Args:
            model: 已載入的 YOLO 物件
        """
        task = getattr(model, "task", None)
        if not task:
            return
        project = Path(self.train_kwargs.get("project", ""))
        if project.name and project.name != task:
            self.train_kwargs["project"] = str(project.parent / task)
            log.i(f"依模型實際 task 修正輸出目錄: {project} -> {self.train_kwargs['project']}")

        # 標籤格式與 task 不符只警告不擋: detect + polygon 標籤 ultralytics 會自動
        # 取外接框照跑 (不報錯), 使用者未必知道自己訓出來的並不是 seg 模型
        expect = {"detect": "bbox", "segment": "seg"}.get(task)
        if self.label_type and expect and self.label_type != expect:
            warn = f"注意: dataset 標籤是 {self.label_type} 格式, 但模型 task 是 {task}"
            log.w(warn)
            self.progress.emit(0, 0, warn)

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
            self._align_task(model)
            start = time.time()

            def on_train_start(trainer):
                self._save_dir = str(getattr(trainer, "save_dir", ""))
                total = int(
                    getattr(trainer, "epochs", self.train_kwargs.get("epochs", 0))
                )
                self.progress.emit(
                    0, total, f"訓練開始，輸出資料夾: {self._save_dir}"
                )

            def on_train_epoch_end(trainer):
                """把使用者的停止請求交給 ultralytics 內建的 stop flag

                trainer 之後是 `self.stop |= ...`, 所以這裡設 True 不會被蓋掉,
                會在當前 epoch 驗證完後跳出訓練迴圈。
                """
                if not self._stop:
                    return
                try:
                    trainer.stop = True
                except Exception as e:
                    log.w(f"設定 trainer.stop 失敗: {e}")

            def on_fit_epoch_end(trainer):
                """回報 epoch 進度與 mAP

                掛 fit 而不是 train_epoch_end: ultralytics 的順序是
                on_train_epoch_end -> validate() -> on_fit_epoch_end, 掛在前者拿到的
                trainer.metrics 還是上一輪的值, 顯示的 mAP 會整整慢一個 epoch。
                """
                total = int(
                    getattr(trainer, "epochs", self.train_kwargs.get("epochs", 0))
                )
                # 訓練結束後的 final_eval 會把 epoch +1 再觸發一次這個 callback,
                # 夾住才不會顯示成 "Epoch 501/500"
                epoch = min(int(getattr(trainer, "epoch", 0)) + 1, total or 1)
                msg = f"Epoch {epoch}/{total}"
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
            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

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
            err_msg = str(e)
            # ultralytics 對「已完成的 last.pt 又被 resume」會丟這個訊息，
            # 把原文跟解法一起回給 UI。
            if "nothing to resume" in err_msg.lower():
                hint = (
                    "原訓練已達設定的最大 epoch 數，無法 Resume。\n"
                    "請取消勾選『Resume mode』改用 Fine-tune 模式，"
                    "並把 Epochs 設成你想要的新 epoch 數重新訓練。"
                )
                info["hint"] = hint
            info["error"] = err_msg
            self.finished_train.emit(False, f"訓練失敗: {err_msg}", info)


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
        # 使用者在訓練中按了關閉: 等 thread 收工後才真的關視窗
        self._close_after_stop = False

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
        model_layout = QVBoxLayout()

        # --- 從現有 .pt 接續訓練（選填）---
        resume_row = QHBoxLayout()
        self.resume_pt_edit = QLineEdit()
        self.resume_pt_edit.setPlaceholderText(
            "選填：選擇之前訓練的 last.pt / best.pt 來再訓練 (留空=從預訓練模型開始)"
        )
        self.resume_pt_edit.setToolTip(
            "指定 .pt 來接續訓練：\n"
            "• 留空：依下方 Task / Model Size / Version 組合預訓練模型 (e.g. yolo26s.pt)\n"
            "• 填入 best.pt：以該權重做新一輪訓練 (fine-tune，會建立新的 runs/ 資料夾)\n"
            "• 填入 last.pt 並勾選『Resume』：從原訓練中斷處繼續，沿用原 epoch 與 optimizer 狀態"
        )
        resume_browse = QPushButton("瀏覽...")
        resume_browse.setFixedWidth(80)
        resume_browse.clicked.connect(self._browse_resume_pt)
        resume_clear = QPushButton("清除")
        resume_clear.setFixedWidth(60)
        resume_clear.clicked.connect(lambda: self.resume_pt_edit.setText(""))
        resume_row.addWidget(QLabel("Resume from .pt:"))
        resume_row.addWidget(self.resume_pt_edit)
        resume_row.addWidget(resume_browse)
        resume_row.addWidget(resume_clear)
        model_layout.addLayout(resume_row)

        self.resume_check = QCheckBox(
            "Resume mode：從原訓練的 epoch 接續 (需指向 last.pt，沿用原 optimizer/scheduler/dataset)"
        )
        self.resume_check.setToolTip(
            "勾選 = ultralytics resume=True：\n"
            "  - 從原訓練中斷的 epoch 繼續，optimizer/scheduler 狀態保留\n"
            "  - .pt 旁需有 args.yaml (ultralytics 自動產出)，dataset 結構不可變動\n"
            "未勾選且填了 .pt = fine-tune：\n"
            "  - 以該權重為起點，使用此對話框的所有參數做新一輪訓練"
        )
        model_layout.addWidget(self.resume_check)

        # Resume 時哪些欄位其實無效, 直接寫在 UI 上 (欄位本身也會被鎖住)
        self.resume_hint = QLabel(
            "Resume 模式：Epochs / Name / 進階參數 (optimizer、lr、增強) 全部沿用原訓練的 "
            "args.yaml，改了也不會生效，因此已鎖住。\n"
            "只有 Batch / Image Size / Device / Patience / Save Period 可以覆寫；"
            "要改輪數或增強請取消 Resume 改用 Fine-tune。"
        )
        self.resume_hint.setStyleSheet("color: #b8860b; font-size: 11px;")
        self.resume_hint.setWordWrap(True)
        self.resume_hint.setVisible(False)
        model_layout.addWidget(self.resume_hint)

        self.resume_pt_edit.textChanged.connect(self._update_resume_state)
        self.resume_check.toggled.connect(self._update_resume_state)

        # --- 預訓練模型組合（resume_pt_edit 為空時使用）---
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        model_layout.addWidget(sep)

        base_form = QFormLayout()

        self.task_combo = QComboBox()
        self.task_combo.addItem("Object Detection — bbox 偵測", "detect")
        self.task_combo.addItem("Segmentation — 多邊形分割 (-seg.pt)", "segment")
        self.task_combo.setToolTip(
            "依 dataset.yaml 內標註類型選擇\n"
            "Detect: 一般物件偵測 (bbox)\n"
            "Segment: 多邊形分割 (需要 seg 模型權重)"
        )
        base_form.addRow("Task:", self.task_combo)

        self.size_combo = QComboBox()
        for size, desc in self.MODEL_SIZES:
            self.size_combo.addItem(f"{size} — {desc}", size)
        self.size_combo.setToolTip("模型規模越大越準確但越慢、越吃 VRAM")
        base_form.addRow("Model Size:", self.size_combo)

        self.version_edit = QLineEdit()
        self.version_edit.setToolTip(
            "YOLO 版本前綴 (例如 yolo26 / yolov8 / yolo12)\n"
            "會組合成 <version><size>[-seg].pt"
        )
        base_form.addRow("Version:", self.version_edit)

        self.model_info_label = QLabel()
        self.model_info_label.setStyleSheet("color: gray; font-size: 11px;")
        base_form.addRow(self.model_info_label)
        # 自動更新顯示的最終模型檔名
        self.task_combo.currentIndexChanged.connect(self._update_model_info_label)
        self.size_combo.currentIndexChanged.connect(self._update_model_info_label)
        self.version_edit.textChanged.connect(self._update_model_info_label)

        model_layout.addLayout(base_form)
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # === 訓練參數 ===
        param_group = QGroupBox("訓練參數")
        param_layout = QFormLayout()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setToolTip(
            "最大訓練輪數 (預設 500)。\n"
            "這是「上限」不是「一定要跑完」: 搭配 Patience，連續 N 輪 mAP 沒進步就會自動early stop，\n"
            "所以設大一點只是留餘裕，不會白跑。\n"
            "• 100 以下: 資料量少時通常還沒收斂，是精度不佳最常見的原因\n"
            "• 300~600: 一般建議範圍\n"
            "• 資料集越小 / 類別越多，需要的輪數越多\n"
            "註: Resume 模式下此欄無效，輪數沿用原訓練的 args.yaml"
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
        self._update_resume_state()

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
        """組合最終要載入的模型路徑/名稱

        - 若有指定 resume .pt 路徑且檔案存在，回傳該路徑
        - 否則用 version+size+task 組合預訓練模型檔名 (例如 yolo26s.pt 或 yolo26s-seg.pt)
        """
        resume_pt = self.resume_pt_edit.text().strip()
        if resume_pt and Path(resume_pt).is_file():
            return resume_pt
        version = self.version_edit.text().strip() or self.DEFAULT_VERSION
        size = self.size_combo.currentData() or "s"
        task = self.task_combo.currentData() or "detect"
        suffix = "-seg" if task == "segment" else ""
        return f"{version}{size}{suffix}.pt"

    def _update_model_info_label(self) -> None:
        """更新顯示最終模型檔名的提示 label"""
        resume_pt = self.resume_pt_edit.text().strip()
        if resume_pt:
            mode = "Resume (從原 epoch 接續)" if self.resume_check.isChecked() else "Fine-tune (以此權重開新訓練)"
            self.model_info_label.setText(
                f"最終使用模型: {resume_pt}\n"
                f"模式: {mode}"
            )
        else:
            self.model_info_label.setText(
                f"最終使用模型: {self._build_model_info()} "
                f"(若本地不存在，ultralytics 會自動下載)"
            )

    def _update_resume_state(self) -> None:
        """根據 resume_pt_edit 是否有值，切換預訓練組合欄位的可用性，並刷新提示 label"""
        has_resume = bool(self.resume_pt_edit.text().strip())
        # 有指定 resume .pt 時，version/size/task 由該 .pt 決定，下方欄位 disable
        self.task_combo.setEnabled(not has_resume)
        self.size_combo.setEnabled(not has_resume)
        self.version_edit.setEnabled(not has_resume)
        # resume mode checkbox 只在有指定 .pt 時有意義
        self.resume_check.setEnabled(has_resume)
        if not has_resume:
            self.resume_check.setChecked(False)

        # Resume 時 ultralytics 的 check_resume() 會整份沿用 last.pt 旁的 args.yaml,
        # 只有白名單 (imgsz / batch / device / patience / save_period / workers /
        # cache / close_mosaic / freeze / val / plots) 能覆寫。epochs、name 與所有
        # optimizer / lr / 增強參數傳了也會被丟掉, 因此鎖起來避免誤會。
        resume_mode = has_resume and self.resume_check.isChecked()
        self.epochs_spin.setEnabled(not resume_mode)
        self.name_edit.setEnabled(not resume_mode)
        self.advanced_btn.setEnabled(not resume_mode)
        self.resume_hint.setVisible(resume_mode)
        self._update_model_info_label()

    def _browse_resume_pt(self) -> None:
        """瀏覽選擇要接續訓練的 .pt 檔"""
        # 預設從工作目錄下的 runs/ 開始找，找不到再退回 default_folder
        runs_dir = Path.cwd() / "runs"
        start = self.resume_pt_edit.text().strip()
        if not start:
            start = str(runs_dir) if runs_dir.is_dir() else (self._default_folder or "")
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇要接續訓練的 .pt 檔", start, "PyTorch Weights (*.pt)"
        )
        if path:
            self.resume_pt_edit.setText(path)

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
        self.epochs_spin.setValue(t.epochs or 500)
        self.batch_spin.setValue(t.batch if t.batch is not None else 16)
        self.imgsz_spin.setValue(t.imgsz or 640)
        self.patience_spin.setValue(t.patience if t.patience is not None else 50)
        self.device_edit.setText(t.device or "0")
        self.save_period_spin.setValue(
            t.save_period if t.save_period is not None else -1
        )
        # 再訓練設定：只在 .pt 路徑仍存在時還原，避免 UI 帶到一個失效的舊路徑
        prev_pt = (t.resume_pt_path or "").strip()
        if prev_pt and Path(prev_pt).is_file():
            self.resume_pt_edit.setText(prev_pt)
            self.resume_check.setChecked(bool(t.resume_mode))
        else:
            self.resume_pt_edit.setText("")
            self.resume_check.setChecked(False)

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
        t.resume_pt_path = self.resume_pt_edit.text().strip()
        t.resume_mode = bool(self.resume_check.isChecked() and t.resume_pt_path)

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

        # batch 沒有 0 這個合法值 (-1=自動偵測, 其餘要 >= 1)。ultralytics 不做驗證,
        # 0 會一路傳到 DataLoader 才丟出看不懂的 batch_size 錯誤
        if self.batch_spin.value() == 0:
            QMessageBox.warning(
                self, "Warning", "Batch 不可為 0，請填 -1 (自動) 或 1 以上"
            )
            return

        # 再訓練 .pt 檢查
        resume_pt = self.resume_pt_edit.text().strip()
        if resume_pt and not Path(resume_pt).is_file():
            QMessageBox.warning(
                self, "Warning", f"找不到指定的 .pt 檔: {resume_pt}"
            )
            return
        resume_mode = bool(resume_pt) and self.resume_check.isChecked()
        if resume_mode and Path(resume_pt).name != "last.pt":
            reply = QMessageBox.question(
                self,
                "確認 Resume",
                "Resume 模式建議使用 last.pt（旁邊需有 args.yaml 才能正確接續）。\n"
                f"目前選的是: {Path(resume_pt).name}\n是否仍要繼續?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Task 與 dataset 實際標籤格式的交叉檢查。指定 .pt 時 task 由權重決定 (這裡
        # 讀不到, 不做無謂的確認), 改由 _TrainerThread._align_task() 載入後警告
        label_type = _detect_dataset_label_type(yaml_path)
        if not resume_pt and label_type:
            expect = "seg" if self.task_combo.currentData() == "segment" else "bbox"
            if label_type != expect:
                reply = QMessageBox.question(
                    self,
                    "Task 與標籤格式不符",
                    f"dataset 的標籤是 {label_type} 格式，但 Task 選的是 "
                    f"{self.task_combo.currentText()}。\n\n"
                    "• seg 標籤 + Detect：ultralytics 會自動取外接框訓練，不會報錯，"
                    "但訓出來的是偵測模型\n"
                    "• bbox 標籤 + Segment：訓練會直接失敗\n\n"
                    "是否仍要繼續?",
                )
                if reply != QMessageBox.StandardButton.Yes:
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
        train_kwargs = _build_train_kwargs(name, resume=resume_mode)
        model_info = self._build_model_info()

        # UI 狀態切換
        self._set_running(True)
        self.progress_bar.setRange(0, train_kwargs["epochs"])
        self.progress_bar.setValue(0)
        self.status_label.setText(f"準備載入模型: {model_info} ...")
        self.result_text.setVisible(False)
        self.result_text.clear()
        self._save_dir = ""

        # 在主執行緒先 import ultralytics: 若讓 _TrainerThread 子執行緒首次 import 這類
        # 重型原生套件, 在 Windows 會觸發 native crash / 程式無聲跳出 (與 detect 同一個坑)。
        # 已 import 過則為即時 cache 命中, 不影響效能。
        QApplication.processEvents()  # 先把上面的狀態訊息畫出來, import 可能短暫凍結 UI
        try:
            import ultralytics  # noqa: F401
        except Exception as e:
            log.e(f"ultralytics 匯入失敗: {e}")
            QMessageBox.warning(self, "Warning", "ultralytics 未安裝或匯入失敗")
            self._set_running(False)
            return

        self._thread = _TrainerThread(model_info, train_kwargs, label_type)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished_train.connect(self._on_finished)
        self._thread.start()

    def _on_progress(self, epoch: int, total: int, message: str) -> None:
        """每個 epoch 完成的進度更新

        Args:
            epoch: 目前輪數; 0 代表訓練開始前的訊息 (輸出資料夾、格式警告等)
            total: 總輪數
            message: 要顯示的訊息
        """
        if epoch == 0:
            # 開跑前的訊息也留一份在結果框: 只寫 status_label 的話, 下一則訊息
            # (或第一個 epoch) 一進來就被蓋掉, 標籤格式警告等於沒看到
            self.status_label.setText(message)
            self.result_text.append(message)
            self.result_text.setVisible(True)
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
            err = info.get("error") or "請查看 console log 取得詳細資訊"
            hint = info.get("hint")
            text = f"訓練失敗：\n{err}"
            if hint:
                text += f"\n\n建議：\n{hint}"
            QMessageBox.warning(self, "Warning", text)

        # 訓練中按了關閉: thread 已收工, 這裡才真的關視窗
        if self._close_after_stop:
            # run() 正在返回途中, 這個 wait 幾乎立即結束。真的沒等到就不關,
            # 免得 QThread 在執行中被銷毀 (Qt 會直接 abort 行程)
            if self._thread and not self._thread.wait(5000):
                log.w("訓練 thread 未在時限內結束, 暫不關閉視窗")
                self._close_after_stop = False
                self.status_label.setText("訓練 thread 尚未結束，請稍後再關閉視窗")
                return
            self.accept()

    def _on_stop(self) -> None:
        """請求中止訓練"""
        if self._thread and self._thread.isRunning():
            tail = "，停止後會自動關閉視窗" if self._close_after_stop else ""
            self.status_label.setText(
                f"正在停止訓練 (將在當前 epoch 結束後停止){tail}..."
            )
            self.stop_btn.setEnabled(False)
            self._thread.stop()

    def _open_folder(self) -> None:
        """開啟訓練輸出資料夾 (若尚未開始或無 save_dir 則退回 cwd/runs/<task>/)"""
        target = self._save_dir
        if not target or not Path(target).is_dir():
            task = self.task_combo.currentData() or "detect"
            target = str(Path.cwd() / "runs" / task)
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
        """關閉；訓練中則先請求中止，等 thread 真的結束才關掉視窗

        不能只 wait 個兩秒就放行: 停止旗標要等當前 epoch 跑完才生效, 一個 epoch
        動輒好幾分鐘。視窗先關掉的話訓練會在背景繼續、而且沒有 UI 可以再停它,
        最後關主視窗時 QThread 還在跑就被銷毀, Qt 會直接 abort 整個行程。
        """
        if self._thread and self._thread.isRunning():
            reply = QMessageBox.question(
                self,
                "確認",
                "訓練尚未結束，要中止嗎?\n"
                "(會等目前 epoch 結束才真正停止，停止完成後視窗才會關閉)",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._close_after_stop = True
            self._on_stop()
            return
        self.accept()

    def closeEvent(self, event) -> None:
        """視窗右上角的關閉鈕走與「關閉」按鈕相同的流程

        Args:
            event: Qt 的關閉事件; 訓練中一律 ignore, 改由 _on_close 決定
        """
        if self._thread and self._thread.isRunning():
            event.ignore()
            self._on_close()
            return
        event.accept()

    def _set_running(self, running: bool) -> None:
        """切換 UI 為訓練中/閒置狀態"""
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.advanced_btn.setEnabled(not running)
        self.yaml_edit.setEnabled(not running)
        self.resume_pt_edit.setEnabled(not running)
        self.epochs_spin.setEnabled(not running)
        self.batch_spin.setEnabled(not running)
        self.imgsz_spin.setEnabled(not running)
        self.patience_spin.setEnabled(not running)
        self.device_edit.setEnabled(not running)
        self.save_period_spin.setEnabled(not running)
        self.name_edit.setEnabled(not running)
        if running:
            # 訓練中所有 model 區塊都鎖住
            self.task_combo.setEnabled(False)
            self.size_combo.setEnabled(False)
            self.version_edit.setEnabled(False)
            self.resume_check.setEnabled(False)
        else:
            # 閒置時依 resume 是否有值決定哪些欄位可用
            self._update_resume_state()
