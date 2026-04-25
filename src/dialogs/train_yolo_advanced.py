# Train YOLO Advanced 對話框：詳細訓練參數設定（優化器、增強、系統等），暫存至 settings.training
# 更新日期: 2026-04-25
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.utils.dynamic_settings import TrainingSettings, save_settings, settings
from src.utils.logger import getUniqueLogger

log = getUniqueLogger(__file__)


class TrainYoloAdvancedDialog(QDialog):
    """詳細訓練參數設定 (對應 ultralytics model.train() 的進階參數)"""

    OPTIMIZERS = ["auto", "SGD", "Adam", "AdamW", "RMSProp"]
    CACHE_MODES = [
        ("False — 每次從磁碟讀", "false"),
        ("ram — 快取到 RAM", "ram"),
        ("disk — 快取到硬碟", "disk"),
    ]

    def __init__(self, parent=None):
        """初始化詳細參數對話框，從 settings.training 載入既有值"""
        super().__init__(parent)
        self.setWindowTitle("Train YOLO — 詳細參數")
        self.setMinimumWidth(520)
        self.setMinimumHeight(440)

        main_layout = QVBoxLayout(self)

        hint = QLabel(
            "進階參數會直接傳給 ultralytics model.train()。\n"
            "若不確定意義，建議保留預設值。所有設定會儲存至 cfg/settings.yaml"
        )
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)
        main_layout.addWidget(hint)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self._build_optimizer_tab()
        self._build_geom_flip_tab()
        self._build_color_tab()
        self._build_mix_tab()
        self._build_system_tab()

        # === 按鈕 ===
        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("重設預設值")
        reset_btn.setToolTip("將所有進階參數恢復成程式預設")
        reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()
        save_btn = QPushButton("確定")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        main_layout.addLayout(btn_layout)

        self._load_from_settings()

    # === Tab builders ===

    def _build_optimizer_tab(self) -> None:
        """優化器與學習率相關參數"""
        widget = QWidget()
        form = QFormLayout(widget)

        self.optimizer_combo = QComboBox()
        for o in self.OPTIMIZERS:
            self.optimizer_combo.addItem(o, o)
        self.optimizer_combo.setToolTip(
            "auto = 由 ultralytics 自動選擇 (預設 SGD)\n"
            "AdamW 對複雜場景更穩定，但需搭配低 lr0"
        )
        form.addRow("Optimizer:", self.optimizer_combo)

        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setRange(1e-6, 1.0)
        self.lr0_spin.setDecimals(5)
        self.lr0_spin.setSingleStep(0.001)
        self.lr0_spin.setToolTip("初始學習率，default=0.01；AdamW 建議 0.001~0.002")
        form.addRow("lr0 (初始學習率):", self.lr0_spin)

        self.lrf_spin = QDoubleSpinBox()
        self.lrf_spin.setRange(0.0001, 1.0)
        self.lrf_spin.setDecimals(4)
        self.lrf_spin.setSingleStep(0.001)
        self.lrf_spin.setToolTip("最終學習率 = lr0 × lrf, default=0.01")
        form.addRow("lrf (lr 衰減比):", self.lrf_spin)

        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.1)
        self.weight_decay_spin.setDecimals(5)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setToolTip(
            "L2 正則化, default=0.0005；AdamW 建議 0.01~0.05"
        )
        form.addRow("weight_decay:", self.weight_decay_spin)

        self.warmup_epochs_spin = QDoubleSpinBox()
        self.warmup_epochs_spin.setRange(0.0, 50.0)
        self.warmup_epochs_spin.setDecimals(1)
        self.warmup_epochs_spin.setSingleStep(0.5)
        self.warmup_epochs_spin.setToolTip("前 N epoch 線性暖機, default=3.0")
        form.addRow("warmup_epochs:", self.warmup_epochs_spin)

        self.warmup_momentum_spin = QDoubleSpinBox()
        self.warmup_momentum_spin.setRange(0.0, 1.0)
        self.warmup_momentum_spin.setDecimals(2)
        self.warmup_momentum_spin.setSingleStep(0.05)
        self.warmup_momentum_spin.setToolTip("暖機期間起始 momentum, default=0.8")
        form.addRow("warmup_momentum:", self.warmup_momentum_spin)

        self.tabs.addTab(widget, "優化器")

    def _build_geom_flip_tab(self) -> None:
        """幾何增強與翻轉"""
        widget = QWidget()
        form = QFormLayout(widget)

        self.degrees_spin = QDoubleSpinBox()
        self.degrees_spin.setRange(0.0, 180.0)
        self.degrees_spin.setDecimals(1)
        self.degrees_spin.setSingleStep(1.0)
        self.degrees_spin.setToolTip("隨機旋轉角度範圍 ±N°, default=0.0")
        form.addRow("degrees (旋轉°):", self.degrees_spin)

        self.translate_spin = QDoubleSpinBox()
        self.translate_spin.setRange(0.0, 1.0)
        self.translate_spin.setDecimals(2)
        self.translate_spin.setSingleStep(0.05)
        self.translate_spin.setToolTip("隨機平移比例 0.0~1.0, default=0.1")
        form.addRow("translate:", self.translate_spin)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 1.5)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setToolTip("隨機縮放比例 ±N, default=0.5")
        form.addRow("scale:", self.scale_spin)

        self.perspective_spin = QDoubleSpinBox()
        self.perspective_spin.setRange(0.0, 0.001)
        self.perspective_spin.setDecimals(5)
        self.perspective_spin.setSingleStep(0.0001)
        self.perspective_spin.setToolTip(
            "透視變換強度, default=0.0；極小值即可，太大會扭曲標註"
        )
        form.addRow("perspective:", self.perspective_spin)

        self.flipud_spin = QDoubleSpinBox()
        self.flipud_spin.setRange(0.0, 1.0)
        self.flipud_spin.setDecimals(2)
        self.flipud_spin.setSingleStep(0.05)
        self.flipud_spin.setToolTip("上下翻轉機率, default=0.0")
        form.addRow("flipud (上下翻轉):", self.flipud_spin)

        self.fliplr_spin = QDoubleSpinBox()
        self.fliplr_spin.setRange(0.0, 1.0)
        self.fliplr_spin.setDecimals(2)
        self.fliplr_spin.setSingleStep(0.05)
        self.fliplr_spin.setToolTip("左右翻轉機率, default=0.5")
        form.addRow("fliplr (左右翻轉):", self.fliplr_spin)

        self.tabs.addTab(widget, "幾何 / 翻轉")

    def _build_color_tab(self) -> None:
        """色彩 (HSV) 增強"""
        widget = QWidget()
        form = QFormLayout(widget)

        hint = QLabel("HSV 色彩抖動 — 對灰階/紅外線影像也有效")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        form.addRow(hint)

        self.hsv_h_spin = QDoubleSpinBox()
        self.hsv_h_spin.setRange(0.0, 1.0)
        self.hsv_h_spin.setDecimals(3)
        self.hsv_h_spin.setSingleStep(0.005)
        self.hsv_h_spin.setToolTip("色相偏移範圍, default=0.015")
        form.addRow("hsv_h (色相):", self.hsv_h_spin)

        self.hsv_s_spin = QDoubleSpinBox()
        self.hsv_s_spin.setRange(0.0, 1.0)
        self.hsv_s_spin.setDecimals(2)
        self.hsv_s_spin.setSingleStep(0.05)
        self.hsv_s_spin.setToolTip("飽和度變化範圍, default=0.7")
        form.addRow("hsv_s (飽和度):", self.hsv_s_spin)

        self.hsv_v_spin = QDoubleSpinBox()
        self.hsv_v_spin.setRange(0.0, 1.0)
        self.hsv_v_spin.setDecimals(2)
        self.hsv_v_spin.setSingleStep(0.05)
        self.hsv_v_spin.setToolTip("亮度變化範圍, default=0.4")
        form.addRow("hsv_v (亮度):", self.hsv_v_spin)

        self.tabs.addTab(widget, "色彩")

    def _build_mix_tab(self) -> None:
        """混合增強 (Mosaic / MixUp / Copy-Paste)"""
        widget = QWidget()
        form = QFormLayout(widget)

        self.mosaic_spin = QDoubleSpinBox()
        self.mosaic_spin.setRange(0.0, 1.0)
        self.mosaic_spin.setDecimals(2)
        self.mosaic_spin.setSingleStep(0.05)
        self.mosaic_spin.setToolTip(
            "馬賽克拼接機率, default=1.0\n"
            "4 張圖拼成 1 張，增加小物件多樣性"
        )
        form.addRow("mosaic:", self.mosaic_spin)

        self.close_mosaic_spin = QSpinBox()
        self.close_mosaic_spin.setRange(0, 1000)
        self.close_mosaic_spin.setToolTip(
            "最後 N 個 epoch 關閉 mosaic, default=10\n讓模型最後學完整圖"
        )
        form.addRow("close_mosaic:", self.close_mosaic_spin)

        self.mixup_spin = QDoubleSpinBox()
        self.mixup_spin.setRange(0.0, 1.0)
        self.mixup_spin.setDecimals(2)
        self.mixup_spin.setSingleStep(0.05)
        self.mixup_spin.setToolTip(
            "MixUp 機率, default=0.0\n兩張圖疊加混合，太高會模糊特徵"
        )
        form.addRow("mixup:", self.mixup_spin)

        self.copy_paste_spin = QDoubleSpinBox()
        self.copy_paste_spin.setRange(0.0, 1.0)
        self.copy_paste_spin.setDecimals(2)
        self.copy_paste_spin.setSingleStep(0.05)
        self.copy_paste_spin.setToolTip(
            "Copy-Paste 機率, default=0.0\n複製物件貼到其他圖，太高會讓 loss 降不下"
        )
        form.addRow("copy_paste:", self.copy_paste_spin)

        self.tabs.addTab(widget, "混合增強")

    def _build_system_tab(self) -> None:
        """DataLoader、cache、AMP、freeze 等系統參數"""
        widget = QWidget()
        form = QFormLayout(widget)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setToolTip(
            "DataLoader worker 數, default=8\nWindows 上若有問題可設為 0"
        )
        form.addRow("workers:", self.workers_spin)

        self.cache_combo = QComboBox()
        for label, value in self.CACHE_MODES:
            self.cache_combo.addItem(label, value)
        self.cache_combo.setToolTip(
            "RAM 最快但很吃記憶體；disk 適合資料集大但 RAM 不夠用"
        )
        form.addRow("cache:", self.cache_combo)

        self.rect_check = QCheckBox()
        self.rect_check.setToolTip(
            "矩形訓練 (非正方形): 減少 padding 加速，但可能降低精度"
        )
        form.addRow("rect:", self.rect_check)

        self.amp_check = QCheckBox()
        self.amp_check.setToolTip(
            "混合精度訓練, default=True\n減少 VRAM 用量並加速，建議保持開啟"
        )
        form.addRow("amp:", self.amp_check)

        self.fraction_spin = QDoubleSpinBox()
        self.fraction_spin.setRange(0.01, 1.0)
        self.fraction_spin.setDecimals(2)
        self.fraction_spin.setSingleStep(0.05)
        self.fraction_spin.setToolTip(
            "使用訓練集的比例, default=1.0\n設 0.1 可快速測試 pipeline"
        )
        form.addRow("fraction:", self.fraction_spin)

        self.freeze_spin = QSpinBox()
        self.freeze_spin.setRange(0, 100)
        self.freeze_spin.setToolTip(
            "凍結前 N 層不更新, default=0\n遷移學習時可凍結 backbone"
        )
        form.addRow("freeze:", self.freeze_spin)

        self.tabs.addTab(widget, "系統")

    # === Load / Save ===

    def _load_from_settings(self) -> None:
        """從 settings.training 把值灌到 UI"""
        t = settings.training

        # Optimizer
        idx = self.optimizer_combo.findData(t.optimizer)
        self.optimizer_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.lr0_spin.setValue(t.lr0)
        self.lrf_spin.setValue(t.lrf)
        self.weight_decay_spin.setValue(t.weight_decay)
        self.warmup_epochs_spin.setValue(t.warmup_epochs)
        self.warmup_momentum_spin.setValue(t.warmup_momentum)

        # Geom / flip
        self.degrees_spin.setValue(t.degrees)
        self.translate_spin.setValue(t.translate)
        self.scale_spin.setValue(t.scale)
        self.perspective_spin.setValue(t.perspective)
        self.flipud_spin.setValue(t.flipud)
        self.fliplr_spin.setValue(t.fliplr)

        # Color
        self.hsv_h_spin.setValue(t.hsv_h)
        self.hsv_s_spin.setValue(t.hsv_s)
        self.hsv_v_spin.setValue(t.hsv_v)

        # Mix
        self.mosaic_spin.setValue(t.mosaic)
        self.close_mosaic_spin.setValue(t.close_mosaic)
        self.mixup_spin.setValue(t.mixup)
        self.copy_paste_spin.setValue(t.copy_paste)

        # System
        self.workers_spin.setValue(t.workers)
        idx = self.cache_combo.findData((t.cache or "false").lower())
        self.cache_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.rect_check.setChecked(bool(t.rect))
        self.amp_check.setChecked(bool(t.amp))
        self.fraction_spin.setValue(t.fraction)
        self.freeze_spin.setValue(t.freeze)

    def _reset_defaults(self) -> None:
        """重設為 TrainingSettings 預設值 (不影響其他 tab/dialog 中的基本參數)"""
        defaults = TrainingSettings()
        # 只重設進階欄位，保留主對話框管理的基本欄位
        for field in (
            "optimizer", "lr0", "lrf", "weight_decay",
            "warmup_epochs", "warmup_momentum",
            "degrees", "translate", "scale", "perspective",
            "flipud", "fliplr", "hsv_h", "hsv_s", "hsv_v",
            "mosaic", "close_mosaic", "mixup", "copy_paste",
            "workers", "cache", "rect", "amp", "fraction", "freeze",
        ):
            setattr(settings.training, field, getattr(defaults, field))
        self._load_from_settings()

    def _save(self) -> None:
        """寫回 settings.training 並持久化"""
        t = settings.training

        t.optimizer = self.optimizer_combo.currentData()
        t.lr0 = self.lr0_spin.value()
        t.lrf = self.lrf_spin.value()
        t.weight_decay = self.weight_decay_spin.value()
        t.warmup_epochs = self.warmup_epochs_spin.value()
        t.warmup_momentum = self.warmup_momentum_spin.value()

        t.degrees = self.degrees_spin.value()
        t.translate = self.translate_spin.value()
        t.scale = self.scale_spin.value()
        t.perspective = self.perspective_spin.value()
        t.flipud = self.flipud_spin.value()
        t.fliplr = self.fliplr_spin.value()

        t.hsv_h = self.hsv_h_spin.value()
        t.hsv_s = self.hsv_s_spin.value()
        t.hsv_v = self.hsv_v_spin.value()

        t.mosaic = self.mosaic_spin.value()
        t.close_mosaic = self.close_mosaic_spin.value()
        t.mixup = self.mixup_spin.value()
        t.copy_paste = self.copy_paste_spin.value()

        t.workers = self.workers_spin.value()
        t.cache = self.cache_combo.currentData()
        t.rect = self.rect_check.isChecked()
        t.amp = self.amp_check.isChecked()
        t.fraction = self.fraction_spin.value()
        t.freeze = self.freeze_spin.value()

        try:
            save_settings()
        except Exception as e:
            log.e(f"儲存 settings 失敗: {e}")
        self.accept()
