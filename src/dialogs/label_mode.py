# Label Mode 對話框：設定標註儲存模式 (整張圖 / Cropped 裁切) 與 cropped 裁切參數
# 更新日期: 2026-07-14
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from src.utils.cropper import CROP_MODE_FIXED, CROP_MODE_PADDING
from src.utils.dynamic_settings import save_settings, settings

# 兩種儲存模式的說明與使用情境
FULL_MODE_DESC = (
    "<b>整張圖模式（預設）</b><br>"
    "儲存目前<b>整張影像</b> + 標註 (VOC XML)。即使沒有任何框也會儲存，"
    "可作為訓練用的背景 (負樣本)。<br>"
    "<span style='color:gray;'>使用情境：一般 YOLO 偵測 / 分割 dataset。"
    "物件在畫面中的大小與比例接近實際部署場景。</span>"
)
CROPPED_MODE_DESC = (
    "<b>Cropped 裁切模式</b><br>"
    "只裁切「<b>有畫框</b>」的區域，各自存成小圖 + 標註 (VOC XML)；"
    "沒有任何框則<b>不儲存</b>。每個框會外擴增加背景資訊，碰到影像邊緣時往對邊補足像素；"
    "相鄰、能落在同一裁切區內的多個框會合併成一張 (含多個標籤)。<br>"
    "<span style='color:gray;'>使用情境：對動態區 / ROI 過濾後的目標做放大裁切，"
    "讓小物件在 YOLO 輸入維度 (如 640) 下佔比更大，提升小目標偵測的訓練效果。</span>"
)


class LabelModeDialog(QDialog):
    """設定標註儲存模式與 cropped 裁切參數的對話框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label Mode")
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)

        # === 儲存模式選擇 ===
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("儲存模式："))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("整張圖 (Full Image)", "full")
        self.mode_combo.addItem("只存畫框 (Cropped)", "cropped")
        mode_row.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_row)

        # === 模式說明 ===
        self.desc_label = QLabel()
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet(
            "font-size: 12px; padding: 8px; background: rgba(128,128,128,0.12);"
            " border-radius: 4px;"
        )
        layout.addWidget(self.desc_label)

        # === Cropped 裁切參數 ===
        self.crop_group = QGroupBox("Cropped 裁切參數")
        form = QFormLayout(self.crop_group)

        self.size_mode_combo = QComboBox()
        self.size_mode_combo.addItem("固定外擴 padding (px)", CROP_MODE_PADDING)
        self.size_mode_combo.addItem("至少固定尺寸 (px, 對齊 YOLO 輸入)", CROP_MODE_FIXED)
        form.addRow("尺寸模式：", self.size_mode_combo)

        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 2000)
        self.padding_spin.setSingleStep(10)
        self.padding_spin.setSuffix(" px")
        form.addRow("每邊外擴：", self.padding_spin)

        self.fixed_spin = QSpinBox()
        self.fixed_spin.setRange(32, 4096)
        self.fixed_spin.setSingleStep(32)
        self.fixed_spin.setSuffix(" px")
        form.addRow("最小邊長：", self.fixed_spin)

        size_hint = QLabel(
            "固定外擴：每個框四周各外擴指定 pixel。\n"
            "至少固定尺寸：裁切區至少為此邊長 (置中於框)，框更大時取框尺寸。"
        )
        size_hint.setStyleSheet("color: gray; font-size: 11px;")
        size_hint.setWordWrap(True)
        form.addRow(size_hint)

        layout.addWidget(self.crop_group)

        # === 按鈕 ===
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
        self._load_settings()

        # 綁定連動 (放在載入之後，避免載入時觸發不必要的更新)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.size_mode_combo.currentIndexChanged.connect(self._update_size_controls)
        self._on_mode_changed()

    def _load_settings(self):
        """從 settings 載入目前的 label 設定到各控制項"""
        idx = self.mode_combo.findData(settings.label.save_mode or "full")
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

        idx = self.size_mode_combo.findData(
            settings.label.crop_size_mode or CROP_MODE_FIXED
        )
        if idx >= 0:
            self.size_mode_combo.setCurrentIndex(idx)

        self.padding_spin.setValue(settings.label.crop_padding_px or 50)
        self.fixed_spin.setValue(settings.label.crop_fixed_size or 640)

    def _on_mode_changed(self):
        """切換儲存模式時更新說明文字與 cropped 參數區的啟用狀態"""
        is_cropped = self.mode_combo.currentData() == "cropped"
        self.desc_label.setText(CROPPED_MODE_DESC if is_cropped else FULL_MODE_DESC)
        self.crop_group.setEnabled(is_cropped)
        self._update_size_controls()

    def _update_size_controls(self):
        """依尺寸模式啟用對應的參數輸入 (padding / fixed 二選一)"""
        is_cropped = self.mode_combo.currentData() == "cropped"
        is_padding = self.size_mode_combo.currentData() == CROP_MODE_PADDING
        self.padding_spin.setEnabled(is_cropped and is_padding)
        self.fixed_spin.setEnabled(is_cropped and not is_padding)

    def _save(self):
        """寫回 settings 並持久化"""
        settings.label.save_mode = self.mode_combo.currentData()
        settings.label.crop_size_mode = self.size_mode_combo.currentData()
        settings.label.crop_padding_px = self.padding_spin.value()
        settings.label.crop_fixed_size = self.fixed_spin.value()
        save_settings()
        self.accept()
