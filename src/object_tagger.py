# 主視窗：工具列、選單、快捷鍵、儲存標註等主要UI邏輯
# 更新日期: 2026-04-25
import random
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSlider,
    QStatusBar,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from ruamel.yaml import YAML

from src.config import cfg
from src.core import AppState
from src.utils.const import IMAGE_EXTS
from src.image_widget import DrawingMode, ImageWidget
from src.dialogs import (
    CategorizeMediaDialog,
    ConvertSettingsDialog,
    SetSam3ModelDialog,
    SetYoloModelDialog,
    TrainYoloDialog,
)
from src.utils.dynamic_settings import save_settings, settings
from src.utils.file_handler import file_h
from src.utils.img_handler import inferencer
from src.utils.func import getMaskPath, getXmlPath
from src.utils.global_param import g_param
from src.utils.logger import getUniqueLogger
from src.utils.model import FileType, ModelType, PlayState, ShowImageCmd, ViewMode

log = getUniqueLogger(__file__)
yaml = YAML()

YOLO_LABELS_FOLDER = "labels"


class _ModelLoader(QThread):
    """背景載入模型（含下載）的 worker thread"""
    finished = pyqtSignal(bool, str)  # (success, message)

    def __init__(self, model_type: str):
        super().__init__()
        self.model_type = model_type

    def run(self):
        try:
            ok = inferencer.ensure_loaded(self.model_type)
            if ok:
                self.finished.emit(True, "模型載入完成")
            else:
                self.finished.emit(False, "模型載入失敗")
        except Exception as e:
            log.e(f"模型載入錯誤: {e}")
            self.finished.emit(False, "模型載入失敗")


class MainWindow(QMainWindow):
    def __init__(self):

        super().__init__()

        self.setWindowTitle("Image Tagger")

        # 設定最小尺寸
        self.setMinimumSize(500, 500)

        # Initialize state
        self.app_state = AppState()
        self._model_loader: _ModelLoader | None = None
        self._detect_after_load = False

        # 儲存
        self.save_action = QAction("Save", self)
        self.save_action.triggered.connect(self.saveImgAndLabels)

        self.auto_save_action = QAction("Auto Save", self)
        self.auto_save_action.setCheckable(True)
        self.auto_save_action.triggered.connect(self.app_state.toggle_auto_save)

        self.save_mask_action = QAction("Save Mask", self)
        self.save_mask_action.triggered.connect(self.saveMask)

        # 自動使用偵測 (GPU不好速度就會慢)
        self.auto_detect_action = QAction("Auto Detect", self)
        self.auto_detect_action.setCheckable(True)
        self.auto_detect_action.triggered.connect(self.app_state.toggle_auto_detect)

        # 檔案相關動作
        self.open_folder_action = QAction("Open Folder", self)
        self.open_folder_action.triggered.connect(self.open_folder)

        # 狀態列
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")

        # 中央 Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.image_widget = ImageWidget(self.app_state)
        self.main_layout.addWidget(self.image_widget)

        # 工具列
        self.toolbar = QToolBar()
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.toolbar)

        self.toolbar.addAction(self.auto_save_action)
        self.toolbar.addAction(self.auto_detect_action)
        self.toolbar.addAction(self.save_action)

        # 播放控制
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.play_state = PlayState.STOP

        self.play_pause_action = QAction("", self)
        self.play_pause_action.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.play_pause_action.triggered.connect(self.toggle_play_pause)
        self.toolbar.addAction(self.play_pause_action)
        self.refresh_interval = 30

        self.progress_bar = QSlider(Qt.Orientation.Horizontal)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.progress_bar.valueChanged.connect(self.set_media_position)
        self.toolbar.addWidget(self.progress_bar)

        self.speed_control = QComboBox()
        self.speed_control.addItem("0.5x", 0.5)
        self.speed_control.addItem("1.0x", 1.0)
        self.speed_control.addItem("1.5x", 1.5)
        self.speed_control.addItem("2.0x", 2.0)
        self.speed_control.setCurrentIndex(1)  # 預設為 1.0x
        self.speed_control.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.speed_control.currentIndexChanged.connect(self.set_playback_speed)
        self.toolbar.addWidget(self.speed_control)

        self.play_pause_action.setEnabled(False)
        self.progress_bar.setEnabled(False)
        self.speed_control.setEnabled(False)
        # ^播放控制

        # Drawing Mode Toolbar
        self.drawing_toolbar = QToolBar("Drawing Tools")
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.drawing_toolbar)
        self.drawing_mode_group = QActionGroup(self)
        self.drawing_mode_group.setExclusive(True)

        # 選取模式（預設）
        self.select_mode_action = QAction("Select", self)
        self.select_mode_action.setCheckable(True)
        self.select_mode_action.setChecked(True)
        self.select_mode_action.triggered.connect(
            lambda: self.image_widget.set_drawing_mode(DrawingMode.SELECT)
        )
        self.drawing_toolbar.addAction(self.select_mode_action)
        self.drawing_mode_group.addAction(self.select_mode_action)

        self.bbox_mode_action = QAction("BBox", self)
        self.bbox_mode_action.setCheckable(True)
        self.bbox_mode_action.triggered.connect(
            lambda: self.image_widget.set_drawing_mode(DrawingMode.BBOX)
        )
        self.drawing_toolbar.addAction(self.bbox_mode_action)
        self.drawing_mode_group.addAction(self.bbox_mode_action)

        # Mask 工具 (Draw / Erase / Fill)，由 cfg.enable_mask_tools 控制
        self.draw_mode_action = QAction("Draw", self)
        self.draw_mode_action.setCheckable(True)
        self.draw_mode_action.triggered.connect(
            lambda: self.image_widget.set_drawing_mode(DrawingMode.MASK_DRAW)
        )

        self.erase_mode_action = QAction("Erase", self)
        self.erase_mode_action.setCheckable(True)
        self.erase_mode_action.triggered.connect(
            lambda: self.image_widget.set_drawing_mode(DrawingMode.MASK_ERASE)
        )

        self.fill_mode_action = QAction("Fill", self)
        self.fill_mode_action.setCheckable(True)
        self.fill_mode_action.triggered.connect(
            lambda: self.image_widget.set_drawing_mode(DrawingMode.MASK_FILL)
        )

        if cfg.enable_mask_tools:
            self.drawing_toolbar.addAction(self.draw_mode_action)
            self.drawing_mode_group.addAction(self.draw_mode_action)
            self.drawing_toolbar.addAction(self.erase_mode_action)
            self.drawing_mode_group.addAction(self.erase_mode_action)
            self.drawing_toolbar.addAction(self.fill_mode_action)
            self.drawing_mode_group.addAction(self.fill_mode_action)

        self.polygon_mode_action = QAction("Polygon", self)
        self.polygon_mode_action.setCheckable(True)
        self.polygon_mode_action.triggered.connect(
            lambda: self.image_widget.set_drawing_mode(DrawingMode.POLYGON)
        )
        self.drawing_toolbar.addAction(self.polygon_mode_action)
        self.drawing_mode_group.addAction(self.polygon_mode_action)

        if cfg.enable_mask_tools:
            self.drawing_toolbar.addSeparator()
            self.brush_size_slider = QSlider(Qt.Orientation.Vertical)
            self.brush_size_slider.setRange(1, 100)
            self.brush_size_slider.setValue(20)
            self.brush_size_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.brush_size_slider.valueChanged.connect(self.image_widget.set_brush_size)
            self.drawing_toolbar.addWidget(QLabel("Brush Size"))
            self.drawing_toolbar.addWidget(self.brush_size_slider)
        # ^Drawing Mode Toolbar

        # 退出
        self.quit_action = QAction("Quit", self)
        self.quit_action.triggered.connect(self.close)

        # 偵測
        self.detect_action = QAction("Detect", self)
        self.detect_action.triggered.connect(self.manual_detect)

        # 主選單
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")
        self.edit_menu = self.menu.addMenu("Edit")
        self.ai_menu = self.menu.addMenu("Ai")
        self.train_menu = self.menu.addMenu("Train")
        self.view_menu = self.menu.addMenu("View")
        # self.help_menu = self.menu.addMenu("&Help")

        # View mode actions
        self.view_action_group = QActionGroup(self)
        self.view_action_group.setExclusive(True)

        self.view_all_action = QAction("Show All", self)
        self.view_all_action.setCheckable(True)
        self.view_all_action.setChecked(True)
        self.view_all_action.triggered.connect(
            lambda: self.image_widget.set_view_mode(ViewMode.ALL)
        )
        self.view_action_group.addAction(self.view_all_action)

        self.view_bbox_action = QAction("Show BBox Only", self)
        self.view_bbox_action.setCheckable(True)
        self.view_bbox_action.triggered.connect(
            lambda: self.image_widget.set_view_mode(ViewMode.BBOX)
        )
        self.view_action_group.addAction(self.view_bbox_action)

        self.view_seg_action = QAction("Show Seg Only", self)
        self.view_seg_action.setCheckable(True)
        self.view_seg_action.triggered.connect(
            lambda: self.image_widget.set_view_mode(ViewMode.SEG)
        )
        self.view_action_group.addAction(self.view_seg_action)

        self.view_menu.addAction(self.view_all_action)
        self.view_menu.addAction(self.view_bbox_action)
        self.view_menu.addAction(self.view_seg_action)

        self.convert_voc_yolo_action = QAction("VOC to YOLO", self)
        self.convert_voc_yolo_action.triggered.connect(self.convert_voc_to_yolo)

        self.train_yolo_action = QAction("Train YOLO", self)
        self.train_yolo_action.triggered.connect(self.train_yolo)

        self.train_menu.addAction(self.convert_voc_yolo_action)
        self.train_menu.addAction(self.train_yolo_action)

        self.open_file_by_index_action = QAction("Open File by Index", self)
        self.open_file_by_index_action.triggered.connect(self.open_file_by_index)

        self.file_menu.addAction(self.open_folder_action)
        self.file_menu.addAction(self.open_file_by_index_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_action)
        self.file_menu.addAction(self.auto_save_action)
        if cfg.enable_mask_tools:
            self.file_menu.addAction(self.save_mask_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.quit_action)

        # 變更標籤
        self.edit_label_action = QAction("Edit Label", self)
        self.edit_label_action.triggered.connect(self.promptInputLabel)

        self.edit_menu.addAction(self.edit_label_action)

        # Model selection radio group
        self.model_action_group = QActionGroup(self)
        self.model_action_group.setExclusive(True)

        self.use_yolo_action = QAction("Use YOLO", self)
        self.use_yolo_action.setCheckable(True)
        self.use_yolo_action.triggered.connect(
            lambda: self._set_model(ModelType.YOLO)
        )
        self.model_action_group.addAction(self.use_yolo_action)

        self.use_sam_action = QAction("Use SAM3", self)
        self.use_sam_action.setCheckable(True)
        self.use_sam_action.triggered.connect(
            lambda: self._set_model(ModelType.SAM3)
        )
        self.model_action_group.addAction(self.use_sam_action)

        self.ai_menu.addAction(self.use_yolo_action)
        if cfg.enable_sam3:
            self.ai_menu.addAction(self.use_sam_action)
        self.ai_menu.addSeparator()

        self.set_yolo_model_action = QAction("Set YOLO Model", self)
        self.set_yolo_model_action.triggered.connect(self.set_yolo_model)

        self.set_sam3_model_action = QAction("Set SAM3 Model", self)
        self.set_sam3_model_action.triggered.connect(self.set_sam3_model)

        self.ai_menu.addAction(self.set_yolo_model_action)
        if cfg.enable_sam3:
            self.ai_menu.addAction(self.set_sam3_model_action)
        self.ai_menu.addSeparator()
        self.ai_menu.addAction(self.detect_action)
        self.ai_menu.addAction(self.auto_detect_action)
        self.ai_menu.addSeparator()

        self.categorize_media_action = QAction("Categorize Media", self)
        self.categorize_media_action.triggered.connect(self.categorize_media)
        self.ai_menu.addAction(self.categorize_media_action)

        # Store model paths only (lazy load on first inference)
        # Verify model files exist before assigning
        missing_models = []
        if cfg.enable_sam3 and settings.models.sam3_model_path:
            if Path(settings.models.sam3_model_path).is_file():
                inferencer.sam_model_path = settings.models.sam3_model_path
            else:
                missing_models.append(f"SAM3: {settings.models.sam3_model_path}")
        if settings.models.model_path:
            if Path(settings.models.model_path).is_file():
                inferencer.model_path = settings.models.model_path
            elif re.match(r"^yolo\d+\w*\.pt$", settings.models.model_path):
                # 預設 YOLO 模型名稱，ultralytics 會自動下載，不需警示
                inferencer.model_path = settings.models.model_path
            else:
                missing_models.append(f"YOLO: {settings.models.model_path}")
        if missing_models:
            QMessageBox.warning(
                self,
                "Model Not Found",
                "以下模型檔案不存在:\n" + "\n".join(missing_models),
            )
        # Fallback: 沒有任何 YOLO 模型時，使用預設模型（ultralytics 會自動下載）
        if not inferencer.model_path:
            inferencer.model_path = "yolo26s.pt"
            settings.models.model_path = "yolo26s.pt"
        # Restore last active model from settings, fallback to priority logic
        if cfg.enable_sam3 and settings.models.active_model == ModelType.SAM3 and inferencer.sam_model_path:
            inferencer.active_model_type = ModelType.SAM3
        elif settings.models.active_model == ModelType.YOLO and inferencer.model_path:
            inferencer.active_model_type = ModelType.YOLO
            self.app_state.auto_detect = True
        elif inferencer.model_path:
            inferencer.active_model_type = ModelType.YOLO
            self.app_state.auto_detect = True
        elif cfg.enable_sam3 and inferencer.sam_model_path:
            inferencer.active_model_type = ModelType.SAM3
        self.choose_folder(settings.file_system.folder_path, settings.file_system.file_index)

        try:
            with open("cfg/system.yaml", "r", encoding="utf-8") as f:
                config = yaml.load(f)
            yaml_labels = config.get("labels", {})
            if yaml_labels:
                self.app_state.preset_labels = {
                    str(key): value
                    for key, value in yaml_labels.items()
                    if isinstance(key, (int, str))
                }
            self.app_state.last_used_label = config.get("default_label", "object")

        except FileNotFoundError:
            QMessageBox.warning(self, "Warning", "config/cfg.yaml not found.")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error parsing config/cfg.yaml: {e}")

        # Register callbacks
        self.app_state.register_callback(
            "auto_save_changed", self._on_auto_save_changed
        )
        self.app_state.register_callback(
            "auto_detect_changed", self._on_auto_detect_changed
        )
        self.app_state.register_callback("status_message", self.statusbar.showMessage)

        # Multi-digit label key input timer
        self._label_key_timer = QTimer(self)
        self._label_key_timer.setSingleShot(True)
        self._label_key_timer.setInterval(int(cfg.label_key_timeout * 1000))
        self._label_key_timer.timeout.connect(self._on_label_key_timeout)

        # Set image widget callbacks
        self.image_widget.set_callbacks(
            on_mouse_press=self.cbMousePress,
            on_wheel_event=self.cbWheelEvent,
            on_video_loaded=self.cbVideoLoaded,
            on_image_loaded=self.cbImageLoaded,
        )

        # Sync initial UI state with app_state
        self._sync_ui_state()

    def _sync_ui_state(self):
        """Sync UI components with app_state."""
        self.auto_save_action.setChecked(self.app_state.auto_save)
        self.auto_detect_action.setChecked(self.app_state.auto_detect)
        if inferencer.active_model_type == ModelType.YOLO:
            self.use_yolo_action.setChecked(True)
        elif inferencer.active_model_type == ModelType.SAM3:
            self.use_sam_action.setChecked(True)

    def _on_auto_save_changed(self, enabled: bool):
        """Callback when auto save state changes."""
        self.auto_save_action.setChecked(enabled)

    def _on_auto_detect_changed(self, enabled: bool):
        """Callback when auto detect state changes."""
        self.auto_detect_action.setChecked(enabled)
        if enabled:
            self.image_widget.runInference()

    def _set_model(self, model_type: str, model_path: str = None):
        """Set active model and sync UI. 若模型尚未載入則觸發背景載入。"""
        inferencer.set_active_model(model_type, model_path)
        if model_type == ModelType.YOLO:
            self.use_yolo_action.setChecked(True)
        elif model_type == ModelType.SAM3:
            self.use_sam_action.setChecked(True)
        settings.models.active_model = model_type
        save_settings()
        # 背景預載模型
        self._load_model_async(model_type)

    def _load_model_async(self, model_type: str, detect_after: bool = False):
        """背景載入模型，不阻塞 UI"""
        if inferencer.is_loaded(model_type):
            if detect_after:
                self._run_detect()
            return
        if inferencer.is_loading:
            return
        self._detect_after_load = detect_after
        self._model_loader = _ModelLoader(model_type)
        self._model_loader.finished.connect(self._on_model_loaded)
        # 判斷是否需要下載
        model_path = (
            inferencer.model_path if model_type == ModelType.YOLO
            else inferencer.sam_model_path
        ) or ""
        if model_path and not Path(model_path).is_file():
            self.statusbar.showMessage(f"正在下載並載入模型: {Path(model_path).name} ...")
        else:
            self.statusbar.showMessage(f"正在載入模型: {Path(model_path).name} ...")
        self._model_loader.start()

    def _on_model_loaded(self, success: bool, message: str):
        """模型載入完成的回呼"""
        self.statusbar.showMessage(message)
        if success and self._detect_after_load:
            self._detect_after_load = False
            self._run_detect()

    def _run_detect(self):
        """執行偵測並更新 statusbar"""
        self.image_widget.runInference()
        nb = len(self.image_widget.bboxes) + len(self.image_widget.polygons)
        self.statusbar.showMessage(f"Detect ({inferencer.active_model_type}): {nb} annotations")

    def manual_detect(self):
        """手動偵測，提供狀態回饋。模型未載入時觸發背景載入，載入完成後自動偵測。"""
        if inferencer.active_model_type == ModelType.NONE:
            self.statusbar.showMessage("Detect: no model selected (use Ai menu)")
            return
        if not file_h.current_image_path():
            self.statusbar.showMessage("Detect: no image loaded")
            return
        if inferencer.is_loading:
            self.statusbar.showMessage("模型載入中，請稍候...")
            return
        if not inferencer.is_loaded(inferencer.active_model_type):
            # 模型尚未載入，背景載入後自動偵測
            self._load_model_async(inferencer.active_model_type, detect_after=True)
            return
        self._run_detect()

    def resetStates(self):
        g_param.auto_save_counter = 0
        self.play_state = PlayState.STOP
        self.image_widget.clearBboxes()

    def set_yolo_model(self):
        """開啟 Set YOLO Model 對話框"""
        dialog = SetYoloModelDialog(self)
        if dialog.exec():
            self._set_model(ModelType.YOLO, settings.models.model_path)
            save_settings()

    def set_sam3_model(self):
        """開啟 Set SAM3 Model 對話框（含 output mode、tolerance、text prompts）"""
        dialog = SetSam3ModelDialog(self)
        if dialog.exec():
            self._set_model(ModelType.SAM3, settings.models.sam3_model_path)
            save_settings()

    def cycle_view_mode(self):
        """Cycle view mode: ALL -> BBOX -> SEG -> ALL"""
        current = self.image_widget.view_mode
        if current == ViewMode.ALL:
            self.image_widget.set_view_mode(ViewMode.BBOX)
            self.view_bbox_action.setChecked(True)
        elif current == ViewMode.BBOX:
            self.image_widget.set_view_mode(ViewMode.SEG)
            self.view_seg_action.setChecked(True)
        else:
            self.image_widget.set_view_mode(ViewMode.ALL)
            self.view_all_action.setChecked(True)
        self.statusbar.showMessage(f"View mode: {self.image_widget.view_mode}")

    def open_file_by_index(self):
        self.resetStates()
        if not file_h.folder_path:
            QMessageBox.warning(self, "Warning", "No folder opened.")
            return

        if not file_h.image_files:
            # QMessageBox.warning(self, "Warning", "No files in the folder.")
            self.statusbar.showMessage("No files in the folder")
            return

        num_files = len(file_h.image_files)
        index, ok = QInputDialog.getInt(
            self,
            "Open File by Index",
            f"Enter file index (1-{num_files}):",
            1,  # 預設值
            1,  # 最小值
            num_files,  # 最大值
        )

        if ok:
            file_h.current_index = index - 1
            self.image_widget.load_image(file_h.current_image_path())
            self.statusbar.showMessage(
                f"[{file_h.current_index + 1} / {len(file_h.image_files)}] "
                f"Image: {file_h.current_image_path()}"
            )

    def open_folder(self):
        """
        用pyqt瀏覽並選定資料夾
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Open Folder")  # PyQt6
        self.choose_folder(folder_path)
        settings.file_system.folder_path = folder_path
        settings.file_system.file_index = 0

    def choose_folder(self, folder_path: str, file_index: int = 0):
        """
        開啟資料夾的檔案
        """
        if folder_path and Path(folder_path).is_dir():
            file_h.load_folder(folder_path)
            if file_h.image_files:
                file_h.current_index = min(file_index, len(file_h.image_files) - 1)
                self.image_widget.load_image(file_h.current_image_path())
                self.statusbar.showMessage(
                    f"[{file_h.current_index + 1} / {len(file_h.image_files)}] "
                    f"Opened folder: {folder_path} "
                )
            else:
                # QMessageBox.information(self, "Info", "No files in the folder.")
                self.statusbar.showMessage(f"No files in the folder `{folder_path}`")
        else:
            self.statusbar.showMessage(f"Invalid folder path `{folder_path}`")

    def show_image(self, cmd: str):
        """show下一個影校或影片, 如有自動記錄則要先儲存之前的labels"""
        if self.app_state.auto_save or g_param.user_labeling:
            self.saveImgAndLabels()
        self.resetStates()
        if file_h.show_image(cmd):
            self.image_widget.load_image(file_h.current_image_path())
            self.statusbar.showMessage(
                f"[{file_h.current_index + 1} / {len(file_h.image_files)}] "
                f"Image: {file_h.current_image_path()}"
            )
            settings.file_system.file_index = file_h.current_index
            save_settings()

    def update_frame(self):
        if self.play_state == PlayState.PLAY and self.image_widget.cap:
            ret, self.image_widget.cv_img = self.image_widget.cap.read()
            self.image_widget.clearBboxes()
            if not ret:
                self.timer.stop()
                self.play_state = PlayState.STOP
                icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
                self.play_pause_action.setIcon(icon)
                return

            height, width, channel = self.image_widget.cv_img.shape
            bytesPerLine = 3 * width
            qImg = QImage(
                self.image_widget.cv_img.data,
                width,
                height,
                bytesPerLine,
                QImage.Format.Format_RGB888,
            ).rgbSwapped()
            self.image_widget.pixmap = QPixmap.fromImage(qImg)

            if self.app_state.auto_detect:
                self.image_widget.runInference()
            else:
                self.image_widget.update()

            position = self.image_widget.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.progress_bar.blockSignals(True)  # 暫時阻止信號傳遞
            self.progress_bar.setValue(int(position))
            self.progress_bar.blockSignals(False)  # 恢復信號傳遞

            # 自動儲存邏輯
            if self.app_state.auto_save and cfg.auto_save_per_second > 0:
                g_param.auto_save_counter += 1
                if (
                    g_param.auto_save_counter
                    >= cfg.auto_save_per_second * self.image_widget.fps
                ):
                    self.saveImgAndLabels()
                    g_param.auto_save_counter = 0

    def saveImgAndLabels(self):
        """
        儲存標記, 注意就算沒有bbox也是要儲存, 表示有處理過, 也能讓trainer知道這是在訓練背景
        """
        current_path = file_h.current_image_path()
        if not current_path:
            return
        Path(file_h.folder_path, cfg.save_folder).mkdir(parents=True, exist_ok=True)
        # Save BBox annotations (XML)
        if self.image_widget.file_type == FileType.VIDEO:
            # 儲存影片當前幀
            frame = self.image_widget.pixmap.toImage()  # 從 pixmap 取得
            pure_name = Path(current_path).stem
            frame_number = int(self.image_widget.cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_filename = f"{pure_name}_frame{frame_number}.jpg"
            save_path = Path(
                file_h.folder_path, cfg.save_folder, frame_filename
            ).as_posix()
            frame.save(save_path)
        else:
            # 搬移圖片
            file_name = Path(current_path).name
            save_path = Path(file_h.folder_path, cfg.save_folder, file_name).as_posix()
            shutil.copy(current_path, save_path)

        # 儲存xml
        xml_path = getXmlPath(save_path)
        bboxes = self.image_widget.bboxes
        polygons = self.image_widget.polygons
        xml_content = file_h.generate_voc_xml(bboxes, save_path, polygons)
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        g_param.user_labeling = False
        status_message = f"Annotations saved to {xml_path}"
        self.statusbar.showMessage(status_message)

    def saveMask(self):
        current_path = file_h.current_image_path()
        if not current_path:
            current_path = "./"

        if self.image_widget.mask_pixmap:
            mask_path = getMaskPath(current_path).as_posix()
            self.image_widget.mask_pixmap.save(mask_path, "PNG")
        self.statusbar.showMessage(f"Mask saved to {mask_path}")

    def toggle_play_pause(self):
        if self.image_widget.file_type != FileType.VIDEO:
            self.play_state = PlayState.STOP
            icon = self.style().standardIcon(
                QStyle.StandardPixmap.SP_TitleBarCloseButton
            )
            self.play_pause_action.setIcon(icon)
            return

        if self.play_state == PlayState.STOP:
            self.progress_bar.setValue(0)

        if self.play_state in [PlayState.STOP, PlayState.PAUSE]:
            # 播放前先儲存標籤資訊
            if self.app_state.auto_save or g_param.user_labeling:
                self.saveImgAndLabels()
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            self.play_pause_action.setIcon(icon)
            self.timer.start(self.refresh_interval)
            self.play_state = PlayState.PLAY
        else:
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            self.play_pause_action.setIcon(icon)
            self.timer.stop()
            self.play_state = PlayState.PAUSE

    def set_media_position(self, position):
        # 設定影片播放位置 (以毫秒為單位), 並顯示該位置的 frame
        iw = self.image_widget
        if not iw.cap:
            return
        self.progress_bar.blockSignals(True)
        iw.cap.set(cv2.CAP_PROP_POS_MSEC, position)
        ret, iw.cv_img = iw.cap.read()
        iw.clearBboxes()
        self.progress_bar.blockSignals(False)
        self.progress_bar.setValue(position)
        if ret:
            h, w, _ = iw.cv_img.shape
            qImg = QImage(
                iw.cv_img.data, w, h, 3 * w, QImage.Format.Format_RGB888
            ).rgbSwapped()
            iw.pixmap = QPixmap.fromImage(qImg)
        iw.update()

    def set_playback_speed(self, index):
        # 設定播放速度
        if self.image_widget.file_type != FileType.VIDEO:
            return
        speed = self.speed_control.itemData(index)
        self.timer.stop()
        if self.image_widget.fps:
            fps = self.image_widget.fps
        else:
            fps = 30
        self.refresh_interval = int(fps / speed)
        if self.play_state == PlayState.PLAY:
            self.timer.start(self.refresh_interval)  # 重新啟動定時器

    def updateFocusedAnnotation(self):
        """更新選取中的bbox或polygon的標籤名稱（支援多選）"""
        iw = self.image_widget
        label = self.app_state.last_used_label

        # 多選：更新所有選取的項目
        if iw.select_type == "multi":
            for i in iw.selected_bbox_indices:
                if 0 <= i < len(iw.bboxes):
                    iw.bboxes[i].label = label
            for i in iw.selected_polygon_indices:
                if 0 <= i < len(iw.polygons):
                    iw.polygons[i].label = label
            iw.update()
            return

        # 單選
        if iw.select_type == "polygon" and 0 <= iw.idx_focus_polygon < len(iw.polygons):
            iw.polygons[iw.idx_focus_polygon].label = label
            iw.update()
            return
        if iw.select_type == "bbox" and 0 <= iw.idx_focus_bbox < len(iw.bboxes):
            iw.bboxes[iw.idx_focus_bbox].label = label
            iw.update()
            return

        # 非SELECT模式：更新focus的bbox或最後一個
        idx = (
            iw.idx_focus_bbox
            if 0 <= iw.idx_focus_bbox < len(iw.bboxes)
            else -1
        )
        if iw.bboxes:
            iw.bboxes[idx].label = label
            iw.update()

    def _format_label_hints(self, matches: list[tuple[str, str]]) -> str:
        """Format matching label candidates for status bar display."""
        return ", ".join(f"[{k}] {v}" for k, v in matches)

    def _handle_label_digit(self, digit: str):
        """Handle a digit key press for multi-digit label code input."""
        state = self.app_state
        buf = state.append_key_buffer(digit)

        if state.is_unique_prefix():
            # Exact match with no ambiguity — apply immediately
            self._label_key_timer.stop()
            self._apply_label_from_buffer()
        elif state.has_any_prefix_match():
            # Could still become a valid key — wait for more digits
            self._label_key_timer.start()
            hints = self._format_label_hints(state.get_prefix_matches())
            self.statusBar().showMessage(f"[{buf}] → {hints}")
        else:
            # No key starts with this buffer — discard
            self._label_key_timer.stop()
            self.statusBar().showMessage(f"[{buf}] — no match")
            state.clear_key_buffer()

    def _on_label_key_timeout(self):
        """Timer expired — resolve whatever is in the buffer."""
        self._apply_label_from_buffer()

    def _apply_label_from_buffer(self):
        """Resolve the key buffer to a label and apply it."""
        state = self.app_state
        buf = state.key_buffer
        label = state.resolve_key_buffer()
        if label:
            state.last_used_label = label
            self.updateFocusedAnnotation()
            self.statusBar().showMessage(f"Label: [{buf}] → {label}")
        else:
            self.statusBar().showMessage(f"Label key: [{buf}] — no match")

    def promptInputLabel(self):
        """彈出輸入框，讓使用者輸入標籤名稱（支援多選）"""
        iw = self.image_widget
        current_label = self.app_state.last_used_label

        # 取得目前選取物件的label作為預設值
        if iw.select_type == "multi":
            # 多選時取第一個選取項目的label
            if iw.selected_bbox_indices:
                first_idx = min(iw.selected_bbox_indices)
                if 0 <= first_idx < len(iw.bboxes):
                    current_label = iw.bboxes[first_idx].label
            elif iw.selected_polygon_indices:
                first_idx = min(iw.selected_polygon_indices)
                if 0 <= first_idx < len(iw.polygons):
                    current_label = iw.polygons[first_idx].label
        elif iw.select_type == "polygon" and 0 <= iw.idx_focus_polygon < len(iw.polygons):
            current_label = iw.polygons[iw.idx_focus_polygon].label
        elif iw.select_type == "bbox" and 0 <= iw.idx_focus_bbox < len(iw.bboxes):
            current_label = iw.bboxes[iw.idx_focus_bbox].label

        label, ok = QInputDialog.getText(
            self, "Input", "Edit current label name:", text=current_label
        )
        if ok and label.strip():
            self.app_state.set_last_used_label(label)
            self.updateFocusedAnnotation()

    def deletePairOfImgXml(self):
        """
        只有在擁有圖片跟有同名.xml的情況下, 才能一起刪除
        """
        current_path = file_h.current_image_path()
        xml_path = getXmlPath(current_path)
        if not current_path or not Path(xml_path).is_file():
            QMessageBox.warning(self, "Warning", "No img or no .xml")
            return

        Path(xml_path).unlink(missing_ok=True)
        Path(current_path).unlink(missing_ok=True)

        file_h.image_files.pop(file_h.current_index)
        log.i(f"delete {current_path} and his .xml")
        self.show_image(ShowImageCmd.SAME_INDEX)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_PageDown:
            self.show_image(ShowImageCmd.NEXT)
        elif event.key() == Qt.Key.Key_PageUp:
            self.show_image(ShowImageCmd.PREV)
        elif event.key() == Qt.Key.Key_Home:
            self.show_image(ShowImageCmd.FIRST)
        elif event.key() == Qt.Key.Key_End:
            self.show_image(ShowImageCmd.LAST)
        elif event.key() == Qt.Key.Key_Right:
            self.set_media_position(self.progress_bar.value() + 3000)
        elif event.key() == Qt.Key.Key_Left:
            self.set_media_position(self.progress_bar.value() - 3000)
        elif event.key() == Qt.Key.Key_Q:
            self.close()
        elif event.key() == Qt.Key.Key_S:
            self.saveImgAndLabels()
        elif event.key() == Qt.Key.Key_A:
            self.app_state.toggle_auto_save()
        elif event.key() == Qt.Key.Key_D:
            self.manual_detect()
        elif event.key() == Qt.Key.Key_Delete:
            # SELECT模式下先嘗試刪除選取的標註
            if self.image_widget.drawing_mode == DrawingMode.SELECT:
                if self.image_widget.deleteSelectedAnnotation():
                    self.statusbar.showMessage("已刪除選取的標註")
                    return
            # 不要刪除img和xml
            # self.deletePairOfImgXml()
        elif event.key() == Qt.Key.Key_Space:
            self.toggle_play_pause()
        elif event.key() == Qt.Key.Key_L:
            self.promptInputLabel()
        elif event.key() == Qt.Key.Key_P:
            self.polygon_mode_action.setChecked(True)
            self.image_widget.set_drawing_mode(DrawingMode.POLYGON)
        elif event.key() == Qt.Key.Key_V:
            self.select_mode_action.setChecked(True)
            self.image_widget.set_drawing_mode(DrawingMode.SELECT)
        elif event.key() == Qt.Key.Key_B:
            self.bbox_mode_action.setChecked(True)
            self.image_widget.set_drawing_mode(DrawingMode.BBOX)

        elif Qt.Key.Key_0 <= event.key() <= Qt.Key.Key_9:
            digit = str(event.key() - Qt.Key.Key_0)
            self._handle_label_digit(digit)

    def closeEvent(self, event):
        """
        關閉前需要儲存最後的標籤資訊, 並更新動態設定檔
        """
        if self.app_state.auto_save or g_param.user_labeling:
            self.saveImgAndLabels()
        save_settings()

    def convert_voc_to_yolo(self):
        """
        將 VOC XML 格式的標註檔案轉換為 YOLO TXT 標籤檔，
        並依 train/val 比例整理成 dataset 結構 + 產生 data.yaml
        """
        default_dir = str(Path(file_h.folder_path, cfg.save_folder)) if file_h.folder_path else ""
        dialog = ConvertSettingsDialog(self, self.app_state, default_dir)
        if not dialog.exec():
            return

        base = Path(dialog.folder_path)
        train_ratio = dialog.train_ratio
        copy_images = dialog.copy_images
        start_time = datetime.now()

        # 1) 轉換 VOC XML → YOLO txt（先輸出到暫存 labels/ 下），顯示進度條
        tmp_labels = base / YOLO_LABELS_FOLDER
        tmp_labels.mkdir(parents=True, exist_ok=True)

        progress = QProgressDialog("正在轉換 VOC → YOLO ...", "取消", 0, 100, self)
        progress.setWindowTitle("轉換進度")
        progress.setMinimumDuration(0)
        progress.setValue(0)

        canceled = False

        def on_progress(current: int, total: int):
            nonlocal canceled
            if progress.wasCanceled():
                canceled = True
                return
            progress.setMaximum(total)
            progress.setValue(current)
            progress.setLabelText(f"正在轉換 VOC → YOLO ... ({current}/{total})")
            QApplication.processEvents()

        not_matched = file_h.convertVocInFolder(
            str(base), tmp_labels, self.app_state, progress_callback=on_progress
        )
        progress.close()

        if canceled:
            self.statusbar.showMessage("轉換已取消")
            return

        # 寫入未對應的 class_name 記錄檔
        not_match_path = None
        if not_matched:
            not_match_name = f"not_match_{start_time.strftime('%Y_%m%d_%H%M%S')}.txt"
            not_match_path = base / not_match_name
            with open(not_match_path, "w", encoding="utf-8") as f:
                for image_filename, class_name in not_matched:
                    f.write(f"{image_filename}\t{class_name}\n")
            log.w(f"未對應的 class_name 已寫入: {not_match_path}")

        # 2) 收集有對應 label 的圖片
        image_files = sorted(
            f for f in base.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        paired = [
            f for f in image_files
            if (tmp_labels / f"{f.stem}.txt").exists()
        ]
        if not paired:
            QMessageBox.warning(self, "Warning", "沒有找到成功轉換的圖片/標籤配對")
            return

        random.shuffle(paired)
        split_idx = max(1, int(len(paired) * train_ratio))
        train_files = paired[:split_idx]
        val_files = paired[split_idx:] if split_idx < len(paired) else []

        # 3) 建立目錄結構並移動檔案
        for split_name, files in [("train", train_files), ("val", val_files)]:
            if not files:
                continue
            img_dir = base / "images" / split_name
            lbl_dir = base / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for img_path in files:
                txt_path = tmp_labels / f"{img_path.stem}.txt"
                if copy_images:
                    shutil.copy2(str(img_path), str(img_dir / img_path.name))
                else:
                    shutil.move(str(img_path), str(img_dir / img_path.name))
                shutil.move(str(txt_path), str(lbl_dir / txt_path.name))

        # 清除暫存 labels/ (已搬空)
        if tmp_labels.exists() and not any(tmp_labels.iterdir()):
            tmp_labels.rmdir()

        # 4) 產生 dataset yaml
        categories = settings.class_names.categories  # {name: id}
        # 反轉成 {id: name}，依 id 排序
        id_to_name = dict(sorted(
            ((v, k) for k, v in categories.items()),
            key=lambda x: x[0],
        ))

        data_yaml = {"path": str(base.resolve())}
        data_yaml["train"] = "images/train"
        # ultralytics 要求 train/val 都必須存在；無 val split 時退回指向 train
        data_yaml["val"] = "images/val" if val_files else "images/train"
        data_yaml["nc"] = len(id_to_name)
        data_yaml["names"] = id_to_name

        yaml_name = f"dataset_{start_time.strftime('%Y_%m%d_%H%M%S')}.yaml"
        yaml_path = base / yaml_name
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f)

        # 5) 顯示轉換結果摘要
        self._show_convert_summary(
            id_to_name, len(train_files), len(val_files),
            yaml_name, not_matched, not_match_path,
        )

    def _show_convert_summary(
        self,
        id_to_name: dict[int, str],
        train_count: int,
        val_count: int,
        yaml_name: str,
        not_matched: list[tuple[str, str]],
        not_match_path: Path | None,
    ):
        """顯示 VOC → YOLO 轉換完成的摘要對話框"""
        # class_name 對應表
        lines = ["轉換完成\n"]
        lines.append(f"  Train: {train_count} 張, Val: {val_count} 張")
        lines.append(f"  Dataset YAML: {yaml_name}")
        if val_count == 0:
            lines.append("  ⚠ 無 val split，dataset.yaml 的 val 已退回指向 train (僅供訓練啟動，建議下次設定 val 比例)")
        lines.append("")
        lines.append("── Class 對應表 ──")
        for cid, cname in id_to_name.items():
            lines.append(f"  {cid}: {cname}")

        # 未對應提示
        if not_matched and not_match_path:
            unique_names = sorted(set(cn for _, cn in not_matched))
            lines.append(f"\n⚠ 有 {len(not_matched)} 筆標註的 class_name 未對應到 categories:")
            for name in unique_names:
                lines.append(f"  - {name}")
            lines.append(f"\n詳細記錄: {not_match_path.name}")

        QMessageBox.information(self, "VOC → YOLO 轉換結果", "\n".join(lines))
        self.statusbar.showMessage(
            f"轉換完成 — train: {train_count}, val: {val_count}, "
            f"yaml: {yaml_name}"
        )

    def categorize_media(self):
        """開啟 Categorize Media 對話框，依 YOLO 偵測結果分類媒體檔案"""
        default_folder = str(file_h.folder_path) if file_h.folder_path else ""
        default_model = settings.models.model_path or ""
        dialog = CategorizeMediaDialog(self, default_folder, default_model)
        dialog.exec()

    def train_yolo(self):
        """開啟 Train YOLO 對話框，設定參數並啟動 ultralytics 訓練"""
        default_folder = ""
        if file_h.folder_path:
            # 優先用 dataset 轉換的輸出 (save_folder)，沒有再退回目前資料夾
            converted = Path(file_h.folder_path, cfg.save_folder)
            default_folder = str(converted) if converted.is_dir() else str(file_h.folder_path)
        dialog = TrainYoloDialog(self, default_folder)
        dialog.exec()

    def cbWheelEvent(self, wheel_up):
        if self.app_state.auto_save or g_param.user_labeling:
            self.saveImgAndLabels()
        if wheel_up:
            self.show_image(ShowImageCmd.PREV)
        else:
            self.show_image(ShowImageCmd.NEXT)

    def cbMousePress(self, event):
        if self.play_state == PlayState.PLAY:
            self.play_state = PlayState.PAUSE

    def cbVideoLoaded(self, total_msec: int):
        """Callback when a video is loaded."""
        self.progress_bar.setRange(0, total_msec)
        self.progress_bar.blockSignals(True)
        self.progress_bar.setValue(0)
        self.image_widget.cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        self.progress_bar.blockSignals(False)
        # 啟用與影片相關的控制項
        self.play_pause_action.setEnabled(True)
        self.progress_bar.setEnabled(True)
        self.speed_control.setEnabled(True)
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.play_pause_action.setIcon(icon)

    def cbImageLoaded(self):
        """Callback when an image is loaded."""
        self.progress_bar.blockSignals(True)
        self.progress_bar.setValue(0)
        self.progress_bar.blockSignals(False)
        # 禁用與影片相關的控制項
        self.play_pause_action.setEnabled(False)
        self.progress_bar.setEnabled(False)
        self.speed_control.setEnabled(False)
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton)
        self.play_pause_action.setIcon(icon)


def main():
    app = QApplication(sys.argv)
    MainWindow().show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
