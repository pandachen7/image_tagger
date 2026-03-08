# 主視窗：工具列、選單、快捷鍵、儲存標註等主要UI邏輯
# 更新日期: 2026-03-08
import shutil
import sys
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
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
from src.image_widget import DrawingMode, ImageWidget
from src.utils.dialogs import (
    CategorySettingsDialog,
    ConvertSettingsDialog,
    ParamDialog,
    TextPromptsDialog,
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


class MainWindow(QMainWindow):
    def __init__(self):

        super().__init__()

        self.setWindowTitle("Image Tagger")

        # 設定最小尺寸
        self.setMinimumSize(500, 500)

        # Initialize state
        self.app_state = AppState()

        # 儲存
        self.save_action = QAction("Save", self)
        self.save_action.triggered.connect(self.saveImgAndLabels)

        self.auto_save_action = QAction("Auto Save", self)
        self.auto_save_action.setCheckable(True)
        self.auto_save_action.triggered.connect(self.app_state.toggle_auto_save)

        self.save_mask_action = QAction("Save &Mask", self)
        self.save_mask_action.triggered.connect(self.saveMask)

        # 自動使用偵測 (GPU不好速度就會慢)
        self.auto_detect_action = QAction("Auto Detect", self)
        self.auto_detect_action.setCheckable(True)
        self.auto_detect_action.triggered.connect(self.app_state.toggle_auto_detect)

        # 檔案相關動作
        self.open_folder_action = QAction("&Open Folder", self)
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

        self.draw_mode_action = QAction("Draw", self)
        self.draw_mode_action.setCheckable(True)
        self.draw_mode_action.triggered.connect(
            lambda: self.image_widget.set_drawing_mode(DrawingMode.MASK_DRAW)
        )
        self.drawing_toolbar.addAction(self.draw_mode_action)
        self.drawing_mode_group.addAction(self.draw_mode_action)

        self.erase_mode_action = QAction("Erase", self)
        self.erase_mode_action.setCheckable(True)
        self.erase_mode_action.triggered.connect(
            lambda: self.image_widget.set_drawing_mode(DrawingMode.MASK_ERASE)
        )
        self.drawing_toolbar.addAction(self.erase_mode_action)
        self.drawing_mode_group.addAction(self.erase_mode_action)

        self.fill_mode_action = QAction("Fill", self)
        self.fill_mode_action.setCheckable(True)
        self.fill_mode_action.triggered.connect(
            lambda: self.image_widget.set_drawing_mode(DrawingMode.MASK_FILL)
        )
        self.drawing_toolbar.addAction(self.fill_mode_action)
        self.drawing_mode_group.addAction(self.fill_mode_action)

        self.polygon_mode_action = QAction("Polygon", self)
        self.polygon_mode_action.setCheckable(True)
        self.polygon_mode_action.triggered.connect(
            lambda: self.image_widget.set_drawing_mode(DrawingMode.POLYGON)
        )
        self.drawing_toolbar.addAction(self.polygon_mode_action)
        self.drawing_mode_group.addAction(self.polygon_mode_action)

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
        self.quit_action = QAction("&Quit", self)
        self.quit_action.triggered.connect(self.close)

        # 偵測
        self.detect_action = QAction("&Detect", self)
        self.detect_action.triggered.connect(self.image_widget.runInference)

        # 主選單
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")
        self.edit_menu = self.menu.addMenu("Edit")
        self.ai_menu = self.menu.addMenu("Ai")
        self.convert_menu = self.menu.addMenu("&Convert")
        self.view_menu = self.menu.addMenu("&View")
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

        self.edit_categories_action = QAction("&Edit Categories", self)
        self.edit_categories_action.triggered.connect(self.edit_categories)

        self.convert_settings_action = QAction("&Settings", self)
        self.convert_settings_action.triggered.connect(self.show_convert_settings)

        self.convert_voc_yolo_action = QAction("&VOC to YOLO", self)
        self.convert_voc_yolo_action.triggered.connect(self.convert_voc_to_yolo)

        self.convert_menu.addAction(self.edit_categories_action)
        self.convert_menu.addAction(self.convert_settings_action)
        self.convert_menu.addAction(self.convert_voc_yolo_action)

        self.open_file_by_index_action = QAction("Open File by &Index", self)
        self.open_file_by_index_action.triggered.connect(self.open_file_by_index)

        self.file_menu.addAction(self.open_folder_action)
        self.file_menu.addAction(self.open_file_by_index_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_action)
        self.file_menu.addAction(self.auto_save_action)
        self.file_menu.addAction(self.save_mask_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.quit_action)

        # 變更標籤
        self.edit_label_action = QAction("&Edit Label", self)
        self.edit_label_action.triggered.connect(self.promptInputLabel)

        self.edit_menu.addAction(self.edit_label_action)

        self.edit_text_prompts_action = QAction("&Text Prompts", self)
        self.edit_text_prompts_action.triggered.connect(self.edit_text_prompts)
        self.edit_menu.addAction(self.edit_text_prompts_action)

        self.edit_param_action = QAction("&Param", self)
        self.edit_param_action.triggered.connect(self.edit_param)
        self.edit_menu.addAction(self.edit_param_action)

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
        self.ai_menu.addAction(self.use_sam_action)
        self.ai_menu.addSeparator()

        self.select_model_action = QAction("Select &YOLO Model...", self)
        self.select_model_action.triggered.connect(self.select_model)

        self.select_sam_model_action = QAction("Select &SAM3 Model...", self)
        self.select_sam_model_action.triggered.connect(self.select_sam_model)

        self.ai_menu.addAction(self.select_model_action)
        self.ai_menu.addAction(self.select_sam_model_action)
        self.ai_menu.addSeparator()
        self.ai_menu.addAction(self.detect_action)
        self.ai_menu.addAction(self.auto_detect_action)

        # Store model paths only (lazy load on first inference)
        # Verify model files exist before assigning
        missing_models = []
        if settings.models.sam3_model_path:
            if Path(settings.models.sam3_model_path).is_file():
                inferencer.sam_model_path = settings.models.sam3_model_path
            else:
                missing_models.append(f"SAM3: {settings.models.sam3_model_path}")
        if settings.models.model_path:
            if Path(settings.models.model_path).is_file():
                inferencer.model_path = settings.models.model_path
            else:
                missing_models.append(f"YOLO: {settings.models.model_path}")
        if missing_models:
            QMessageBox.warning(
                self,
                "Model Not Found",
                "以下模型檔案不存在:\n" + "\n".join(missing_models),
            )
        # Restore last active model from settings, fallback to priority logic
        if settings.models.active_model == ModelType.SAM3 and inferencer.sam_model_path:
            inferencer.active_model_type = ModelType.SAM3
        elif settings.models.active_model == ModelType.YOLO and inferencer.model_path:
            inferencer.active_model_type = ModelType.YOLO
            self.app_state.auto_detect = True
        elif inferencer.model_path:
            inferencer.active_model_type = ModelType.YOLO
            self.app_state.auto_detect = True
        elif inferencer.sam_model_path:
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
        """Set active model and sync UI."""
        inferencer.set_active_model(model_type, model_path)
        if model_type == ModelType.YOLO:
            self.use_yolo_action.setChecked(True)
        elif model_type == ModelType.SAM3:
            self.use_sam_action.setChecked(True)
        settings.models.active_model = model_type
        save_settings()
        self.statusbar.showMessage(f"Active model: {model_type}")

    def resetStates(self):
        g_param.auto_save_counter = 0
        self.play_state = PlayState.STOP
        self.image_widget.clearBboxes()

    def select_model(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Open YOLO Model File", "", "Model Files (*.pt)"
        )
        if model_path:
            settings.models.model_path = model_path
            self._set_model(ModelType.YOLO, model_path)

    def select_sam_model(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Open SAM3 Model File", "", "Model Files (*.pt)"
        )
        if model_path:
            settings.models.sam3_model_path = model_path
            self._set_model(ModelType.SAM3, model_path)

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
        # 設定影片播放位置 (以毫秒為單位)
        if self.image_widget.cap:
            self.progress_bar.blockSignals(True)
            self.image_widget.cap.set(cv2.CAP_PROP_POS_MSEC, position)
            self.progress_bar.blockSignals(False)
            self.progress_bar.setValue(position)
            self.image_widget.update()

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
        """更新選取中的bbox或polygon的標籤名稱"""
        iw = self.image_widget
        label = self.app_state.last_used_label

        # 優先處理SELECT模式下的選取物件
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

    def promptInputLabel(self):
        """彈出輸入框，讓使用者輸入標籤名稱"""
        iw = self.image_widget
        current_label = self.app_state.last_used_label

        # 取得目前選取物件的label作為預設值
        if iw.select_type == "polygon" and 0 <= iw.idx_focus_polygon < len(iw.polygons):
            current_label = iw.polygons[iw.idx_focus_polygon].label
        elif iw.select_type == "bbox" and 0 <= iw.idx_focus_bbox < len(iw.bboxes):
            current_label = iw.bboxes[iw.idx_focus_bbox].label

        label, ok = QInputDialog.getText(
            self, "Input", "Enter label name:", text=current_label
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
            self.app_state.toggle_auto_detect()
        elif event.key() == Qt.Key.Key_Delete:
            # SELECT模式下先嘗試刪除選取的標註
            if self.image_widget.drawing_mode == DrawingMode.SELECT:
                if self.image_widget.deleteSelectedAnnotation():
                    self.statusbar.showMessage("已刪除選取的標註")
                    return
            self.deletePairOfImgXml()
        elif event.key() == Qt.Key.Key_Space:
            self.toggle_play_pause()
        elif event.key() == Qt.Key.Key_L:
            self.promptInputLabel()
        elif event.key() == Qt.Key.Key_P:
            self.polygon_mode_action.setChecked(True)
            self.image_widget.set_drawing_mode(DrawingMode.POLYGON)
        elif event.key() == Qt.Key.Key_G:
            self.image_widget.runInference()
        elif event.key() == Qt.Key.Key_V:
            self.cycle_view_mode()

        elif event.key() in [
            Qt.Key.Key_1,
            Qt.Key.Key_2,
            Qt.Key.Key_3,
            Qt.Key.Key_4,
            Qt.Key.Key_5,
            Qt.Key.Key_6,
            Qt.Key.Key_7,
            Qt.Key.Key_8,
            Qt.Key.Key_9,
            Qt.Key.Key_0,
        ]:
            key_val = event.key() - Qt.Key.Key_1 + 1
            self.app_state.last_used_label = self.app_state.get_label_by_key(
                str(key_val)
            )
            self.updateFocusedAnnotation()
            self.statusBar().showMessage(f"labels: {self.app_state.preset_labels}")

    def closeEvent(self, event):
        """
        關閉前需要儲存最後的標籤資訊, 並更新動態設定檔
        """
        if self.app_state.auto_save or g_param.user_labeling:
            self.saveImgAndLabels()
        save_settings()

    def convert_voc_to_yolo(self):
        """
        將 VOC XML 格式的標註檔案轉換為 YOLO TXT 標籤檔
        """
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select folder containing VOC XML files"
        )
        if folder_path:
            output_folder = Path(
                folder_path, YOLO_LABELS_FOLDER
            )  # 在同資料夾下建立 yolo labels 資料夾
            output_folder.mkdir(parents=True, exist_ok=True)
            file_h.convertVocInFolder(folder_path, output_folder, self.app_state)
            self.statusbar.showMessage(
                f"VOC to YOLO conversion completed in folder: {output_folder}"
            )

    def edit_categories(self):
        dialog = CategorySettingsDialog(self)
        if dialog.exec():
            self.statusbar.showMessage("Categories設定已儲存")
            save_settings()

    def edit_text_prompts(self):
        dialog = TextPromptsDialog(self)
        if dialog.exec():
            save_settings()
            self.statusbar.showMessage(f"Text prompts: {settings.class_names.text_prompts}")

    def edit_param(self):
        dialog = ParamDialog(self)
        if dialog.exec():
            save_settings()
            self.statusbar.showMessage(
                f"Polygon tolerance: {settings.models.polygon_tolerance}"
            )

    def show_convert_settings(self):
        """顯示轉換設定對話框"""
        dialog = ConvertSettingsDialog(self, self.app_state)
        if dialog.exec():
            self.statusbar.showMessage("轉換設定已儲存")

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
