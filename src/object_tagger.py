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
from ultralytics import YOLO

from src.config import cfg
from src.utils.func import getMaskPath, getXmlPath
from src.image_widget import DrawingMode, ImageWidget
from src.utils.loglo import getUniqueLogger
from src.utils.model import FileType, PlayState, ShowImageCmd
from src.utils.dialogs import CategorySettingsDialog
from src.utils.dynamic_settings import save_settings, settings
from src.utils.file_handler import file_h
from src.utils.global_param import g_param

log = getUniqueLogger(__file__)
yaml = YAML()

YOLO_LABELS_FOLDER = "labels"
DEFAULT_DETECT_MODEL = "yolov8n.pt"


class MainWindow(QMainWindow):
    def __init__(self):

        super().__init__()

        self.setWindowTitle("Image Tagger")

        # 設定最小尺寸
        self.setMinimumSize(500, 500)

        # 自動
        self.auto_save = False
        self.auto_detect = False

        # 儲存
        self.save_action = QAction("Save", self)
        self.save_action.triggered.connect(self.saveImgAndLabels)

        self.auto_save_action = QAction("Auto Save", self)
        self.auto_save_action.setCheckable(True)
        self.auto_save_action.triggered.connect(self.toggle_auto_save)

        self.save_mask_action = QAction("Save &Mask", self)
        self.save_mask_action.triggered.connect(self.saveMask)

        self.show_fps = False

        # 自動使用偵測 (GPU不好速度就會慢)
        self.auto_detect_action = QAction("Auto Detect", self)
        self.auto_detect_action.setCheckable(True)
        self.auto_detect_action.triggered.connect(self.toggle_auto_detect)

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

        self.image_widget = ImageWidget(self)
        self.main_layout.addWidget(self.image_widget)

        # 工具列
        self.toolbar = QToolBar()
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.toolbar)

        self.toolbar_auto_save = QAction("Auto Save", self)
        self.toolbar_auto_save.triggered.connect(self.toggle_auto_save)
        self.toolbar.addAction(self.auto_save_action)

        self.toolbar_auto_detect = QAction("Auto Detect", self)
        self.toolbar_auto_detect.triggered.connect(self.toggle_auto_detect)
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

        self.bbox_mode_action = QAction("BBox", self)
        self.bbox_mode_action.setCheckable(True)
        self.bbox_mode_action.setChecked(True)
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
        self.detect_action.triggered.connect(self.image_widget.detectImage)

        # 主選單
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")
        self.edit_menu = self.menu.addMenu("Edit")
        self.ai_menu = self.menu.addMenu("Ai")
        self.convert_menu = self.menu.addMenu("&Convert")
        # self.view_menu = self.menu.addMenu("&View")
        # self.help_menu = self.menu.addMenu("&Help")

        self.edit_categories_action = QAction("&Edit Categories", self)
        self.edit_categories_action.triggered.connect(self.edit_categories)

        self.convert_voc_yolo_action = QAction("&VOC to YOLO", self)
        self.convert_voc_yolo_action.triggered.connect(self.convert_voc_to_yolo)

        self.convert_menu.addAction(self.edit_categories_action)
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

        self.select_model_action = QAction("&Select Model", self)
        self.select_model_action.triggered.connect(self.select_model)

        self.use_default_model_action = QAction("&Use default Model", self)
        self.use_default_model_action.triggered.connect(self.use_default_model)

        self.ai_menu.addAction(self.use_default_model_action)
        self.ai_menu.addAction(self.select_model_action)
        self.ai_menu.addSeparator()
        self.ai_menu.addAction(self.detect_action)
        self.ai_menu.addAction(self.auto_detect_action)

        # 讀取預設標籤和上次使用的標籤
        self.preset_labels = {}
        self.last_used_label = "object"  # 預設值

        if settings.model_path:
            self.load_model(settings.model_path)
        self.choose_folder(settings.folder_path, settings.file_index)

        try:
            with open("cfg/system.yaml", "r", encoding="utf-8") as f:
                config = yaml.load(f)
            yaml_labels = config.get("labels", {})
            if yaml_labels:
                self.preset_labels = {
                    str(key): value
                    for key, value in yaml_labels.items()
                    if isinstance(key, (int, str))
                }
            self.last_used_label = config.get("default_label", "object")
            self.show_fps = config.get("show_fps", False)

        except FileNotFoundError:
            QMessageBox.warning(self, "Warning", "config/cfg.yaml not found.")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error parsing config/cfg.yaml: {e}")

    def resetStates(self):
        g_param.auto_save_counter = 0
        self.play_state = PlayState.STOP
        self.image_widget.clearBboxes()

    def use_default_model(self):
        self.load_model(DEFAULT_DETECT_MODEL)
        settings.model_path = DEFAULT_DETECT_MODEL

    def select_model(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Open Ultralytics Model File", "", "Model Files (*.pt)"
        )
        if model_path:
            self.load_model(model_path)
            settings.model_path = model_path

    def load_model(self, model_path):
        try:
            if self.image_widget.model:
                del self.image_widget.model
            self.image_widget.model = YOLO(model_path)
            # self.image_widget.model.to("cuda")
            self.image_widget.use_model = True
            self.auto_detect = True
            self.auto_detect_action.setChecked(self.auto_detect)
            self.image_widget.detectImage()
            self.statusbar.showMessage(f"Model loaded: {model_path}")
            save_settings()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

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
        settings.folder_path = folder_path
        settings.file_index = 0

    def choose_folder(self, folder_path: str, file_index: int = 0):
        """
        開啟資料夾的檔案
        """
        if Path(folder_path).is_dir():
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
                self.statusbar.showMessage(f"No files in the folder {folder_path}")
        else:
            self.statusbar.showMessage(f"Invalid folder path {folder_path}")

    def show_image(self, cmd: str):
        """show下一個影校或影片, 如有自動記錄則要先儲存之前的labels"""
        if self.is_auto_save() or g_param.user_labeling:
            self.saveImgAndLabels()
        self.resetStates()
        if file_h.show_image(cmd):
            self.image_widget.load_image(file_h.current_image_path())
            self.statusbar.showMessage(
                f"[{file_h.current_index + 1} / {len(file_h.image_files)}] "
                f"Image: {file_h.current_image_path()}"
            )
            settings.file_index = file_h.current_index
            save_settings()

    def toggle_auto_save(self):
        self.auto_save = not self.auto_save
        self.auto_save_action.setChecked(self.auto_save)
        self.statusbar.showMessage(f"Auto save: {'on' if self.auto_save else 'off'}")

    def toggle_auto_detect(self):
        self.auto_detect = not self.auto_detect
        self.auto_detect_action.setChecked(self.auto_detect)
        self.statusbar.showMessage(
            f"Auto detect: {'on' if self.auto_detect else 'off'}"
        )
        if self.auto_detect:
            self.image_widget.detectImage()

    def is_auto_save(self):
        return self.auto_save

    def is_auto_detect(self):
        return self.auto_detect

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

            if self.is_auto_detect():
                self.image_widget.detectImage()
            else:
                self.image_widget.update()

            position = self.image_widget.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.progress_bar.blockSignals(True)  # 暫時阻止信號傳遞
            self.progress_bar.setValue(int(position))
            self.progress_bar.blockSignals(False)  # 恢復信號傳遞

            # 自動儲存邏輯
            if self.is_auto_save() and cfg.auto_save_per_second > 0:
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
        xml_content = file_h.generate_voc_xml(bboxes, save_path)
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        g_param.user_labeling = False
        status_message = f"Annotations saved to {xml_path}"
        self.statusbar.showMessage(status_message)

    def saveMask(self):
        current_path = file_h.current_image_path()
        if not current_path:
            return

        if self.image_widget.mask_pixmap:
            mask_path = getMaskPath(current_path).as_posix()
            self.image_widget.mask_pixmap.save(mask_path, "PNG")

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
            if self.is_auto_save() or g_param.user_labeling:
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

    def updateFocusOrLastBbox(self):
        """
        更新focus或最後使用的標籤
        """
        idx = (
            self.image_widget.idx_focus_bbox
            if self.image_widget.idx_focus_bbox >= 0
            and self.image_widget.idx_focus_bbox < len(self.image_widget.bboxes)
            else -1
        )
        if self.image_widget.bboxes:
            self.image_widget.bboxes[idx].label = self.last_used_label
            self.image_widget.update()

    def promptInputLabel(self):
        # 彈出輸入框，讓使用者輸入標籤
        label, ok = QInputDialog.getText(
            self, "Input", "Enter label name:", text=self.last_used_label
        )
        if ok and label.strip():
            self.last_used_label = label.strip()  # 更新上次使用的標籤
            self.updateFocusOrLastBbox()

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
            self.toggle_auto_save()
        elif event.key() == Qt.Key.Key_D:
            self.toggle_auto_detect()
        elif event.key() == Qt.Key.Key_Delete:
            self.deletePairOfImgXml()
        elif event.key() == Qt.Key.Key_Space:
            self.toggle_play_pause()
        elif event.key() == Qt.Key.Key_L:
            self.promptInputLabel()

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
            self.last_used_label = self.preset_labels.get(
                str(key_val), self.last_used_label
            )
            self.updateFocusOrLastBbox()
            self.statusBar().showMessage(f"labels: {self.preset_labels}")

    def closeEvent(self, event):
        """
        關閉前需要儲存最後的標籤資訊, 並更新動態設定檔
        """
        if self.is_auto_save() or g_param.user_labeling:
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
            file_h.convertVocInFolder(folder_path, output_folder)
            self.statusbar.showMessage(
                f"VOC to YOLO conversion completed in folder: {output_folder}"
            )

    def edit_categories(self):
        dialog = CategorySettingsDialog(self)
        if dialog.exec():
            self.statusbar.showMessage("Categories設定已儲存")
            save_settings()

    def cbWheelEvent(self, wheel_up):
        if self.is_auto_save() or g_param.user_labeling:
            self.saveImgAndLabels()
        if wheel_up:
            self.show_image(ShowImageCmd.PREV)
        else:
            self.show_image(ShowImageCmd.NEXT)

    def cbMousePress(self, event):
        if self.play_state == PlayState.PLAY:
            self.play_state = PlayState.PAUSE


def main():
    app = QApplication(sys.argv)
    MainWindow().show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
