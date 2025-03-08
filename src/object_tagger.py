import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import yaml
from PyQt6.QtCore import QAbstractListModel, QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QSlider,
    QStatusBar,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
# from PyQt6.QtMultimediaWidgets import QVideoWidget
from ultralytics import YOLO

from src.model import Bbox, FileType

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".wmv", ".mkv", ".webm")
ALL_EXTS = IMAGE_EXTS + VIDEO_EXTS


def getXmlPath(image_path):
    path_tmp = Path(image_path)
    return path_tmp.parent / f"{path_tmp.stem}.xml"


class ShowImageCmd:
    NEXT = "next"
    PREV = "prev"
    FIRST = "first"
    LAST = "last"


class DynamicConfig:
    """
    有需要的話可以定義樹狀結構
    """

    dict_data = None


class MainWindow(QMainWindow):
    def __init__(self):

        super().__init__()

        self.setWindowTitle("Image Tagger")

        # 設定最小尺寸
        self.setMinimumSize(500, 500)

        # 自動
        self.auto_save = False
        self.auto_detect = False

        # 自動儲存
        self.auto_save_action = QAction("&Auto Save", self)
        self.auto_save_action.setCheckable(True)
        self.auto_save_action.triggered.connect(self.toggle_auto_save)

        # 自動使用偵測 (GPU不好速度就會慢)
        self.auto_detect_action = QAction("&Auto Detect", self)
        self.auto_detect_action.setCheckable(True)
        self.auto_detect_action.triggered.connect(self.toggle_auto_detect)

        # 檔案相關動作
        self.open_folder_action = QAction("&Open Folder", self)
        self.open_folder_action.triggered.connect(self.open_folder)

        # 工具列
        self.toolbar = QToolBar()
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.toolbar)

        self.toolbar_auto_save = QAction("&Auto Save", self)
        self.toolbar_auto_save.triggered.connect(self.toggle_auto_save)
        self.toolbar.addAction(self.auto_save_action)

        self.toolbar_auto_detect = QAction("&Auto Detect", self)
        self.toolbar_auto_detect.triggered.connect(self.toggle_auto_detect)
        self.toolbar.addAction(self.auto_detect_action)

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

        # 建立 Bounding Box 列表
        # self.bbox_list_view = QListView()
        # self.bbox_list_model = BboxListModel([])
        # self.bbox_list_view.setModel(self.bbox_list_model)
        # self.main_layout.addWidget(self.bbox_list_view)

        self.save_action = QAction("&Save", self)
        self.save_action.triggered.connect(self.save_annotations)
        self.toolbar.addAction(self.save_action)

        # 播放控制
        self.play_pause_action = QAction("", self)
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
        self.play_pause_action.setIcon(icon)
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

        # 退出
        self.quit_action = QAction("&Quit", self)
        self.quit_action.triggered.connect(self.close)

        # 主選單
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("&File")
        # self.edit_menu = self.menu.addMenu("&Edit")
        self.ai_menu = self.menu.addMenu("&Ai")
        # self.view_menu = self.menu.addMenu("&View")
        # self.help_menu = self.menu.addMenu("&Help")

        self.open_file_by_index_action = QAction("&Open File by Index", self)
        self.open_file_by_index_action.triggered.connect(self.open_file_by_index)

        self.file_menu.addAction(self.open_folder_action)
        self.file_menu.addAction(self.open_file_by_index_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_action)
        self.file_menu.addAction(self.auto_save_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.quit_action)

        self.ai_menu.addAction(self.auto_detect_action)

        # 檔案處理器
        self.file_handler = FileHandler()

        # 讀取預設標籤和上次使用的標籤
        self.preset_labels = {}
        self.last_used_label = "object"  # 預設值
        try:
            with open("config/static.yaml", "r") as f:
                config = yaml.safe_load(f)
                yaml_labels = config.get("labels", {})
                if yaml_labels:
                    self.preset_labels = {
                        str(key): value
                        for key, value in yaml_labels.items()
                        if isinstance(key, (int, str))
                    }
                self.last_used_label = config.get("default_label", "object")

        except FileNotFoundError:
            QMessageBox.warning(self, "Warning", "config/static.yaml not found.")
        except yaml.YAMLError as e:
            QMessageBox.warning(
                self, "Warning", f"Error parsing config/static.yaml: {e}"
            )

        try:
            with open("config/dynamic.yaml", "r") as f:
                DynamicConfig.dict_data = yaml.safe_load(f)
            model_path = DynamicConfig.dict_data.get("model_path", None)
            if model_path:
                self.load_model(model_path)
            folder_path = DynamicConfig.dict_data.get("folder_path", None)
            self.choose_folder(folder_path)
        except FileNotFoundError:
            # QMessageBox.warning(self, "Warning", "config/dynamic.yaml not found.")
            pass
        except yaml.YAMLError as e:
            QMessageBox.warning(
                self, "Warning", f"Error parsing config/label.yaml: {e}"
            )

    def select_model(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Open Model File", "", "Model Files (*.pt)"
        )
        if model_path:
            self.load_model(model_path)
            # update config
            DynamicConfig.dict_data["model_path"] = model_path
            with open("config/dynamic.yaml", "w") as f:
                yaml.dump(DynamicConfig.dict_data, f)

    def load_model(self, model_path):
        try:
            if self.image_widget.model:
                del self.image_widget.model
            self.image_widget.model = YOLO(model_path)
            self.statusbar.showMessage(f"Model loaded: {model_path}")
            self.image_widget.use_model = True
            self.auto_detect = True
            self.auto_detect_action.setChecked(self.auto_detect)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def open_file_by_index(self):
        if not self.file_handler.folder_path:
            QMessageBox.warning(self, "Warning", "No folder opened.")
            return

        if not self.file_handler.image_files:
            QMessageBox.warning(self, "Warning", "No files in the folder.")
            return

        num_files = len(self.file_handler.image_files)
        index, ok = QInputDialog.getInt(
            self,
            "Open File by Index",
            f"Enter file index (1-{num_files}):",
            1,  # 預設值
            1,  # 最小值
            num_files,  # 最大值
        )

        if ok:
            self.file_handler.current_index = index - 1
            self.image_widget.load_image(self.file_handler.current_image_path())
            self.statusbar.showMessage(
                f"Image: {self.file_handler.current_image_path()} "
                f"[{self.file_handler.current_index + 1} / {len(self.file_handler.image_files)}]"
            )

    def open_folder(self):
        """
        用pyqt瀏覽並選定資料夾
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Open Folder")  # PyQt6
        self.choose_folder(folder_path)
        # update config
        DynamicConfig.dict_data["folder_path"] = folder_path
        with open("config/dynamic.yaml", "w") as f:
            yaml.dump(DynamicConfig.dict_data, f)

    def choose_folder(self, folder_path):
        """
        開啟資料夾的檔案
        """
        if folder_path:
            self.file_handler.load_folder(folder_path)
            if self.file_handler.image_files:
                self.image_widget.load_image(self.file_handler.current_image_path())
                self.statusbar.showMessage(
                    f"Opened folder: {folder_path} "
                    f"[{self.file_handler.current_index + 1} / {len(self.file_handler.image_files)}]"
                )

    def show_image(self, cmd: str):
        if self.is_auto_save():
            self.save_annotations()
        if self.file_handler.show_image(cmd):
            self.image_widget.load_image(self.file_handler.current_image_path())
            self.statusbar.showMessage(
                f"Image: {self.file_handler.current_image_path()}"
                f"[{self.file_handler.current_index + 1} / {len(self.file_handler.image_files)}]"
            )

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

    def is_auto_save(self):
        return self.auto_save

    def is_auto_detect(self):
        return self.auto_detect

    def save_annotations(self):
        """
        儲存標記, 注意就算沒有bbox也是要儲存, 表示有處理過, 也能讓trainer知道這是在訓練背景
        """
        if self.file_handler.current_image_path():
            if self.image_widget.file_type == FileType.VIDEO:
                # 儲存影片當前幀和對應的 XML
                frame = self.image_widget.pixmap.toImage()  # 從 pixmap 取得
                original_filename = os.path.basename(
                    self.file_handler.current_image_path()
                )
                filename_no_ext = os.path.splitext(original_filename)[0]
                frame_number = int(self.image_widget.cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame_filename = f"{filename_no_ext}_frame{frame_number}.jpg"
                frame_path = os.path.join(self.file_handler.folder_path, frame_filename)
                frame.save(frame_path)

                xml_path = getXmlPath(frame_path)
                bboxes = self.image_widget.bboxes
                xml_content = self.file_handler.generate_voc_xml(bboxes, frame_path)
                with open(xml_path, "w") as f:
                    f.write(xml_content)

                if self.is_auto_save():
                    self.statusbar.showMessage(f"Annotations auto saved to {xml_path}")
                else:
                    self.statusbar.showMessage(f"Annotations saved to {xml_path}")

            else:
                # 儲存圖片的 XML
                file_path = getXmlPath(self.file_handler.current_image_path())
                bboxes = self.image_widget.bboxes
                xml_content = self.file_handler.generate_voc_xml(
                    bboxes, self.file_handler.current_image_path()
                )
                with open(file_path, "w") as f:
                    f.write(xml_content)
                if self.is_auto_save():
                    self.statusbar.showMessage(f"Annotations auto saved to {file_path}")
                else:
                    self.statusbar.showMessage(f"Annotations saved to {file_path}")

    def toggle_play_pause(self):
        if self.image_widget.file_type != FileType.VIDEO:
            self.image_widget.is_playing = False
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            self.play_pause_action.setIcon(icon)
            return
        self.image_widget.is_playing = not self.image_widget.is_playing
        if self.image_widget.is_playing:
            # 播放前先儲存標籤資訊
            if self.is_auto_save():
                self.save_annotations()

            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            self.play_pause_action.setIcon(icon)
            self.image_widget.timer.start(self.refresh_interval)
        else:
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            self.play_pause_action.setIcon(icon)
            # self.image_widget.detectImage()
            self.image_widget.timer.stop()

    def set_media_position(self, position):
        # 設定影片播放位置 (以毫秒為單位)
        if self.image_widget.cap:
            self.progress_bar.blockSignals(True)
            self.image_widget.cap.set(cv2.CAP_PROP_POS_MSEC, position)
            self.progress_bar.blockSignals(False)

    def set_playback_speed(self, index):
        # 設定播放速度
        speed = self.speed_control.itemData(index)
        self.image_widget.timer.stop()
        if self.image_widget.fps:
            fps = self.image_widget.fps
        else:
            fps = 30
        self.refresh_interval = int(fps / speed)
        if self.image_widget.is_playing:
            self.image_widget.timer.start(self.refresh_interval)  # 重新啟動定時器

    def updateLastBbox(self):
        """
        更新最後使用的標籤
        """
        if self.image_widget.bboxes:
            self.image_widget.bboxes[-1].label = self.last_used_label
        self.image_widget.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Right or event.key() == Qt.Key.Key_PageDown:
            self.show_image(ShowImageCmd.NEXT)
        elif event.key() == Qt.Key.Key_Left or event.key() == Qt.Key.Key_PageUp:
            self.show_image(ShowImageCmd.PREV)
        elif event.key() == Qt.Key.Key_Home:
            self.show_image(ShowImageCmd.FIRST)
        elif event.key() == Qt.Key.Key_Home:
            self.show_image(ShowImageCmd.LAST)
        elif event.key() == Qt.Key.Key_Q:
            self.close()
        elif event.key() == Qt.Key.Key_A:
            self.toggle_auto_save()
        elif event.key() == Qt.Key.Key_Space:
            self.toggle_play_pause()
        elif event.key() == Qt.Key.Key_L:
            # 彈出輸入框，讓使用者輸入標籤
            label, ok = QInputDialog.getText(
                self, "Input", "Enter label name:", text=self.last_used_label
            )
            if ok and label.strip():
                self.last_used_label = label.strip()  # 更新上次使用的標籤
            self.updateLastBbox()

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
            self.updateLastBbox()
            self.statusBar().showMessage(f"labels: {self.preset_labels}")


class ImageWidget(QWidget):
    def __init__(self, main_window: MainWindow):
        super().__init__()
        self.main_window = main_window
        self.image_label = QLabel()
        self.pixmap = None
        self.bboxes: list[Bbox] = []
        self.start_pos = None
        self.end_pos = None
        self.drawing = False
        self.resizing = False
        self.resizing_bbox = None
        self.resizing_corner = None
        self.original_bbox = None  # 儲存原始 bbox 資訊

        self.model: None | YOLO = None
        self.use_model = False

        # 角落大小
        self.CORNER_SIZE = 10
        self.cv_img = None
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )  # 設定大小策略

        # # 影片播放相關
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        self.is_playing = False  # 追蹤是否在播放影片

        # 縮放後的影像尺寸
        self.scaled_width = None
        self.scaled_height = None
        self.cap = None
        # (x, y)--->┌－－－－－－－┐ ╮
        #           │             │ │
        #           │<---width--->│  height
        #           │             │ │
        #           └－－－－－－－┘ ╯

    def _update_frame(self):
        if self.is_playing and self.cap:
            ret, self.cv_img = self.cap.read()
            if not ret:
                self.timer.stop()
                return

            height, width, channel = self.cv_img.shape
            bytesPerLine = 3 * width
            qImg = QImage(
                self.cv_img.data,
                width,
                height,
                bytesPerLine,
                QImage.Format.Format_RGB888,
            ).rgbSwapped()
            self.pixmap = QPixmap.fromImage(qImg)

            self.detectImage()

            position = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.main_window.progress_bar.blockSignals(True)  # 暫時阻止信號傳遞
            self.main_window.progress_bar.setValue(int(position))
            self.main_window.progress_bar.blockSignals(False)  # 恢復信號傳遞

            self.update()

    def _scale_to_original(self, point):
        if self.pixmap:
            scale_x = self.pixmap.width() / self.scaled_width
            scale_y = self.pixmap.height() / self.scaled_height
            return QPoint(int(point.x() * scale_x), int(point.y() * scale_y))
        else:
            return point

    def _scale_to_widget(self, point):
        if self.pixmap:
            scale_x = self.scaled_width / self.pixmap.width()
            scale_y = self.scaled_height / self.pixmap.height()
            return QPoint(int(point.x() * scale_x), int(point.y() * scale_y))
        else:
            return point

    def _is_in_corner(self, pos, bbox):
        """檢查滑鼠是否在角落"""
        # 將視窗座標轉換為原始影像座標
        pos = self._scale_to_original(pos)

        x1, y1 = bbox.x, bbox.y
        x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height

        # 計算四個角落的範圍
        corners = {
            "top_left": QRect(
                x1 - self.CORNER_SIZE,
                y1 - self.CORNER_SIZE,
                self.CORNER_SIZE * 2,
                self.CORNER_SIZE * 2,
            ),
            "top_right": QRect(
                x2 - self.CORNER_SIZE,
                y1 - self.CORNER_SIZE,
                self.CORNER_SIZE * 2,
                self.CORNER_SIZE * 2,
            ),
            "bottom_left": QRect(
                x1 - self.CORNER_SIZE,
                y2 - self.CORNER_SIZE,
                self.CORNER_SIZE * 2,
                self.CORNER_SIZE * 2,
            ),
            "bottom_right": QRect(
                x2 - self.CORNER_SIZE,
                y2 - self.CORNER_SIZE,
                self.CORNER_SIZE * 2,
                self.CORNER_SIZE * 2,
            ),
        }

        for corner, rect in corners.items():
            if rect.contains(pos):
                return corner
        return None

    def loadBboxFromXml(self, xml_path) -> bool:
        """
        讀取xml的bbox資訊

        Args:
            xml_path (str): xml檔案路徑

        Returns:
            bool: 是否有bbox
        """
        if os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    name = obj.find("name").text
                    bndbox = obj.find("bndbox")
                    xmin = int(bndbox.find("xmin").text)
                    ymin = int(bndbox.find("ymin").text)
                    xmax = int(bndbox.find("xmax").text)
                    ymax = int(bndbox.find("ymax").text)
                    confidence = float(bndbox.find("confidence").text)
                    width = xmax - xmin
                    height = ymax - ymin
                    self.bboxes.append(
                        Bbox(xmin, ymin, width, height, name, confidence)
                    )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to parse XML: {e}")

            if self.bboxes:
                return True
            else:
                return False

    def detectImage(self):
        """
        執行物件偵測, 只有在model讀取且is_auto_detect==True時才會執行
        執行前先清除bboxes, 以免搞混
        """
        if not self.main_window.is_auto_detect():
            return

        self.bboxes = []
        if self.model:
            results = self.model.predict(self.cv_img)
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        b = box.xyxy[
                            0
                        ]  # get box coordinates in (top, left, bottom, right) format
                        c = box.cls
                        conf = box.conf
                        label = self.model.names[int(c)]
                        self.bboxes.append(
                            Bbox(
                                int(b[0]),
                                int(b[1]),
                                int(b[2] - b[0]),
                                int(b[3] - b[1]),
                                label,
                                float(conf),
                            )
                        )

    def get_total_msec(self):
        # 取得影片總毫秒數
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(total_frames * 1000 / self.fps)

    def load_image(self, file_path):
        # 判斷檔案是否為影片
        if file_path.lower().endswith(VIDEO_EXTS):
            self.file_type = FileType.VIDEO

            self.is_playing = False
            self.cap = cv2.VideoCapture(file_path)
            ret, self.cv_img = self.cap.read()
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.main_window.progress_bar.setRange(0, self.get_total_msec())

            self.main_window.progress_bar.blockSignals(True)
            self.main_window.progress_bar.setValue(0)
            self.main_window.progress_bar.blockSignals(False)
        else:
            self.file_type = FileType.IMAGE
            self.cv_img = cv2.imread(file_path)

        height, width, channel = self.cv_img.shape
        bytesPerLine = 3 * width
        qImg = QImage(
            self.cv_img.data, width, height, bytesPerLine, QImage.Format.Format_RGB888
        ).rgbSwapped()
        self.pixmap = QPixmap.fromImage(qImg)
        self.bboxes = []  # 清空 Bounding Box

        # 嘗試讀取 XML 檔案
        # TODO: 加上影片的frame number
        xml_path = getXmlPath(file_path)
        if not self.loadBboxFromXml(xml_path):
            # 如果 bbox (來自xml) 不存在, 才嘗試使用 YOLO 偵測
            self.detectImage()
        self.update()  # 觸發 paintEvent

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.pixmap:
            # 計算繪製區域，將縮放後的影像置於左上
            scaled_pixmap = self.pixmap.scaled(
                self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatio
            )

            # 計算縮放後的影像尺寸, 之後都幾乎以這兩個為參考
            self.scaled_width = scaled_pixmap.width()
            self.scaled_height = scaled_pixmap.height()

            painter.drawPixmap(0, 0, scaled_pixmap)

            # 繪製 Bounding Box
            for bbox in self.bboxes:
                if hasattr(bbox, "label_color") and bbox.label_color == "red":
                    pen = QPen(QColor(255, 0, 0), 2)  # 紅色
                else:
                    pen = QPen(QColor(0, 255, 0), 2)  # 綠色，寬度 2
                painter.setPen(pen)
                rect = QRect(
                    self._scale_to_widget(QPoint(bbox.x, bbox.y)),
                    self._scale_to_widget(
                        QPoint(bbox.x + bbox.width, bbox.y + bbox.height)
                    ),
                )
                painter.drawRect(rect)

                # 計算文字大小
                text = f"{bbox.label} ({bbox.confidence:.2f})"
                font_metrics = painter.fontMetrics()
                text_width = font_metrics.horizontalAdvance(text) * 2
                text_height = font_metrics.height()

                # 繪製文字底色 (調整位置和大小)
                qpt_text = QPoint(bbox.x, bbox.y - text_height)
                bg_rect = QRect(
                    self._scale_to_widget(
                        QPoint(qpt_text.x(), qpt_text.y() - int(text_height))
                    ),
                    QPoint(
                        self._scale_to_widget(qpt_text).x()
                        + int(text_width * self.scaled_width / self.pixmap.width()),
                        self._scale_to_widget(qpt_text).y()
                        + int(text_height * self.scaled_height / self.pixmap.height()),
                    ),
                )
                painter.fillRect(bg_rect, QColor(0, 0, 0, 150))  # 黑色半透明底色

                # 繪製文字 (調整位置)
                painter.drawText(
                    self._scale_to_widget(QPoint(bbox.x, bbox.y - text_height)), text
                )

            if self.drawing:
                pen = QPen(QColor(255, 0, 0), 2)  # 繪製中的 Bounding Box 用紅色
                painter.setPen(pen)
                rect = QRect(self.start_pos, self.end_pos)
                painter.drawRect(rect)

    def mousePressEvent(self, event):
        if self.is_playing:
            self.is_playing = False

        if event.button() == Qt.MouseButton.LeftButton:
            for bbox in self.bboxes:
                corner = self._is_in_corner(event.pos(), bbox)
                if corner:
                    self.resizing_bbox = bbox
                    self.resizing_corner = corner
                    self.original_bbox = (
                        bbox.x,
                        bbox.y,
                        bbox.width,
                        bbox.height,
                    )  # 儲存原始大小
                    self.start_pos = self._scale_to_original(
                        event.pos()
                    )  # 紀錄原始座標
                    self.resizing_bbox.label_color = "red"  # 改變顏色
                    self.resizing = True
                    break
            else:
                self.start_pos = event.pos()
                self.end_pos = event.pos()
                self.drawing = True

        elif event.button() == Qt.MouseButton.RightButton:  # 刪除
            pos = event.pos()
            for bbox in reversed(self.bboxes):  # 從後面開始找，避免 index 錯誤
                # 將原始影像座標轉換為視窗座標
                rect = QRect(
                    self._scale_to_widget(QPoint(bbox.x, bbox.y)),
                    self._scale_to_widget(
                        QPoint(bbox.x + bbox.width, bbox.y + bbox.height)
                    ),
                )
                if rect.contains(pos):
                    self.bboxes.remove(bbox)
                    self.update()
                    break

    def mouseMoveEvent(self, event):
        # 檢查是否在角落
        cursor_changed = False
        for bbox in self.bboxes:
            corner = self._is_in_corner(event.pos(), bbox)
            if corner:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                cursor_changed = True
                break
        if not cursor_changed:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        if self.drawing:
            self.end_pos = event.pos()
            self.update()
        elif self.resizing:
            pos = self._scale_to_original(event.pos())
            dx = pos.x() - self.start_pos.x()
            dy = pos.y() - self.start_pos.y()

            # 根據不同的角落調整大小
            if self.resizing_corner == "top_left":
                self.resizing_bbox.x = self.original_bbox[0] + dx
                self.resizing_bbox.y = self.original_bbox[1] + dy
                self.resizing_bbox.width = self.original_bbox[2] - dx
                self.resizing_bbox.height = self.original_bbox[3] - dy
            elif self.resizing_corner == "top_right":
                self.resizing_bbox.y = self.original_bbox[1] + dy
                self.resizing_bbox.width = self.original_bbox[2] + dx
                self.resizing_bbox.height = self.original_bbox[3] - dy
            elif self.resizing_corner == "bottom_left":
                self.resizing_bbox.x = self.original_bbox[0] + dx
                self.resizing_bbox.width = self.original_bbox[2] - dx
                self.resizing_bbox.height = self.original_bbox[3] + dy
            elif self.resizing_corner == "bottom_right":
                self.resizing_bbox.width = self.original_bbox[2] + dx
                self.resizing_bbox.height = self.original_bbox[3] + dy

            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.resizing:
                x1, y1, width, height = (
                    self.resizing_bbox.x,
                    self.resizing_bbox.y,
                    self.resizing_bbox.width,
                    self.resizing_bbox.height,
                )
                x2, y2 = x1 + width, y1 + height
                self.resizing = False
                self.resizing_bbox.label_color = "green"

                self.resizing_bbox = None
                self.resizing_corner = None
                self.original_bbox = None

                self.update()

            elif self.drawing:
                self.drawing = False
                # 取得座標 (視窗座標)

                x1, y1 = self.start_pos.x(), self.start_pos.y()
                x2, y2 = self.end_pos.x(), self.end_pos.y()

                # x1與y1永遠都是最小
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1

                # 取得寬高 (視窗座標)
                width = abs(x2 - x1)
                height = abs(y2 - y1)

                # 檢查寬高是否大於最小限制
                if width < 5 or height < 5:
                    return

                # 將視窗座標轉換為原始影像座標
                x1_original, y1_original = (
                    self._scale_to_original(QPoint(x1, y1)).x(),
                    self._scale_to_original(QPoint(x1, y1)).y(),
                )
                width_original, height_original = int(
                    width * self.pixmap.width() / self.scaled_width
                ), int(height * self.pixmap.height() / self.scaled_height)
                # 建立 Bbox 物件 (使用原始影像座標)
                self.bboxes.append(
                    Bbox(
                        min(x1_original, x1_original + width_original),
                        min(y1_original, y1_original + height_original),
                        width_original,
                        height_original,
                        self.main_window.last_used_label,
                        1.0,
                    )
                )
                self.update()

    def wheelEvent(self, event):
        if self.main_window.is_auto_save():
            self.main_window.save_annotations()
        if event.angleDelta().y() > 0:
            self.main_window.prev_image()
        else:
            self.main_window.next_image()


class BboxListModel(QAbstractListModel):
    def __init__(self, bboxes, parent=None):
        super().__init__(parent)
        self.bboxes = bboxes

    def rowCount(self, parent=None):
        return len(self.bboxes)

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            bbox = self.bboxes[index.row()]
            return f"{bbox.label} ({bbox.x}, {bbox.y}, {bbox.width}, {bbox.height})"
        return None


class FileHandler:
    def __init__(self):
        self.image_files = []
        self.current_index = 0
        self.folder_path = ""

    def load_folder(self, folder_path):
        self.folder_path = folder_path
        self.image_files = []
        self.current_index = 0
        for file in os.listdir(folder_path):
            if file.lower().endswith(ALL_EXTS):
                self.image_files.append(file)
        self.image_files.sort()  # 排序

    def current_image_path(self):
        if self.image_files:
            return os.path.join(self.folder_path, self.image_files[self.current_index])
        return None

    def show_image(self, cmd: str):
        """
        show [next, prev, first, last] image
        Args:
            cmd: one of "next", "prev", "first", "last"

        Returns:
            True if file is changed
        """
        if cmd == ShowImageCmd.NEXT:
            if self.current_index < len(self.image_files) - 1:
                self.current_index += 1
                return True
        elif cmd == ShowImageCmd.PREV:
            if self.current_index > 0:
                self.current_index -= 1
                return True
        elif cmd == ShowImageCmd.FIRST:
            if self.current_index != 0:
                self.current_index = 0
                return True
        elif cmd == ShowImageCmd.LAST:
            if self.current_index != len(self.image_files) - 1:
                self.current_index = len(self.image_files) - 1
                return True
        return False

    def generate_voc_xml(self, bboxes, image_path):
        image_filename = os.path.basename(image_path)
        folder_name = os.path.basename(os.path.dirname(image_path))

        xml_str = "<annotation>\n"
        xml_str += f"    <folder>{folder_name}</folder>\n"
        xml_str += f"    <filename>{image_filename}</filename>\n"
        # xml_str += f"    <path>{image_path}</path>\n"
        # xml_str += "    <source>\n        <database>Unknown</database>\n    </source>\n"

        # 讀取圖片大小
        img = cv2.imread(image_path)
        height, width, depth = img.shape

        xml_str += f"    <size>\n        <width>{width}</width>\n        <height>{height}</height>\n    </size>\n"

        for bbox in bboxes:
            xml_str += "    <object>\n"
            xml_str += f"        <name>{bbox.label}</name>\n"
            xml_str += "        <bndbox>\n"
            xml_str += f"            <xmin>{bbox.x}</xmin>\n"
            xml_str += f"            <ymin>{bbox.y}</ymin>\n"
            xml_str += f"            <xmax>{bbox.x + bbox.width}</xmax>\n"
            xml_str += f"            <ymax>{bbox.y + bbox.height}</ymax>\n"
            xml_str += f"            <confidence>{bbox.confidence}</confidence>\n"
            xml_str += "        </bndbox>\n"
            xml_str += "    </object>\n"

        xml_str += "</annotation>\n"
        return xml_str


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
