import cv2
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QFileDialog,
                             QMenuBar, QToolBar, QStatusBar, QVBoxLayout,
                             QListView, QMessageBox, QInputDialog, QSizePolicy,
                             QSlider, QComboBox, QStyle)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QAction, QEnterEvent, QCursor
from PyQt6.QtCore import Qt, QAbstractListModel, QTimer, QRect, QPoint, QUrl
from ultralytics import YOLO
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget


from src.model import Bbox

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.wmv', '.mkv', '.webm')
ALL_EXTS = IMAGE_EXTS + VIDEO_EXTS


def getXmlPath(image_path):
    path_tmp = Path(image_path)
    return path_tmp.parent / f"{path_tmp.stem}.xml"


class MainWindow(QMainWindow):
    def __init__(self):
        
        super().__init__()

        self.setWindowTitle("Object Tagger")

        # 設定最小尺寸
        self.setMinimumSize(500, 500)

        # 自動儲存
        self.auto_save = False

        # 選單
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("&File")
        self.edit_menu = self.menu.addMenu("&Edit")
        self.ai_menu = self.menu.addMenu("&Ai")
        # self.view_menu = self.menu.addMenu("&View")
        # self.help_menu = self.menu.addMenu("&Help")

        # 退出
        self.quit_action = QAction("&Quit", self)
        self.quit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.quit_action)

        # 自動儲存
        self.auto_save_action = QAction("&Auto Save", self)
        self.auto_save_action.setCheckable(True)
        self.auto_save_action.triggered.connect(self.toggle_auto_save)
        self.edit_menu.addAction(self.auto_save_action)
        
        # 工具列
        self.toolbar = QToolBar()
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.toolbar)

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
        self.main_layout.addWidget(self.image_widget.video_widget) # 將 video_widget 加入 layout

        # 建立 Bounding Box 列表
        # self.bbox_list_view = QListView()
        # self.bbox_list_model = BboxListModel([])
        # self.bbox_list_view.setModel(self.bbox_list_model)
        # self.main_layout.addWidget(self.bbox_list_view)

        self.model_select_action = QAction("&Select Model", self)
        self.model_select_action.triggered.connect(self.select_model)
        self.toolbar.addAction(self.model_select_action)
        # 檔案相關動作
        self.open_folder_action = QAction("&Open Folder", self)
        self.open_folder_action.triggered.connect(self.open_folder)
        # self.file_menu.addAction(self.open_folder_action)
        self.toolbar.addAction(self.open_folder_action)

        self.save_action = QAction("&Save", self)
        self.save_action.triggered.connect(self.save_annotations)
        # self.file_menu.addAction(self.save_action)
        self.toolbar.addAction(self.save_action)

        # 播放控制
        self.play_pause_action = QAction("", self)
        pix_icon = QStyle.StandardPixmap.SP_MediaPause
        icon = self.style().standardIcon(pix_icon)
        self.play_pause_action.setIcon(icon)
        self.play_pause_action.triggered.connect(self.toggle_play_pause)
        self.toolbar.addAction(self.play_pause_action)

        self.progress_bar = QSlider(Qt.Orientation.Horizontal)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.valueChanged.connect(self.set_media_position)
        self.toolbar.addWidget(self.progress_bar)

        self.speed_control = QComboBox()
        self.speed_control.addItem("0.5x", 0.5)
        self.speed_control.addItem("1.0x", 1.0)
        self.speed_control.addItem("1.5x", 1.5)
        self.speed_control.addItem("2.0x", 2.0)
        self.speed_control.setCurrentIndex(1)  # 預設為 1.0x
        self.speed_control.currentIndexChanged.connect(self.set_playback_speed)
        self.toolbar.addWidget(self.speed_control)


        # 檔案處理器
        self.file_handler = FileHandler()

    def select_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, 'Open Model File', '', "Model Files (*.pt)")
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        try:
            if self.image_widget.model:
                del self.image_widget.model
            self.image_widget.model = YOLO(model_path)
            self.statusbar.showMessage(f"Model loaded: {model_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open Folder") #PyQt6
        if folder_path:
            self.file_handler.load_folder(folder_path)
            if self.file_handler.image_files:
                self.image_widget.load_image(self.file_handler.current_image_path())
                self.statusbar.showMessage(f"Opened folder: {folder_path}")

    def next_image(self):
        if self.file_handler.next_image():
            self.image_widget.load_image(self.file_handler.current_image_path())
            self.statusbar.showMessage(f"Image: {self.file_handler.current_image_path()}")

    def prev_image(self):
        if self.file_handler.prev_image():
            self.image_widget.load_image(self.file_handler.current_image_path())
            self.statusbar.showMessage(f"Image: {self.file_handler.current_image_path()}")
    
    def toggle_auto_save(self):
        self.auto_save = not self.auto_save
        self.auto_save_action.setChecked(self.auto_save)
        self.statusbar.showMessage(f"Auto save: {'on' if self.auto_save else 'off'}")

    def is_auto_save(self):
        return self.auto_save

    def save_annotations(self):
        if self.file_handler.current_image_path():
            if self.image_widget.is_playing:
                # 儲存影片當前幀和對應的 XML
                frame = self.image_widget.video_widget.grab()
                original_filename = os.path.basename(self.file_handler.current_image_path())
                filename_no_ext = os.path.splitext(original_filename)[0]
                frame_number = int(self.image_widget.media_player.position() / 1000 * 30)  # 假設 30 fps
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
                xml_content = self.file_handler.generate_voc_xml(bboxes, self.file_handler.current_image_path())
                with open(file_path, "w") as f:
                    f.write(xml_content)
                if self.is_auto_save():
                    self.statusbar.showMessage(f"Annotations auto saved to {file_path}")
                else:
                    self.statusbar.showMessage(f"Annotations saved to {file_path}")

    def toggle_play_pause(self):
        if self.image_widget.is_playing:
            self.image_widget.pause_video()

            pix_icon = QStyle.StandardPixmap.SP_MediaPlay
            icon = self.style().standardIcon(pix_icon)
            self.play_pause_action.setIcon(icon)
        else:
            self.image_widget.play_video()
            # 自動儲存
            # if self.is_auto_save():
            #     self.save_annotations()
            
            pix_icon = QStyle.StandardPixmap.SP_MediaPause
            icon = self.style().standardIcon(pix_icon)
            self.play_pause_action.setIcon(icon)

    def set_media_position(self, position):
        # 設定影片播放位置 (以毫秒為單位)
        duration = self.image_widget.media_player.duration()
        self.image_widget.media_player.setPosition(int(position / 100 * duration))

    def set_playback_speed(self, index):
        # 設定播放速度
        speed = self.speed_control.itemData(index)
        self.image_widget.media_player.setPlaybackRate(speed)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Right or event.key() == Qt.Key.Key_PageDown:
            if self.is_auto_save():
                self.save_annotations()
            self.next_image()
        elif event.key() == Qt.Key.Key_Left or event.key() == Qt.Key.Key_PageUp:
            if self.is_auto_save():
                self.save_annotations()
            self.prev_image()
        elif event.key() == Qt.Key.Key_Q:
            self.close()
        elif event.key() == Qt.Key.Key_A:
            self.toggle_auto_save()
        elif event.key() == Qt.Key.Key_Space: # 空白鍵
            self.toggle_play_pause()

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

        self.model: None|YOLO = None

        # 角落大小
        self.CORNER_SIZE = 10

        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # 設定大小策略

        # 影片播放相關
        self.media_player = QMediaPlayer()
        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)
        # self.audio_output = QAudioOutput()
        # self.media_player.setAudioOutput(self.audio_output)
        self.is_playing = False  # 追蹤是否在播放影片
        self.timer = QTimer(self) # 使用 QTimer
        self.timer.timeout.connect(self._update_frame)

        self.update_timer = QTimer(self) # 更新進度條
        self.update_timer.timeout.connect(self.update_progress)


        # 縮放後的影像尺寸
        self.scaled_width = None
        self.scaled_height = None
        # (x, y)--->┌－－－－－－－┐ ╮
        #           │             │ │
        #           │<---width--->│  height
        #           │             │ │
        #           └－－－－－－－┘ ╯

    def _update_frame(self):
        self.pixmap = self.video_widget.grab()
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
            "top_left": QRect(x1 - self.CORNER_SIZE, y1 - self.CORNER_SIZE, self.CORNER_SIZE * 2, self.CORNER_SIZE * 2),
            "top_right": QRect(x2 - self.CORNER_SIZE, y1 - self.CORNER_SIZE, self.CORNER_SIZE * 2, self.CORNER_SIZE * 2),
            "bottom_left": QRect(x1 - self.CORNER_SIZE, y2 - self.CORNER_SIZE, self.CORNER_SIZE * 2, self.CORNER_SIZE * 2),
            "bottom_right": QRect(x2 - self.CORNER_SIZE, y2 - self.CORNER_SIZE, self.CORNER_SIZE * 2, self.CORNER_SIZE * 2),
        }

        for corner, rect in corners.items():
            if rect.contains(pos):
                return corner
        return None

    def load_image(self, image_path):
        # 判斷檔案是否為影片
        if image_path.lower().endswith(VIDEO_EXTS):
            self.media_player.setSource(QUrl.fromLocalFile(image_path))
            # self.pixmap = self.video_widget.grab()
            # self.video_widget.show()
            self.is_playing = False
        else:
            self.image = cv2.imread(image_path)
            height, width, channel = self.image.shape
            bytesPerLine = 3 * width
            qImg = QImage(self.image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888).rgbSwapped()
            self.pixmap = QPixmap.fromImage(qImg)
            self.bboxes = []  # 清空 Bounding Box
            # self.video_widget.hide()

            # 嘗試讀取 XML 檔案
            xml_path = getXmlPath(image_path)
            if os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for obj in root.findall('object'):
                        name = obj.find('name').text
                        bndbox = obj.find('bndbox')
                        xmin = int(bndbox.find('xmin').text)
                        ymin = int(bndbox.find('ymin').text)
                        xmax = int(bndbox.find('xmax').text)
                        ymax = int(bndbox.find('ymax').text)
                        confidence = float(bndbox.find('confidence').text)
                        width = xmax - xmin
                        height = ymax - ymin
                        self.bboxes.append(Bbox(xmin, ymin, width, height, name, confidence))
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to parse XML: {e}")
        
        # 執行物件偵測
        if hasattr(self, 'model') and self.model:
            results = self.model.predict(self.image)
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
                        c = box.cls
                        conf = box.conf
                        label = self.model.names[int(c)]
                        self.bboxes.append(Bbox(int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1]), label, float(conf)))
        self.update()  # 觸發 paintEvent
        
    def play_video(self):
        if self.media_player.source().isEmpty() is False:
            self.media_player.play()
            self.is_playing = True
            self.timer.start(100)  # 每 100 毫秒更新一次
            self.pixmap = self.video_widget.grab() # 播放後抓取
            self.image_label.setPixmap(self.pixmap) # 更新圖片

    def pause_video(self):
        if self.media_player.source().isEmpty() is False:
            self.media_player.pause()
            self.is_playing = False
            self.timer.stop()

    def stop_video(self):
        if self.media_player.source().isEmpty() is False:
            self.media_player.stop()
            self.is_playing = False
            self.update_timer.stop()

    def update_progress(self):
        if self.media_player.source().isEmpty() is False:
            # 更新進度條
            progress = self.media_player.position() / self.media_player.duration() * 100
            self.main_window.progress_bar.setValue(int(progress))

            # self.video_widget.update() # 重要, 強制更新畫面
            # self.pixmap = self.video_widget.grab()
            # self.image_label.setPixmap(self.pixmap)
            self.update()
            
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.is_playing:
            # 繪製當前幀
            # self.video_widget.update() # 重要, 強制更新畫面
            # self.pixmap = self.video_widget.grab()
            painter.drawPixmap(0, 0, self.pixmap)

        if self.pixmap:
            # 計算繪製區域，將縮放後的影像置於左上
            scaled_pixmap = self.pixmap.scaled(self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatio)

            # 計算縮放後的影像尺寸, 之後都幾乎以這兩個為參考
            self.scaled_width = scaled_pixmap.width()
            self.scaled_height = scaled_pixmap.height()

            painter.drawPixmap(0, 0, scaled_pixmap)

            # 繪製 Bounding Box
            for bbox in self.bboxes:
                if hasattr(bbox, 'label_color') and bbox.label_color == "red":
                    pen = QPen(QColor(255, 0, 0), 2)  # 紅色
                else:
                    pen = QPen(QColor(0, 255, 0), 2)  # 綠色，寬度 2
                painter.setPen(pen)
                rect = QRect(self._scale_to_widget(QPoint(bbox.x, bbox.y)),
                                self._scale_to_widget(QPoint(bbox.x + bbox.width, bbox.y + bbox.height)))
                painter.drawRect(rect)

                # 計算文字大小
                text = f"{bbox.label} ({bbox.confidence:.2f})"
                font_metrics = painter.fontMetrics()
                text_width = font_metrics.horizontalAdvance(text)
                text_height = font_metrics.height()

                # 繪製文字底色 (調整位置和大小)
                qpt_text = QPoint(bbox.x, bbox.y - text_height)
                bg_rect = QRect(self._scale_to_widget(qpt_text),
                                    QPoint(self._scale_to_widget(qpt_text).x()
                                            + int(text_width * self.scaled_width / self.pixmap.width()),
                                        self._scale_to_widget(qpt_text).y()
                                            + int(text_height * self.scaled_height / self.pixmap.height())))
                painter.fillRect(bg_rect, QColor(0, 0, 0, 127))  # 黑色半透明底色

                # 繪製文字 (調整位置)
                painter.drawText(self._scale_to_widget(QPoint(bbox.x, bbox.y - text_height)), text)

            if self.drawing:
                pen = QPen(QColor(255, 0, 0), 2)  # 繪製中的 Bounding Box 用紅色
                painter.setPen(pen)
                rect = QRect(self.start_pos, self.end_pos)
                painter.drawRect(rect)

    def mousePressEvent(self, event):
        if self.is_playing:
            self.stop_video()
        
        if event.button() == Qt.MouseButton.LeftButton:
            for bbox in self.bboxes:
                corner = self._is_in_corner(event.pos(), bbox)
                if corner:
                    self.resizing_bbox = bbox
                    self.resizing_corner = corner
                    self.original_bbox = (bbox.x, bbox.y, bbox.width, bbox.height)  # 儲存原始大小
                    self.start_pos = self._scale_to_original(event.pos()) # 紀錄原始座標
                    self.resizing_bbox.label_color = "red" # 改變顏色
                    self.resizing = True
                    break
            else:
                self.start_pos = event.pos()
                self.end_pos = event.pos()
                self.drawing = True

        elif event.button() == Qt.MouseButton.RightButton: # 刪除
            pos = event.pos()
            for bbox in reversed(self.bboxes): # 從後面開始找，避免 index 錯誤
                # 將原始影像座標轉換為視窗座標
                rect = QRect(self._scale_to_widget(QPoint(bbox.x, bbox.y)),
                                self._scale_to_widget(QPoint(bbox.x + bbox.width, bbox.y + bbox.height)))
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
                # x1, y1, width, height = self.resizing_bbox.x, self.resizing_bbox.y, self.resizing_bbox.width, self.resizing_bbox.height
                # x2, y2 = x1 + width, y1 + height
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
                
                # 取得標籤
                label, ok = QInputDialog.getText(self, 'Input', 'Enter label name:', text="object")
                if ok:
                    # 將視窗座標轉換為原始影像座標
                    x1_original, y1_original = self._scale_to_original(QPoint(x1, y1)).x(), self._scale_to_original(QPoint(x1, y1)).y()
                    width_original, height_original = int(width * self.pixmap.width() / self.scaled_width), int(height * self.pixmap.height() / self.scaled_height)
                    # 建立 Bbox 物件 (使用原始影像座標)
                    self.bboxes.append(Bbox(min(x1_original, x1_original + width_original), min(y1_original, y1_original+height_original), width_original, height_original, label, 1.0))
                    self.update()

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
        self.image_files.sort() # 排序

    def current_image_path(self):
        if self.image_files:
            return os.path.join(self.folder_path, self.image_files[self.current_index])
        return None

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            return True
        return False

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
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

if __name__ == '__main__':
    main()
