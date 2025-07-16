import os
import xml.etree.ElementTree as ET

# from typing import TYPE_CHECKING
import cv2
from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QStyle,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QWidget,
)
from ultralytics import YOLO

from src.const import CORNER_SIZE, VIDEO_EXTS
from src.func import getXmlPath
from src.model import Bbox, ColorPen, FileType, ShowImageCmd

from src.loglo import getUniqueLogger

# if TYPE_CHECKING:
#     from src.object_tagger import MainWindow

log = getUniqueLogger(__file__)

class ImageWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.image_label = QLabel()
        self.pixmap = None
        self.bboxes: list[Bbox] = []
        self.start_pos = None
        self.end_pos = None
        self.drawing = False

        self.idx_focus_bbox: int = -1
        self.resizing = False
        self.selected_bbox = None
        self.resizing_corner = None
        self.original_bbox = None  # 儲存原始 bbox 資訊

        self.model: None | YOLO = None
        self.use_model = False

        self.cv_img = None
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )  # 設定大小策略

        # # 影片播放相關
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)

        self.is_playing = False  # 是否在播放影片

        self.auto_save_counter = 0  # 自動儲存計數器

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
                self.is_playing = False
                icon = self.main_window.style().standardIcon(
                    QStyle.StandardPixmap.SP_MediaPlay
                )
                self.main_window.play_pause_action.setIcon(icon)
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

            if self.main_window.is_auto_detect():
                self.detectImage()

            position = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.main_window.progress_bar.blockSignals(True)  # 暫時阻止信號傳遞
            self.main_window.progress_bar.setValue(int(position))
            self.main_window.progress_bar.blockSignals(False)  # 恢復信號傳遞

            self.update()

            # 自動儲存邏輯
            if (
                self.main_window.is_auto_save()
                and self.main_window.auto_save_per_second > 0
            ):
                self.auto_save_counter += 1
                if (
                    self.auto_save_counter
                    >= self.main_window.auto_save_per_second * self.fps
                ):
                    self.main_window.save_annotations()
                    self.auto_save_counter = 0

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

    def _isInBboxArea(self, pos, bbox: Bbox) -> bool:
        """用於選取bbox"""
        pos = self._scale_to_original(pos)

        rect = QRect(bbox.x, bbox.y, bbox.width, bbox.height)
        if rect.contains(pos):
            return True
        return False

    def _isInCorner(self, pos, bbox: Bbox) -> str:
        """檢查滑鼠是否在角落"""
        # 將視窗座標轉換為原始影像座標
        pos = self._scale_to_original(pos)

        x1, y1 = bbox.x, bbox.y
        x2, y2 = bbox.x + bbox.width, bbox.y + bbox.height

        # 計算四個角落的範圍
        corners = {
            "top_left": QRect(
                x1 - CORNER_SIZE,
                y1 - CORNER_SIZE,
                CORNER_SIZE * 2,
                CORNER_SIZE * 2,
            ),
            "top_right": QRect(
                x2 - CORNER_SIZE,
                y1 - CORNER_SIZE,
                CORNER_SIZE * 2,
                CORNER_SIZE * 2,
            ),
            "bottom_left": QRect(
                x1 - CORNER_SIZE,
                y2 - CORNER_SIZE,
                CORNER_SIZE * 2,
                CORNER_SIZE * 2,
            ),
            "bottom_right": QRect(
                x2 - CORNER_SIZE,
                y2 - CORNER_SIZE,
                CORNER_SIZE * 2,
                CORNER_SIZE * 2,
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
        執行物件偵測, 只有在model讀取時才會執行
        執行前先清除bboxes, 以免搞混
        """
        if not self.model:
            self.main_window.auto_detect = False
            self.main_window.auto_detect_action.setChecked(False)
            QMessageBox.critical(self, "Error", "Model not loaded")
            return

        if not self.main_window.file_handler.current_image_path():
            # QMessageBox.critical(self, "Error", "No image loaded")
            return

        self.bboxes = []
        results = self.model.predict(self.cv_img, verbose=False)
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
        self.update()

    def get_total_msec(self):
        # 取得影片總毫秒數
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(total_frames * 1000 / self.fps)

    def load_image(self, file_path):
        # 判斷檔案是否為影片
        self.is_playing = False
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.main_window.play_pause_action.setIcon(icon)
        if file_path.lower().endswith(VIDEO_EXTS):
            # Google AI Gemini-2.0-pro 跟我都試過了, 沒有辦法把video widget的frame傳到畫布中編輯
            # 因此用傳統的方式來把opencv frame轉成pixmap
            self.file_type = FileType.VIDEO

            self.cap = cv2.VideoCapture(file_path)
            ret, self.cv_img = self.cap.read()
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            log.info(f"Video FPS: {self.fps}")

            self.main_window.progress_bar.setRange(0, self.get_total_msec())

            self.main_window.progress_bar.blockSignals(True)
            self.main_window.progress_bar.setValue(0)
            self.cap.set(cv2.CAP_PROP_POS_MSEC, 0)
            self.main_window.progress_bar.blockSignals(False)
            # 啟用與影片相關的控制項
            self.main_window.play_pause_action.setEnabled(True)
            self.main_window.progress_bar.setEnabled(True)
            self.main_window.speed_control.setEnabled(True)
        else:
            self.file_type = FileType.IMAGE
            self.cv_img = cv2.imread(file_path)

            self.main_window.progress_bar.blockSignals(True)
            self.main_window.progress_bar.setValue(0)
            self.main_window.progress_bar.blockSignals(False)
            # 禁用與影片相關的控制項
            self.main_window.play_pause_action.setEnabled(False)
            self.main_window.progress_bar.setEnabled(False)
            self.main_window.speed_control.setEnabled(False)

        height, width, channel = self.cv_img.shape
        bytesPerLine = 3 * width
        qImg = QImage(
            self.cv_img.data, width, height, bytesPerLine, QImage.Format.Format_RGB888
        ).rgbSwapped()
        self.pixmap = QPixmap.fromImage(qImg)
        self.clearBboxes()

        # 嘗試讀取 XML 檔案
        # TODO: 加上影片的frame number
        xml_path = getXmlPath(file_path)
        if not self.loadBboxFromXml(xml_path):
            # 如果 bbox (來自xml) 不存在, 才嘗試使用 YOLO 偵測
            if self.main_window.is_auto_detect():
                self.detectImage()
        self.update()  # 觸發 paintEvent

    def clearBboxes(self):
        # 重置 Bounding Box 資訊
        self.bboxes = []
        self.idx_focus_bbox = -1

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
                painter.setPen(bbox.color_pen)
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
                text_width = font_metrics.horizontalAdvance(text)
                text_height = font_metrics.height()

                # 繪製文字底色
                qpt_text = QPoint(bbox.x, bbox.y)
                bg_rect = QRect(
                    QPoint(
                        self._scale_to_widget(qpt_text).x(),
                        self._scale_to_widget(qpt_text).y() - int(text_height),
                    ),
                    QPoint(
                        self._scale_to_widget(qpt_text).x() + int(text_width),
                        self._scale_to_widget(qpt_text).y(),
                    ),
                )
                painter.fillRect(bg_rect, QColor(0, 0, 0, 150))  # 黑色半透明底色

                # 繪製文字
                painter.drawText(
                    self._scale_to_widget(qpt_text),
                    text,
                )

            if self.drawing:
                painter.setPen(ColorPen.RED)  # 繪製中的 Bounding Box 用紅色
                rect = QRect(self.start_pos, self.end_pos)
                painter.drawRect(rect)

    def mousePressEvent(self, event):
        if self.is_playing:
            self.is_playing = False

        if event.button() == Qt.MouseButton.LeftButton:
            for idx_focus, bbox in enumerate(self.bboxes):
                corner = self._isInCorner(event.pos(), bbox)
                if corner:
                    self.selected_bbox = bbox
                    self.selected_bbox.color_pen = ColorPen.YELLOW
                    self.idx_focus_bbox = idx_focus
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

                    self.resizing = True
                    self.update()
                    break
            else:  # 接在forloop無break之時
                self.start_pos = event.pos()
                self.end_pos = event.pos()
                self.drawing = True
                for idx_focus, bbox in enumerate(self.bboxes):
                    if self._isInBboxArea(event.pos(), bbox):
                        self.selected_bbox = bbox
                        self.selected_bbox.color_pen = ColorPen.YELLOW
                        self.idx_focus_bbox = idx_focus
                        self.update()
                        break

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

                self.idx_focus_bbox = -1  # 重置

    def mouseMoveEvent(self, event):
        # 檢查是否在角落
        cursor_changed = False
        for bbox in self.bboxes:
            corner = self._isInCorner(event.pos(), bbox)
            if corner in ["top_left", "bottom_right"]:
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                cursor_changed = True
                break
            elif corner in ["top_right", "bottom_left"]:
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                cursor_changed = True
                break
        if not cursor_changed:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        if self.drawing:
            # 繪製的話, 就讓之前有被選取的bbox先恢復原樣
            if self.selected_bbox:
                self.selected_bbox.color_pen = ColorPen.GREEN
            self.end_pos = event.pos()
        elif self.resizing:
            pos = self._scale_to_original(event.pos())
            dx = pos.x() - self.start_pos.x()
            dy = pos.y() - self.start_pos.y()

            # 根據不同的角落調整大小
            if self.resizing_corner == "top_left":
                self.selected_bbox.x = self.original_bbox[0] + dx
                self.selected_bbox.y = self.original_bbox[1] + dy
                self.selected_bbox.width = self.original_bbox[2] - dx
                self.selected_bbox.height = self.original_bbox[3] - dy
            elif self.resizing_corner == "top_right":
                self.selected_bbox.y = self.original_bbox[1] + dy
                self.selected_bbox.width = self.original_bbox[2] + dx
                self.selected_bbox.height = self.original_bbox[3] - dy
            elif self.resizing_corner == "bottom_left":
                self.selected_bbox.x = self.original_bbox[0] + dx
                self.selected_bbox.width = self.original_bbox[2] - dx
                self.selected_bbox.height = self.original_bbox[3] + dy
            elif self.resizing_corner == "bottom_right":
                self.selected_bbox.width = self.original_bbox[2] + dx
                self.selected_bbox.height = self.original_bbox[3] + dy
            # # 不太實用, 畢竟有可能從bbox中間再畫出一個bbox
            # elif self.resizing_corner == "bbox_area":
            #     self.selected_bbox.x = self.original_bbox[0] + dx
            #     self.selected_bbox.y = self.original_bbox[1] + dy

        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.resizing:
                self.resizing = False
                self.selected_bbox = None
                self.resizing_corner = None
                self.original_bbox = None

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
                    # 範圍太小
                    self.completeMouseAction()
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

                # 框已成形, 就讓選取index失效
                self.idx_focus_bbox = -1

            self.completeMouseAction()

    def completeMouseAction(self):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        for bbox in self.bboxes:
            bbox.color_pen = ColorPen.GREEN
        self.update()

    def wheelEvent(self, event):
        if self.main_window.is_auto_save():
            self.main_window.save_annotations()
        if event.angleDelta().y() > 0:
            self.main_window.show_image(ShowImageCmd.PREV)
        else:
            self.main_window.show_image(ShowImageCmd.NEXT)
