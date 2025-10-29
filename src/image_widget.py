import math
import time
import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QMessageBox,
    QSizePolicy,
    QWidget,
)

from src.config import cfg
from src.core import AppState
from src.utils.const import (
    CORNER_SIZE,
    ROTATION_HANDLE_DISTANCE,
    ROTATION_HANDLE_RADIUS,
    VIDEO_EXTS,
)
from src.utils.file_handler import file_h
from src.utils.func import getXmlPath
from src.utils.global_param import g_param
from src.utils.loglo import getUniqueLogger
from src.utils.model import Bbox, ColorPen, FileType

log = getUniqueLogger(__file__)


class DrawingMode(Enum):
    BBOX = 0
    MASK_DRAW = 1
    MASK_ERASE = 2
    MASK_FILL = 3


def qimage_to_cv_mat(qimage: QImage) -> np.ndarray:
    """Converts a QImage to an OpenCV Mat."""
    qimage = qimage.convertToFormat(QImage.Format.Format_ARGB32)
    width = qimage.width()
    height = qimage.height()

    ptr = qimage.bits()
    ptr.setsize(height * width * 4)
    arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
    return arr


def cv_mat_to_qimage(cv_mat: np.ndarray) -> QImage:
    """Converts an OpenCV Mat to a QImage."""
    height, width, channel = cv_mat.shape
    bytes_per_line = 4 * width
    return QImage(
        cv_mat.data, width, height, bytes_per_line, QImage.Format.Format_ARGB32
    )


class ImageWidget(QWidget):
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        self.image_label = QLabel()
        self.pixmap = None
        self.bboxes: list[Bbox] = []
        self.start_pos = None
        self.end_pos = None
        self.drawing = False

        self.idx_focus_bbox: int = -1
        self.resizing = False
        self.rotating = False  # 旋轉狀態
        self.selected_bbox = None
        self.resizing_corner = None
        self.original_bbox = None  # 儲存原始 bbox 資訊
        self.original_angle = None  # 儲存原始角度
        self.rotation_start_angle = None  # 旋轉開始時的滑鼠角度
        self.current_mouse_pos = None  # 儲存滑鼠當前位置
        self.fixed_corner_pos = None  # 儲存resize時固定的對角點位置（原始座標）

        # Mask drawing properties
        self.drawing_mode = DrawingMode.BBOX
        self.mask_pixmap: QPixmap | None = None
        self.brush_size = 20
        self.last_pos = None

        self.list_fps = []

        self.cv_img = None
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )  # 設定大小策略

        # 縮放後的影像尺寸
        self.scaled_width = None
        self.scaled_height = None
        self.cap = None

        # Callbacks for main window communication
        self.on_mouse_press_callback = None
        self.on_wheel_event_callback = None
        self.on_video_loaded_callback = None
        self.on_image_loaded_callback = None
        self.file_type = FileType.IMAGE
        self.fps = 30
        # (x, y)--->┌－－－－－－－┐ ╮
        #           │             │ │
        #           │<---width--->│  height
        #           │             │ │
        #           └－－－－－－－┘ ╯

    def set_callbacks(
        self,
        on_mouse_press=None,
        on_wheel_event=None,
        on_video_loaded=None,
        on_image_loaded=None,
    ):
        """Set callback functions for main window communication."""
        if on_mouse_press:
            self.on_mouse_press_callback = on_mouse_press
        if on_wheel_event:
            self.on_wheel_event_callback = on_wheel_event
        if on_video_loaded:
            self.on_video_loaded_callback = on_video_loaded
        if on_image_loaded:
            self.on_image_loaded_callback = on_image_loaded

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
        """用於選取bbox，考慮旋轉後的bbox區域"""
        pos = self._scale_to_original(pos)

        if bbox.angle == 0:
            # 未旋轉時，直接取矩形角落
            rect = QRect(bbox.x, bbox.y, bbox.width, bbox.height)
            return rect.contains(pos)
        else:
            # 旋轉時，多邊形檢測
            corners = self._getRotatedCorners(bbox)
            polygon = np.array(corners, dtype=np.float32)
            point = (float(pos.x()), float(pos.y()))
            result = cv2.pointPolygonTest(polygon, point, False)
            return result >= 0

    def _isInCorner(self, pos, bbox: Bbox) -> str:
        """檢查滑鼠是否在角落，並且resize"""
        pos = self._scale_to_original(pos)

        if bbox.angle == 0:
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
        else:
            # 旋轉的情況，使用角點距離檢測
            corners = self._getRotatedCorners(bbox)
            corner_names = ["top_left", "top_right", "bottom_right", "bottom_left"]

            for i, (cx, cy) in enumerate(corners):
                dx = pos.x() - cx
                dy = pos.y() - cy
                distance = (dx * dx + dy * dy) ** 0.5
                if distance <= CORNER_SIZE * 2:
                    return corner_names[i]
            return None

    def _getRotationHandlePos(self, bbox: Bbox) -> QPoint:
        """取得旋轉控制點的位置（原始座標）"""
        center_x = bbox.x + bbox.width / 2
        center_y = bbox.y + bbox.height / 2

        # 計算旋轉後的控制點位置
        # 預設控制點在上方，需要根據角度旋轉
        angle_rad = math.radians(bbox.angle)
        # 控制點相對於中心的位置（未旋轉時在上方）
        handle_offset_x = 0
        handle_offset_y = -(bbox.height / 2 + ROTATION_HANDLE_DISTANCE)

        # 旋轉這個偏移量
        rotated_x = handle_offset_x * math.cos(angle_rad) - handle_offset_y * math.sin(
            angle_rad
        )
        rotated_y = handle_offset_x * math.sin(angle_rad) + handle_offset_y * math.cos(
            angle_rad
        )

        return QPoint(int(center_x + rotated_x), int(center_y + rotated_y))

    def _getRotatedCorners(self, bbox: Bbox) -> list[tuple[float, float]]:
        """獲取旋轉後的四個角點座標（原始座標）
        返回順序：top_left, top_right, bottom_right, bottom_left
        """
        center_x = bbox.x + bbox.width / 2
        center_y = bbox.y + bbox.height / 2

        # 四個角點相對於中心的位置（未旋轉時）
        corners = [
            (-bbox.width / 2, -bbox.height / 2),  # top_left
            (bbox.width / 2, -bbox.height / 2),  # top_right
            (bbox.width / 2, bbox.height / 2),  # bottom_right
            (-bbox.width / 2, bbox.height / 2),  # bottom_left
        ]

        angle_rad = math.radians(bbox.angle)
        rotated_corners = []

        for dx, dy in corners:
            # 旋轉這個偏移量
            rotated_x = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            rotated_y = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            rotated_corners.append((center_x + rotated_x, center_y + rotated_y))

        return rotated_corners

    def _isOnRotationHandle(self, pos, bbox: Bbox) -> bool:
        """檢查滑鼠是否在旋轉控制點上"""
        pos_original = self._scale_to_original(pos)
        handle_pos = self._getRotationHandlePos(bbox)

        # 計算距離
        dx = pos_original.x() - handle_pos.x()
        dy = pos_original.y() - handle_pos.y()
        distance = (dx * dx + dy * dy) ** 0.5

        return distance <= ROTATION_HANDLE_RADIUS * 2

    def loadBboxFromXml(self, xml_path) -> bool:
        """
        讀取xml的bbox資訊

        Args:
            xml_path (str): xml檔案路徑

        Returns:
            bool: 是否有bbox
        """
        if Path(xml_path).is_file():
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
                    # 讀取 angle 參數，如果不存在則預設為 0
                    angle_element = bndbox.find("angle")
                    angle = (
                        float(angle_element.text) if angle_element is not None else 0.0
                    )
                    width = xmax - xmin
                    height = ymax - ymin
                    self.bboxes.append(
                        Bbox(xmin, ymin, width, height, name, confidence, int(angle))
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
        if not self.app_state.model:
            self.app_state.auto_detect = False
            QMessageBox.critical(self, "Error", "Model not loaded")
            return

        if not file_h.current_image_path():
            # QMessageBox.critical(self, "Error", "No image loaded")
            return

        if cfg.show_fps:
            t1 = time.time()
        self.bboxes = []
        # 強制設定device=0跑gpu
        results = self.app_state.model.predict(self.cv_img, device=0, verbose=False)
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    b = box.xyxy[
                        0
                    ]  # get box coordinates in (top, left, bottom, right) format
                    c = box.cls
                    conf = box.conf
                    label = self.app_state.model.names[int(c)]
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
        if cfg.show_fps:
            self.list_fps.append(1 / (time.time() - t1))
            if len(self.list_fps) > 10:
                self.list_fps.pop(0)
            log.i(f"Detection avg fps: {sum(self.list_fps) / len(self.list_fps):.0f}")
        self.update()

    def get_total_msec(self):
        # 取得影片總毫秒數
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(total_frames * 1000 / self.fps)

    def set_drawing_mode(self, mode: DrawingMode):
        self.drawing_mode = mode
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_brush_size(self, size: int):
        self.brush_size = size

    def load_image(self, file_path):
        if not file_path:
            self.pixmap = None
            self.clearBboxes()
            self.update()
            return

        # 判斷檔案是否為影片
        if file_path.lower().endswith(VIDEO_EXTS):
            # Google AI Gemini-2.0-pro 跟我都試過了, 沒有辦法把video widget的frame傳到畫布中編輯
            # 因此用傳統的方式來把opencv frame轉成pixmap
            self.file_type = FileType.VIDEO

            self.cap = cv2.VideoCapture(file_path)
            ret, self.cv_img = self.cap.read()
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            # log.info(f"Video FPS: {self.fps}")

            if self.on_video_loaded_callback:
                self.on_video_loaded_callback(self.get_total_msec())
        else:
            self.file_type = FileType.IMAGE
            self.cv_img = cv2.imread(file_path)

            if self.on_image_loaded_callback:
                self.on_image_loaded_callback()
        if self.cv_img is None:
            log.w(f"load {file_path} failed")
            QMessageBox.critical(self, "Error", f"Failed to load file `{file_path}`")
            self.pixmap = None
            self.update()
            return

        height, width, channel = self.cv_img.shape
        bytesPerLine = 3 * width
        qImg = QImage(
            self.cv_img.data, width, height, bytesPerLine, QImage.Format.Format_RGB888
        ).rgbSwapped()
        self.pixmap = QPixmap.fromImage(qImg)
        self.clearBboxes()

        # Initialize the mask pixmap
        self.mask_pixmap = QPixmap(self.pixmap.size())
        self.mask_pixmap.fill(Qt.GlobalColor.transparent)

        # 嘗試讀取 XML 檔案
        xml_path = getXmlPath(file_path)
        if not self.loadBboxFromXml(xml_path):
            # 如果 bbox (來自xml) 不存在, 才嘗試使用 YOLO 偵測
            if self.app_state.auto_detect:
                self.detectImage()
        self.update()  # 觸發 paintEvent

    def clearBboxes(self):
        # 重置 Bounding Box 資訊
        self.bboxes = []
        self.idx_focus_bbox = -1

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if not self.pixmap:
            return
        # 計算繪製區域，將縮放後的影像置於左上
        scaled_pixmap = self.pixmap.scaled(
            self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatio
        )

        # 計算縮放後的影像尺寸, 之後都幾乎以這兩個為參考
        self.scaled_width = scaled_pixmap.width()
        self.scaled_height = scaled_pixmap.height()

        painter.drawPixmap(0, 0, scaled_pixmap)

        # Draw mask
        if self.mask_pixmap:
            painter.drawPixmap(
                0,
                0,
                self.mask_pixmap.scaled(
                    self.scaled_width,
                    self.scaled_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                ),
            )

        # 繪製 Bounding Box
        for bbox in self.bboxes:
            painter.setPen(bbox.color_pen)

            if bbox.angle != 0:
                # 繪製旋轉的 bounding box
                # 計算中心點（原始座標）
                center_x = bbox.x + bbox.width / 2
                center_y = bbox.y + bbox.height / 2

                # 轉換到視窗座標
                center_widget = self._scale_to_widget(
                    QPoint(int(center_x), int(center_y))
                )

                # 計算縮放後的寬高
                scaled_width = bbox.width * self.scaled_width / self.pixmap.width()
                scaled_height = bbox.height * self.scaled_height / self.pixmap.height()

                # 保存當前畫筆狀態
                painter.save()
                # 移動到中心點
                painter.translate(center_widget.x(), center_widget.y())
                # 順時針旋轉
                painter.rotate(bbox.angle)
                # 繪製矩形（以中心為原點）
                painter.drawRect(
                    int(-scaled_width / 2),
                    int(-scaled_height / 2),
                    int(scaled_width),
                    int(scaled_height),
                )
                # 恢復畫筆狀態
                painter.restore()

                # 繪製文字（在未旋轉的位置）
                text = f"{bbox.label} ({bbox.confidence:.2f})"
                if bbox.angle != 0:
                    text += f" [{bbox.angle:.0f}°]"
                font_metrics = painter.fontMetrics()
                text_width = font_metrics.horizontalAdvance(text)
                text_height = font_metrics.height()

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
                painter.fillRect(bg_rect, QColor(0, 0, 0, 150))
                painter.drawText(self._scale_to_widget(qpt_text), text)
            else:
                # 繪製一般的 bounding box
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

        # 繪製選中 bbox 的旋轉控制點
        if self.idx_focus_bbox != -1 and 0 <= self.idx_focus_bbox < len(self.bboxes):
            focused_bbox = self.bboxes[self.idx_focus_bbox]

            # 計算中心點和旋轉控制點位置
            center_x = focused_bbox.x + focused_bbox.width / 2
            center_y = focused_bbox.y + focused_bbox.height / 2
            center_widget = self._scale_to_widget(QPoint(int(center_x), int(center_y)))

            handle_pos_original = self._getRotationHandlePos(focused_bbox)
            handle_pos_widget = self._scale_to_widget(handle_pos_original)

            # 繪製虛線（從 bbox 上邊中點到旋轉控制點）
            angle_rad = math.radians(focused_bbox.angle)
            top_center_offset_x = 0
            top_center_offset_y = -focused_bbox.height / 2
            rotated_top_x = top_center_offset_x * math.cos(
                angle_rad
            ) - top_center_offset_y * math.sin(angle_rad)
            rotated_top_y = top_center_offset_x * math.sin(
                angle_rad
            ) + top_center_offset_y * math.cos(angle_rad)
            top_center_original = QPoint(
                int(center_x + rotated_top_x), int(center_y + rotated_top_y)
            )
            top_center_widget = self._scale_to_widget(top_center_original)

            dashed_pen = QPen(QColor(255, 255, 0), 1, Qt.PenStyle.DashLine)
            painter.setPen(dashed_pen)
            painter.drawLine(top_center_widget, handle_pos_widget)

            # 繪製旋轉控制點圓圈
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.setBrush(QColor(255, 255, 255, 200))
            painter.drawEllipse(
                handle_pos_widget, ROTATION_HANDLE_RADIUS, ROTATION_HANDLE_RADIUS
            )

        if self.drawing and self.drawing_mode == DrawingMode.BBOX:
            painter.setPen(ColorPen.RED)  # 繪製中的 Bounding Box 用紅色
            rect = QRect(self.start_pos, self.end_pos)
            painter.drawRect(rect)

            # 顯示繪製中的 bbox 解析度
            if self.start_pos and self.end_pos:
                orig_start = self._scale_to_original(self.start_pos)
                orig_end = self._scale_to_original(self.end_pos)
                w = abs(orig_end.x() - orig_start.x())
                h = abs(orig_end.y() - orig_start.y())
                text = f"{w}x{h}={w * h}"
                font_metrics = painter.fontMetrics()
                text_width = font_metrics.horizontalAdvance(text)
                text_height = font_metrics.height()
                text_pos = self.end_pos + QPoint(15, 15)
                bg_rect = QRect(
                    text_pos,
                    QPoint(text_pos.x() + text_width + 4, text_pos.y() + text_height),
                )
                painter.fillRect(bg_rect, QColor(0, 0, 0, 150))
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(
                    text_pos + QPoint(2, text_height - font_metrics.descent()), text
                )

        # 如果正在調整大小, 也顯示解析度
        if self.resizing and self.selected_bbox and self.current_mouse_pos:
            text = f"{self.selected_bbox.width}x{self.selected_bbox.height}={self.selected_bbox.width * self.selected_bbox.height}"
            font_metrics = painter.fontMetrics()
            text_width = font_metrics.horizontalAdvance(text)
            text_height = font_metrics.height()
            text_pos = self.current_mouse_pos + QPoint(15, 15)
            bg_rect = QRect(
                text_pos,
                QPoint(text_pos.x() + text_width + 4, text_pos.y() + text_height),
            )
            painter.fillRect(bg_rect, QColor(0, 0, 0, 150))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                text_pos + QPoint(2, text_height - font_metrics.descent()), text
            )

    def draw_on_mask(self, pos: QPoint):
        if self.last_pos is None:
            self.last_pos = pos
            return

        painter = QPainter(self.mask_pixmap)

        if self.drawing_mode == DrawingMode.MASK_DRAW:
            pen = QPen(
                QColor(0, 0, 0, 255),
                self.brush_size,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        elif self.drawing_mode == DrawingMode.MASK_ERASE:
            pen = QPen(
                Qt.GlobalColor.transparent,
                self.brush_size,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        else:
            return

        painter.setPen(pen)
        painter.drawLine(self.last_pos, pos)
        self.last_pos = pos
        painter.end()
        self.update()

    def fill_mask(self, pos: QPoint):
        if not self.mask_pixmap:
            return

        q_img = self.mask_pixmap.toImage()
        # 確保是 ARGB 格式
        if q_img.format() != QImage.Format.Format_ARGB32:
            q_img = q_img.convertToFormat(QImage.Format.Format_ARGB32)

        cv_img = qimage_to_cv_mat(q_img)  # 得到 BGRA 格式的 ndarray

        bgr_img = cv_img[:, :, :3]
        alpha = cv_img[:, :, 3]

        scaled_pos = self._scale_to_original(pos)
        x, y = scaled_pos.x(), scaled_pos.y()

        h, w = alpha.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        # floodFill 會原地修改影像，所以我們複製 alpha channel
        alpha_to_fill = alpha.copy()

        mask = np.zeros((h + 2, w + 2), np.uint8)

        # 檢查點擊處的透明度
        if alpha[y, x] == 0:
            # 如果是透明的，填充為不透明 (255)
            fill_value = 255
        else:
            # 如果是不透明的，填充為透明 (0)
            fill_value = 0

        # 在 alpha channel 上執行 flood fill
        cv2.floodFill(alpha_to_fill, mask, (x, y), fill_value)

        # 將原始 BGR 和修改後的 alpha channel 合併
        new_cv_img = np.dstack((bgr_img, alpha_to_fill))

        result_q_img = cv_mat_to_qimage(new_cv_img)
        self.mask_pixmap = QPixmap.fromImage(result_q_img)
        self.update()

    def mousePressEvent(self, event):
        if self.on_mouse_press_callback:
            self.on_mouse_press_callback(event)

        if event.button() == Qt.MouseButton.LeftButton:
            if self.drawing_mode in [DrawingMode.MASK_DRAW, DrawingMode.MASK_ERASE]:
                self.start_pos = event.pos()
                self.end_pos = event.pos()
                self.drawing = True
                scaled_pos = self._scale_to_original(event.pos())
                self.last_pos = scaled_pos
                self.draw_on_mask(scaled_pos)
            elif self.drawing_mode == DrawingMode.MASK_FILL:
                self.fill_mask(event.pos())
            elif self.drawing_mode == DrawingMode.BBOX:
                for idx_focus, bbox in enumerate(self.bboxes):
                    # 先檢查旋轉控制點
                    if self._isOnRotationHandle(event.pos(), bbox):
                        self.selected_bbox = bbox
                        self.selected_bbox.color_pen = ColorPen.YELLOW
                        self.idx_focus_bbox = idx_focus
                        self.rotating = True
                        self.original_angle = bbox.angle

                        # 計算滑鼠相對於 bbox 中心的角度
                        pos_original = self._scale_to_original(event.pos())
                        center_x = bbox.x + bbox.width / 2
                        center_y = bbox.y + bbox.height / 2
                        dx = pos_original.x() - center_x
                        dy = pos_original.y() - center_y
                        self.rotation_start_angle = math.degrees(math.atan2(dy, dx))

                        self.update()
                        break

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

                        # 計算固定的對角點位置（用於旋轉bbox的resize）
                        if bbox.angle != 0:
                            corners = self._getRotatedCorners(bbox)
                            corner_map = {
                                "top_left": 2,  # 固定 bottom_right
                                "top_right": 3,  # 固定 bottom_left
                                "bottom_right": 0,  # 固定 top_left
                                "bottom_left": 1,  # 固定 top_right
                            }
                            fixed_idx = corner_map.get(corner, 0)
                            self.fixed_corner_pos = corners[fixed_idx]

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
            if self.drawing_mode == DrawingMode.BBOX:
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
        self.current_mouse_pos = event.pos()
        if self.drawing_mode in [DrawingMode.MASK_DRAW, DrawingMode.MASK_ERASE]:
            if self.drawing:
                scaled_pos = self._scale_to_original(event.pos())
                self.draw_on_mask(scaled_pos)
            return

        # BBOX mode logic
        cursor_changed = False
        if not self.resizing and not self.drawing and not self.rotating:
            for bbox in self.bboxes:
                # 檢查旋轉控制點
                if self._isOnRotationHandle(event.pos(), bbox):
                    self.setCursor(Qt.CursorShape.CrossCursor)
                    cursor_changed = True
                    break
                corner = self._isInCorner(event.pos(), bbox)
                if corner in ["top_left", "bottom_right"]:
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                    cursor_changed = True
                    break
                elif corner in ["top_right", "bottom_left"]:
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                    cursor_changed = True
                    break
        if not cursor_changed and not self.resizing and not self.rotating:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        if self.drawing:  # BBOX drawing
            # 繪製的話, 就讓之前有被選取的bbox先恢復原樣
            if self.selected_bbox:
                self.selected_bbox.color_pen = ColorPen.GREEN
            self.end_pos = event.pos()
        elif self.resizing:
            pos = self._scale_to_original(event.pos())

            if self.selected_bbox.angle == 0:
                # 未旋轉的情況，使用原有邏輯
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
            else:
                # 旋轉的情況，固定對角點進行resize
                # 獲取固定點和滑鼠位置
                fixed_x, fixed_y = self.fixed_corner_pos
                mouse_x, mouse_y = pos.x(), pos.y()

                # 計算當前bbox的中心點（用於座標轉換）
                orig_center_x = self.original_bbox[0] + self.original_bbox[2] / 2
                orig_center_y = self.original_bbox[1] + self.original_bbox[3] / 2

                # 將固定點和滑鼠點轉換到局部座標系（相對於原始中心，反旋轉）
                angle_rad = -math.radians(self.selected_bbox.angle)  # 反向旋轉

                # 固定點的局部座標
                fixed_dx = fixed_x - orig_center_x
                fixed_dy = fixed_y - orig_center_y
                local_fixed_x = fixed_dx * math.cos(angle_rad) - fixed_dy * math.sin(
                    angle_rad
                )
                local_fixed_y = fixed_dx * math.sin(angle_rad) + fixed_dy * math.cos(
                    angle_rad
                )

                # 滑鼠點的局部座標
                mouse_dx = mouse_x - orig_center_x
                mouse_dy = mouse_y - orig_center_y
                local_mouse_x = mouse_dx * math.cos(angle_rad) - mouse_dy * math.sin(
                    angle_rad
                )
                local_mouse_y = mouse_dx * math.sin(angle_rad) + mouse_dy * math.cos(
                    angle_rad
                )

                # 在局部座標系中計算新的寬高
                new_width = abs(local_mouse_x - local_fixed_x)
                new_height = abs(local_mouse_y - local_fixed_y)

                # 設置最小值
                new_width = max(new_width, 10)
                new_height = max(new_height, 10)

                # 計算新的中心點（局部座標）
                new_local_center_x = (local_mouse_x + local_fixed_x) / 2
                new_local_center_y = (local_mouse_y + local_fixed_y) / 2

                # 將新的中心點轉換回全局座標系（旋轉）
                angle_rad = math.radians(self.selected_bbox.angle)  # 正向旋轉
                new_center_x = (
                    new_local_center_x * math.cos(angle_rad)
                    - new_local_center_y * math.sin(angle_rad)
                    + orig_center_x
                )
                new_center_y = (
                    new_local_center_x * math.sin(angle_rad)
                    + new_local_center_y * math.cos(angle_rad)
                    + orig_center_y
                )

                # 更新bbox
                self.selected_bbox.width = int(new_width)
                self.selected_bbox.height = int(new_height)
                self.selected_bbox.x = int(new_center_x - new_width / 2)
                self.selected_bbox.y = int(new_center_y - new_height / 2)

        elif self.rotating:
            # 計算當前滑鼠相對於 bbox 中心的角度
            pos_original = self._scale_to_original(event.pos())
            center_x = self.selected_bbox.x + self.selected_bbox.width / 2
            center_y = self.selected_bbox.y + self.selected_bbox.height / 2
            dx = pos_original.x() - center_x
            dy = pos_original.y() - center_y
            current_angle = math.degrees(math.atan2(dy, dx))

            # 計算角度變化
            angle_delta = current_angle - self.rotation_start_angle
            new_angle = self.original_angle + angle_delta

            # 正規化角度到 0-360 範圍
            self.selected_bbox.angle = new_angle % 360

        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.resizing:
                self.resizing = False
                self.selected_bbox = None
                self.resizing_corner = None
                self.original_bbox = None
                self.fixed_corner_pos = None

            elif self.rotating:
                self.rotating = False
                self.selected_bbox = None
                self.original_angle = None
                self.rotation_start_angle = None
                self.completeMouseAction()

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
                if width < cfg.minimal_bbox_length or height < cfg.minimal_bbox_length:
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
                        self.app_state.last_used_label,
                        1.0,
                    )
                )

                # 框已成形, focus到最後一個框, 以顯示旋轉控制點
                self.idx_focus_bbox = len(self.bboxes) - 1

            self.completeMouseAction()

    def completeMouseAction(self):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        for bbox in self.bboxes:
            bbox.color_pen = ColorPen.GREEN
        g_param.user_labeling = True
        self.update()

    def wheelEvent(self, event):
        # event.angleDelta().y() > 0, 代表滑鼠滾輪往上滾
        if self.on_wheel_event_callback:
            self.on_wheel_event_callback(event.angleDelta().y() > 0)
