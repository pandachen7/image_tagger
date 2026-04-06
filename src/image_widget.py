# 圖片畫布元件：負責繪製影像、bbox、polygon、mask，以及滑鼠互動（繪製、選取、拖曳、旋轉）
# 更新日期: 2026-03-11
import math
import time
import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
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
    POLYGON_CLOSE_THRESHOLD,
    POLYGON_SELECT_PADDING,
    POLYGON_VERTEX_RADIUS,
    ROTATION_HANDLE_DISTANCE,
    ROTATION_HANDLE_RADIUS,
    VIDEO_EXTS,
)
from src.utils.dynamic_settings import settings
from src.utils.file_handler import file_h
from src.utils.func import getXmlPath
from src.utils.global_param import g_param
from src.utils.img_handler import inferencer
from src.utils.logger import getUniqueLogger
from src.utils.model import Bbox, ColorPen, FileType, ModelType, Polygon, ViewMode

log = getUniqueLogger(__file__)


class DrawingMode(Enum):
    SELECT = 0
    BBOX = 1
    MASK_DRAW = 2
    MASK_ERASE = 3
    MASK_FILL = 4
    POLYGON = 5


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
        self.setMouseTracking(True)  # 即使沒按住按鍵也能追蹤滑鼠移動
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
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
        self.selected_bbox: Optional[Bbox] = None
        self.resizing_corner = None
        self.original_bbox = None  # 儲存原始 bbox 資訊
        self.original_angle = None  # 儲存原始角度
        self.rotation_start_angle = None  # 旋轉開始時的滑鼠角度
        self.current_mouse_pos = None  # 儲存滑鼠當前位置
        self.fixed_corner_pos = None  # 儲存resize時固定的對角點位置（原始座標）

        # Mask drawing properties
        self.drawing_mode = DrawingMode.SELECT
        self.mask_pixmap: QPixmap | None = None
        self.brush_size = 20
        self.last_pos = None

        # Polygon drawing state
        self.polygons: list[Polygon] = []
        self.current_polygon_points: list[QPoint] = []  # widget coords, in-progress
        self.idx_focus_polygon: int = -1

        # SELECT mode state
        self.select_type: str | None = None  # 'bbox', 'polygon', 'multi'
        self.dragging_vertex_idx: int = -1  # 拖曳中的polygon頂點index

        # 框選 (multi-select) state
        self.selected_bbox_indices: set[int] = set()
        self.selected_polygon_indices: set[int] = set()
        self.selection_rect_start: QPoint | None = None  # 框選起點
        self.dragging_selection: bool = False

        self.view_mode = ViewMode.ALL
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

    def _isNearPolygonVertex(self, pos: QPoint, polygon: Polygon) -> int:
        """檢查滑鼠是否靠近polygon的某個頂點

        Args:
            pos: widget座標
            polygon: Polygon物件

        Returns:
            int: 頂點index，若無則回傳-1
        """
        for i, (px, py) in enumerate(polygon.points):
            widget_pt = self._scale_to_widget(QPoint(int(px), int(py)))
            if self._distanceBetweenPoints(pos, widget_pt) < POLYGON_CLOSE_THRESHOLD:
                return i
        return -1

    def _finalizeRectSelection(self):
        """框選結束，找出框內的bbox和polygon"""
        if not self.selection_rect_start or not self.current_mouse_pos:
            return
        sel_rect = QRect(self.selection_rect_start, self.current_mouse_pos).normalized()
        self.selected_bbox_indices = set()
        self.selected_polygon_indices = set()

        # 檢查bbox是否與框選範圍相交（僅在view_mode可見時）
        if self.view_mode in (ViewMode.BBOX, ViewMode.ALL):
            for i, bbox in enumerate(self.bboxes):
                bbox_rect = QRect(
                    self._scale_to_widget(QPoint(bbox.x, bbox.y)),
                    self._scale_to_widget(
                        QPoint(bbox.x + bbox.width, bbox.y + bbox.height)
                    ),
                )
                if sel_rect.intersects(bbox_rect):
                    self.selected_bbox_indices.add(i)

        # 檢查polygon頂點是否在框選範圍內（僅在view_mode可見時）
        if self.view_mode in (ViewMode.SEG, ViewMode.ALL):
            for i, polygon in enumerate(self.polygons):
                for px, py in polygon.points:
                    wpt = self._scale_to_widget(QPoint(int(px), int(py)))
                    if sel_rect.contains(wpt):
                        self.selected_polygon_indices.add(i)
                        break

        if self.selected_bbox_indices or self.selected_polygon_indices:
            self.select_type = "multi"
        else:
            self.select_type = None

    def deleteSelectedAnnotation(self) -> bool:
        """刪除當前選取的bbox或polygon（支援多選）

        Returns:
            bool: 是否有刪除
        """
        deleted = False

        # 多選刪除（從後往前刪以避免index偏移）
        if self.select_type == "multi":
            for i in sorted(self.selected_bbox_indices, reverse=True):
                if 0 <= i < len(self.bboxes):
                    self.bboxes.pop(i)
                    deleted = True
            for i in sorted(self.selected_polygon_indices, reverse=True):
                if 0 <= i < len(self.polygons):
                    self.polygons.pop(i)
                    deleted = True
            self.selected_bbox_indices = set()
            self.selected_polygon_indices = set()
            self.select_type = None
        # 單選刪除
        elif self.select_type == "bbox" and 0 <= self.idx_focus_bbox < len(self.bboxes):
            self.bboxes.pop(self.idx_focus_bbox)
            self.idx_focus_bbox = -1
            self.select_type = None
            deleted = True
        elif self.select_type == "polygon" and 0 <= self.idx_focus_polygon < len(
            self.polygons
        ):
            self.polygons.pop(self.idx_focus_polygon)
            self.idx_focus_polygon = -1
            self.select_type = None
            deleted = True

        if deleted:
            g_param.user_labeling = True
            self.update()
        return deleted

    def loadBboxFromXml(self, xml_path) -> bool:
        """
        讀取xml的bbox與polygon資訊

        Args:
            xml_path (str): xml檔案路徑

        Returns:
            bool: 是否有bbox或polygon
        """
        if Path(xml_path).is_file():
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    name = obj.find("name").text
                    bndbox = obj.find("bndbox")
                    polygon_elem = obj.find("polygon")

                    if bndbox is not None:
                        xmin = int(bndbox.find("xmin").text)
                        ymin = int(bndbox.find("ymin").text)
                        xmax = int(bndbox.find("xmax").text)
                        ymax = int(bndbox.find("ymax").text)
                        confidence = float(bndbox.find("confidence").text)
                        angle_element = bndbox.find("angle")
                        angle = (
                            float(angle_element.text)
                            if angle_element is not None
                            else 0.0
                        )
                        width = xmax - xmin
                        height = ymax - ymin
                        self.bboxes.append(
                            Bbox(
                                xmin, ymin, width, height, name, confidence, int(angle)
                            )
                        )
                    elif polygon_elem is not None:
                        points = []
                        for pt in polygon_elem.findall("point"):
                            px = float(pt.find("x").text)
                            py = float(pt.find("y").text)
                            points.append((px, py))
                        if points:
                            conf_elem = polygon_elem.find("confidence")
                            poly_conf = (
                                float(conf_elem.text) if conf_elem is not None else -1.0
                            )
                            self.polygons.append(Polygon(points, name, poly_conf))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to parse XML: {e}")

            if self.bboxes or self.polygons:
                return True
            else:
                return False

    def runInference(self):
        """Run inference using the active model (YOLO or SAM3)."""
        if inferencer.active_model_type == ModelType.NONE:
            return
        if not file_h.current_image_path():
            return
        model_type = inferencer.active_model_type
        if not inferencer.ensure_loaded(model_type):
            return

        self.bboxes = []
        self.polygons = []

        if cfg.show_fps:
            t1 = time.time()

        if model_type == ModelType.YOLO:
            self.bboxes = inferencer.infer_yolo(self.cv_img)
        elif model_type == ModelType.SAM3:
            src_shape = (self.pixmap.height(), self.pixmap.width())
            bboxes, polygons = inferencer.infer_sam3(self.cv_img, src_shape)
            # 根據 sam3_label_mode 過濾結果
            mode = settings.models.sam3_label_mode or "seg"
            if mode == "seg":
                self.polygons = polygons
            elif mode == "bbox":
                self.bboxes = bboxes
            else:  # "all"
                self.bboxes = bboxes
                self.polygons = polygons

        if cfg.show_fps:
            self.list_fps.append(1 / (time.time() - t1))
            if len(self.list_fps) > 10:
                self.list_fps.pop(0)
            log.i(f"Inference avg fps: {sum(self.list_fps) / len(self.list_fps):.0f}")

        self.update()

    def get_total_msec(self):
        # 取得影片總毫秒數
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(total_frames * 1000 / self.fps)

    def set_drawing_mode(self, mode: DrawingMode):
        """切換繪圖模式"""
        self.drawing_mode = mode
        self.setCursor(Qt.CursorShape.ArrowCursor)
        # 切換模式時重置所有 focus / 選取狀態
        self.idx_focus_bbox = -1
        self.idx_focus_polygon = -1
        self.select_type = None
        self.selected_bbox_indices = set()
        self.selected_polygon_indices = set()
        self.update()

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
                self.runInference()
        self.update()  # 觸發 paintEvent

    def clearBboxes(self):
        """重置 Bounding Box 與 Polygon 資訊"""
        self.bboxes = []
        self.idx_focus_bbox = -1
        self.polygons = []
        self.current_polygon_points = []
        self.idx_focus_polygon = -1
        self.select_type = None
        self.selected_bbox_indices = set()
        self.selected_polygon_indices = set()

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

        # 繪製 Bounding Box (filtered by view mode)
        _bboxes_to_draw = (
            self.bboxes if self.view_mode in (ViewMode.BBOX, ViewMode.ALL) else []
        )
        for bbox_idx, bbox in enumerate(_bboxes_to_draw):
            # 多選中的bbox用黃色顯示
            if bbox_idx in self.selected_bbox_indices:
                painter.setPen(ColorPen.YELLOW)
            else:
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

        # 繪製選中 bbox 的控制點
        if (
            _bboxes_to_draw
            and self.idx_focus_bbox != -1
            and 0 <= self.idx_focus_bbox < len(self.bboxes)
        ):
            focused_bbox = self.bboxes[self.idx_focus_bbox]

            # OBB啟用時繪製旋轉控制點
            if cfg.enable_obb:
                center_x = focused_bbox.x + focused_bbox.width / 2
                center_y = focused_bbox.y + focused_bbox.height / 2
                center_widget = self._scale_to_widget(
                    QPoint(int(center_x), int(center_y))
                )

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

            # SELECT模式下繪製角落拖曳方塊
            if self.drawing_mode == DrawingMode.SELECT:
                painter.setPen(QPen(QColor(255, 255, 0), 1))
                painter.setBrush(QColor(255, 255, 255, 200))
                if focused_bbox.angle == 0:
                    corner_pts = [
                        QPoint(focused_bbox.x, focused_bbox.y),
                        QPoint(focused_bbox.x + focused_bbox.width, focused_bbox.y),
                        QPoint(focused_bbox.x, focused_bbox.y + focused_bbox.height),
                        QPoint(
                            focused_bbox.x + focused_bbox.width,
                            focused_bbox.y + focused_bbox.height,
                        ),
                    ]
                else:
                    rot_corners = self._getRotatedCorners(focused_bbox)
                    corner_pts = [QPoint(int(x), int(y)) for x, y in rot_corners]
                for cpt in corner_pts:
                    wpt = self._scale_to_widget(cpt)
                    painter.drawRect(
                        wpt.x() - CORNER_SIZE,
                        wpt.y() - CORNER_SIZE,
                        CORNER_SIZE * 2,
                        CORNER_SIZE * 2,
                    )
                painter.setBrush(Qt.BrushStyle.NoBrush)

        # 繪製 Polygons (filtered by view mode)
        _polygons_to_draw = (
            self.polygons if self.view_mode in (ViewMode.SEG, ViewMode.ALL) else []
        )
        for poly_idx, polygon in enumerate(_polygons_to_draw):
            # 多選中的polygon用黃色
            if poly_idx in self.selected_polygon_indices:
                painter.setPen(ColorPen.YELLOW)
            else:
                painter.setPen(polygon.color_pen)
            # Semi-transparent fill
            fill_color = QColor(0, 255, 0, 50)
            if (
                poly_idx == self.idx_focus_polygon
                or poly_idx in self.selected_polygon_indices
            ):
                fill_color = QColor(255, 255, 0, 70)

            if len(polygon.points) >= 3:
                qpoly = QPolygonF()
                for px, py in polygon.points:
                    widget_pt = self._scale_to_widget(QPoint(int(px), int(py)))
                    qpoly.append(QPointF(widget_pt.x(), widget_pt.y()))

                painter.setBrush(fill_color)
                painter.drawPolygon(qpoly)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                # Draw vertex dots (SELECT模式下選取的polygon用大圓點)
                if (
                    poly_idx == self.idx_focus_polygon
                    and self.drawing_mode == DrawingMode.SELECT
                ):
                    painter.setPen(QPen(QColor(255, 255, 0), 2))
                    painter.setBrush(QColor(255, 255, 255, 200))
                    for px, py in polygon.points:
                        widget_pt = self._scale_to_widget(QPoint(int(px), int(py)))
                        painter.drawEllipse(
                            widget_pt,
                            POLYGON_VERTEX_RADIUS * 2,
                            POLYGON_VERTEX_RADIUS * 2,
                        )
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                else:
                    for px, py in polygon.points:
                        widget_pt = self._scale_to_widget(QPoint(int(px), int(py)))
                        painter.drawEllipse(
                            widget_pt, POLYGON_VERTEX_RADIUS, POLYGON_VERTEX_RADIUS
                        )

                # Draw label text
                first_pt = self._scale_to_widget(
                    QPoint(int(polygon.points[0][0]), int(polygon.points[0][1]))
                )
                text = f"{polygon.label}"
                if polygon.confidence >= 0:
                    text += f" ({polygon.confidence:.2f})"
                font_metrics = painter.fontMetrics()
                text_width = font_metrics.horizontalAdvance(text)
                text_height = font_metrics.height()
                bg_rect = QRect(
                    QPoint(first_pt.x(), first_pt.y() - text_height),
                    QPoint(first_pt.x() + text_width, first_pt.y()),
                )
                painter.fillRect(bg_rect, QColor(0, 0, 0, 150))
                painter.drawText(first_pt, text)

        # 繪製進行中的 Polygon
        if self.current_polygon_points and self.drawing_mode == DrawingMode.POLYGON:
            painter.setPen(ColorPen.RED)
            # Draw lines between existing points
            for i in range(len(self.current_polygon_points) - 1):
                painter.drawLine(
                    self.current_polygon_points[i],
                    self.current_polygon_points[i + 1],
                )

            # Draw vertex dots
            for i, pt in enumerate(self.current_polygon_points):
                if i == 0:
                    # First point: green "close" indicator
                    painter.setPen(QPen(QColor(0, 255, 0), 2))
                    painter.drawEllipse(
                        pt,
                        POLYGON_CLOSE_THRESHOLD,
                        POLYGON_CLOSE_THRESHOLD,
                    )
                    painter.setPen(ColorPen.RED)
                painter.drawEllipse(pt, POLYGON_VERTEX_RADIUS, POLYGON_VERTEX_RADIUS)

            # Rubber band line from last point to cursor
            if self.current_mouse_pos and self.current_polygon_points:
                painter.setPen(QPen(QColor(255, 0, 0, 128), 1, Qt.PenStyle.DashLine))
                painter.drawLine(
                    self.current_polygon_points[-1], self.current_mouse_pos
                )

        # BBOX兩點模式：繪製中的黃色矩形預覽
        if self.drawing and self.drawing_mode == DrawingMode.BBOX:
            painter.setPen(ColorPen.YELLOW)
            rect = QRect(self.start_pos, self.end_pos)
            painter.drawRect(rect)

        # SELECT模式：繪製框選矩形（淡藍色）
        if (
            self.dragging_selection
            and self.selection_rect_start
            and self.current_mouse_pos
        ):
            sel_pen = QPen(QColor(100, 150, 255), 1, Qt.PenStyle.DashLine)
            painter.setPen(sel_pen)
            painter.setBrush(QColor(100, 150, 255, 40))
            sel_rect = QRect(
                self.selection_rect_start, self.current_mouse_pos
            ).normalized()
            painter.drawRect(sel_rect)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # 框選範圍的寬高與面積（原始pixel座標）
            orig_start = self._scale_to_original(sel_rect.topLeft())
            orig_end = self._scale_to_original(sel_rect.bottomRight())
            sel_w = abs(orig_end.x() - orig_start.x())
            sel_h = abs(orig_end.y() - orig_start.y())
            sel_text = f"{sel_w}x{sel_h}={sel_w * sel_h}"
            fm = painter.fontMetrics()
            sel_text_w = fm.horizontalAdvance(sel_text)
            sel_text_h = fm.height()
            sel_text_pos = sel_rect.bottomRight() + QPoint(5, 5)
            sel_bg = QRect(
                sel_text_pos,
                QPoint(sel_text_pos.x() + sel_text_w + 4, sel_text_pos.y() + sel_text_h),
            )
            painter.fillRect(sel_bg, QColor(0, 0, 0, 150))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                sel_text_pos + QPoint(2, sel_text_h - fm.descent()), sel_text
            )

        # SELECT模式：顯示選取的bbox或resize中的bbox尺寸資訊
        if self.drawing_mode == DrawingMode.SELECT:
            info_bbox = None
            info_pos = None
            if self.resizing and self.selected_bbox and self.current_mouse_pos:
                info_bbox = self.selected_bbox
                info_pos = self.current_mouse_pos
            elif self.select_type == "bbox" and 0 <= self.idx_focus_bbox < len(
                self.bboxes
            ):
                info_bbox = self.bboxes[self.idx_focus_bbox]
                # 顯示在bbox右下角
                info_pos = self._scale_to_widget(
                    QPoint(
                        info_bbox.x + info_bbox.width, info_bbox.y + info_bbox.height
                    )
                )
            if info_bbox and info_pos:
                text = f"{info_bbox.width}x{info_bbox.height}={info_bbox.width * info_bbox.height}"
                font_metrics = painter.fontMetrics()
                text_width = font_metrics.horizontalAdvance(text)
                text_height = font_metrics.height()
                text_pos = info_pos + QPoint(5, 5)
                bg_rect = QRect(
                    text_pos,
                    QPoint(text_pos.x() + text_width + 4, text_pos.y() + text_height),
                )
                painter.fillRect(bg_rect, QColor(0, 0, 0, 150))
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(
                    text_pos + QPoint(2, text_height - font_metrics.descent()), text
                )

        # 右下角顯示 bbox / polygon 數量
        n_bbox = len(self.bboxes)
        n_poly = len(self.polygons)
        if n_bbox or n_poly:
            parts = []
            if n_bbox:
                parts.append(f"bbox:{n_bbox}")
            if n_poly:
                parts.append(f"polygon:{n_poly}")
            count_text = "  ".join(parts)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(count_text)
            th = fm.height()
            margin = 6
            tx = self.width() - tw - margin * 2
            ty = self.height() - th - margin
            bg = QRect(tx - 2, ty, tw + margin, th + 2)
            painter.fillRect(bg, QColor(0, 0, 0, 140))
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(tx + 1, ty + th - fm.descent(), count_text)

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
            if self.drawing_mode == DrawingMode.SELECT:
                pos = event.pos()

                # 根據view_mode決定哪些標籤類型可被選取
                can_select_bbox = self.view_mode in (ViewMode.BBOX, ViewMode.ALL)
                can_select_polygon = self.view_mode in (ViewMode.SEG, ViewMode.ALL)

                # 1. 檢查所有polygon的頂點（直接拖曳，不需先選取）
                if can_select_polygon:
                    for idx, poly in enumerate(self.polygons):
                        vtx_idx = self._isNearPolygonVertex(pos, poly)
                        if vtx_idx >= 0:
                            self.idx_focus_polygon = idx
                            self.idx_focus_bbox = -1
                            self.select_type = "polygon"
                            self.dragging_vertex_idx = vtx_idx
                            self.update()
                            return

                # 2. 檢查所有bbox的旋轉控制點與角落（直接操作，不需先選取）
                for idx, bbox in enumerate(self.bboxes) if can_select_bbox else []:
                    if cfg.enable_obb and self._isOnRotationHandle(pos, bbox):
                        self.idx_focus_bbox = idx
                        self.idx_focus_polygon = -1
                        self.select_type = "bbox"
                        self.selected_bbox = bbox
                        self.selected_bbox.color_pen = ColorPen.YELLOW
                        self.rotating = True
                        self.original_angle = bbox.angle
                        pos_original = self._scale_to_original(pos)
                        center_x = bbox.x + bbox.width / 2
                        center_y = bbox.y + bbox.height / 2
                        dx = pos_original.x() - center_x
                        dy = pos_original.y() - center_y
                        self.rotation_start_angle = math.degrees(math.atan2(dy, dx))
                        self.update()
                        return

                    corner = self._isInCorner(pos, bbox)
                    if corner:
                        self.idx_focus_bbox = idx
                        self.idx_focus_polygon = -1
                        self.select_type = "bbox"
                        self.selected_bbox = bbox
                        self.selected_bbox.color_pen = ColorPen.YELLOW
                        self.resizing_corner = corner
                        self.original_bbox = (bbox.x, bbox.y, bbox.width, bbox.height)
                        self.start_pos = self._scale_to_original(pos)
                        if bbox.angle != 0:
                            rot_corners = self._getRotatedCorners(bbox)
                            corner_map = {
                                "top_left": 2,
                                "top_right": 3,
                                "bottom_right": 0,
                                "bottom_left": 1,
                            }
                            fixed_idx = corner_map.get(corner, 0)
                            self.fixed_corner_pos = rot_corners[fixed_idx]
                        self.resizing = True
                        self.update()
                        return

                # 3. 嘗試選取bbox（點擊內部）
                if can_select_bbox:
                    for idx, bbox in enumerate(self.bboxes):
                        if self._isInBboxArea(pos, bbox):
                            self.idx_focus_bbox = idx
                            self.idx_focus_polygon = -1
                            self.select_type = "bbox"
                            self.update()
                            return

                # 4. 嘗試選取polygon（含邊緣padding範圍）
                if can_select_polygon:
                    for idx, polygon in enumerate(self.polygons):
                        if self._isPointInPolygon(pos, polygon):
                            self.idx_focus_polygon = idx
                            self.idx_focus_bbox = -1
                            self.select_type = "polygon"
                            self.update()
                            return

                # 5. 沒有點到任何物件，開始框選（或取消選取）
                self.idx_focus_bbox = -1
                self.idx_focus_polygon = -1
                self.select_type = None
                self.selected_bbox_indices = set()
                self.selected_polygon_indices = set()
                self.selection_rect_start = pos
                self.dragging_selection = False
                self.update()
                return

            elif self.drawing_mode == DrawingMode.POLYGON:
                pos = event.pos()
                # Check if near first point to close polygon
                if (
                    len(self.current_polygon_points) >= 3
                    and self._distanceBetweenPoints(pos, self.current_polygon_points[0])
                    < POLYGON_CLOSE_THRESHOLD
                ):
                    # Close polygon - convert widget coords to original
                    points = []
                    for pt in self.current_polygon_points:
                        orig = self._scale_to_original(pt)
                        points.append((float(orig.x()), float(orig.y())))
                    self.polygons.append(
                        Polygon(
                            points,
                            self.app_state.last_used_label,
                            1.0,
                        )
                    )
                    self.idx_focus_polygon = len(self.polygons) - 1
                    self.current_polygon_points = []
                    g_param.user_labeling = True
                else:
                    # Add vertex
                    self.current_polygon_points.append(pos)
                self.update()
                return
            elif self.drawing_mode in [DrawingMode.MASK_DRAW, DrawingMode.MASK_ERASE]:
                self.start_pos = event.pos()
                self.end_pos = event.pos()
                self.drawing = True
                scaled_pos = self._scale_to_original(event.pos())
                self.last_pos = scaled_pos
                self.draw_on_mask(scaled_pos)
            elif self.drawing_mode == DrawingMode.MASK_FILL:
                self.fill_mask(event.pos())
            elif self.drawing_mode == DrawingMode.BBOX:
                # BBOX模式：純粹兩點建立，不處理resize/rotate
                if self.drawing:
                    # 第二次click：建立bbox
                    self.end_pos = event.pos()
                    self.drawing = False
                    self._finalizeBbox()
                else:
                    # 第一次click：記錄起點
                    self.start_pos = event.pos()
                    self.end_pos = event.pos()
                    self.drawing = True

        elif event.button() == Qt.MouseButton.RightButton:
            # BBOX兩點模式：右鍵取消進行中的繪製
            if self.drawing_mode == DrawingMode.BBOX and self.drawing:
                self.drawing = False
                self.start_pos = None
                self.end_pos = None
                self.update()
                return

            if self.drawing_mode == DrawingMode.POLYGON:
                if self.current_polygon_points:
                    # Cancel in-progress polygon
                    self.current_polygon_points = []
                    self.update()
                else:
                    # Delete existing polygon under cursor
                    pos = event.pos()
                    for polygon in reversed(self.polygons):
                        if self._isPointInPolygon(pos, polygon):
                            self.polygons.remove(polygon)
                            self.idx_focus_polygon = -1
                            self.update()
                            break
            elif self.drawing_mode == DrawingMode.BBOX:
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

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self.drawing_mode == DrawingMode.BBOX and self.drawing:
                self.drawing = False
                self.start_pos = None
                self.end_pos = None
                self.update()
                return
            if self.drawing_mode == DrawingMode.POLYGON and self.current_polygon_points:
                self.current_polygon_points = []
                self.update()
                return
        # 其他按鍵交給 parent (MainWindow) 處理
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event):
        self.current_mouse_pos = event.pos()

        # SELECT模式: 框選拖曳
        if (
            self.drawing_mode == DrawingMode.SELECT
            and self.selection_rect_start is not None
        ):
            if self._distanceBetweenPoints(self.selection_rect_start, event.pos()) > 5:
                self.dragging_selection = True
            self.update()
            return

        # SELECT模式: polygon頂點拖曳
        if self.drawing_mode == DrawingMode.SELECT and self.dragging_vertex_idx >= 0:
            if 0 <= self.idx_focus_polygon < len(self.polygons):
                orig_pos = self._scale_to_original(event.pos())
                self.polygons[self.idx_focus_polygon].points[
                    self.dragging_vertex_idx
                ] = (float(orig_pos.x()), float(orig_pos.y()))
                self.update()
                return

        if self.drawing_mode == DrawingMode.POLYGON:
            # Rubber band update for polygon drawing
            if self.current_polygon_points:
                self.update()
            return
        if self.drawing_mode in [DrawingMode.MASK_DRAW, DrawingMode.MASK_ERASE]:
            if self.drawing:
                scaled_pos = self._scale_to_original(event.pos())
                self.draw_on_mask(scaled_pos)
            return

        # SELECT模式: 滑鼠hover時顯示對應cursor
        if self.drawing_mode == DrawingMode.SELECT:
            cursor_changed = False
            if not self.resizing and not self.rotating and self.view_mode in (ViewMode.BBOX, ViewMode.ALL):
                for bbox in self.bboxes:
                    if cfg.enable_obb and self._isOnRotationHandle(event.pos(), bbox):
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

        if self.drawing and self.drawing_mode == DrawingMode.BBOX:
            # BBOX兩點模式：只在滑鼠按鍵未按住時（第一點已釋放後移動）才顯示預覽
            if not (event.buttons() & Qt.MouseButton.LeftButton):
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
            # SELECT模式: 框選結束
            if self.selection_rect_start is not None:
                if self.dragging_selection:
                    self._finalizeRectSelection()
                # 不論有無拖曳都要重置框選狀態
                self.selection_rect_start = None
                self.dragging_selection = False
                self.update()
                return

            # SELECT模式: polygon頂點拖曳結束
            if self.dragging_vertex_idx >= 0:
                self.dragging_vertex_idx = -1
                g_param.user_labeling = True
                self.update()
                return

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

            elif self.drawing:
                if self.drawing_mode == DrawingMode.BBOX:
                    # BBOX兩點模式：release不建立bbox，等待第二次click
                    return
                self.drawing = False

            self.completeMouseAction()

    def _finalizeBbox(self):
        """從start_pos和end_pos建立Bbox物件（兩點模式）"""
        if not self.start_pos or not self.end_pos:
            self.completeMouseAction()
            return

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
            self.completeMouseAction()
            return

        # 將視窗座標轉換為原始影像座標
        x1_original = self._scale_to_original(QPoint(x1, y1)).x()
        y1_original = self._scale_to_original(QPoint(x1, y1)).y()
        width_original = int(width * self.pixmap.width() / self.scaled_width)
        height_original = int(height * self.pixmap.height() / self.scaled_height)

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
        self.idx_focus_bbox = len(self.bboxes) - 1
        self.completeMouseAction()

    def completeMouseAction(self):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        for bbox in self.bboxes:
            bbox.color_pen = ColorPen.GREEN
        g_param.user_labeling = True
        self.update()

    def _distanceBetweenPoints(self, p1: QPoint, p2: QPoint) -> float:
        dx = p1.x() - p2.x()
        dy = p1.y() - p2.y()
        return (dx * dx + dy * dy) ** 0.5

    def _isPointInPolygon(self, pos: QPoint, polygon: Polygon) -> bool:
        """檢查widget座標的點是否在polygon內部或邊緣附近（含padding範圍）

        Args:
            pos: widget座標
            polygon: Polygon物件

        Returns:
            bool: 是否在polygon選取範圍內
        """
        if len(polygon.points) < 3:
            return False
        # Convert polygon points to widget coords for comparison
        poly_points = []
        for px, py in polygon.points:
            wpt = self._scale_to_widget(QPoint(int(px), int(py)))
            poly_points.append((float(wpt.x()), float(wpt.y())))
        np_poly = np.array(poly_points, dtype=np.float32)
        # measureDist=True 回傳有符號距離：正值=內部, 0=邊上, 負值=外部(距離邊緣的距離)
        dist = cv2.pointPolygonTest(np_poly, (float(pos.x()), float(pos.y())), True)
        return dist >= -POLYGON_SELECT_PADDING

    def set_view_mode(self, mode):
        self.view_mode = mode
        self.update()

    def wheelEvent(self, event):
        # event.angleDelta().y() > 0, 代表滑鼠滾輪往上滾
        if self.on_wheel_event_callback:
            self.on_wheel_event_callback(event.angleDelta().y() > 0)
