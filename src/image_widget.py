# 圖片畫布元件：負責繪製影像、bbox、polygon、mask，以及滑鼠互動（繪製、選取、拖曳、旋轉）
# 標註的每次變更都會先在 self.history 記下變更前的快照, 供 undo / redo 還原
# 影像的縮放與平移集中在 self.tf (ViewTransform); 原圖 <-> widget 的換算只走
# _scale_to_original / _scale_to_widget, 不在別處自行乘 zoom 或加 offset
# 更新日期: 2026-08-19
import math
import time
import xml.etree.ElementTree as ET
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
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
    EDGE_HANDLE_MIN_SPAN,
    MIN_RESIZE_LENGTH,
    POLYGON_CLOSE_THRESHOLD,
    POLYGON_SELECT_PADDING,
    POLYGON_VERTEX_RADIUS,
    ROTATION_HANDLE_DISTANCE,
    ROTATION_HANDLE_RADIUS,
    VIDEO_EXTS,
)
from src.utils.dynamic_settings import settings
from src.utils.file_handler import file_h
from src.utils.func import getXmlPath, imread_unicode
from src.utils.global_param import g_param
from src.utils.history import AnnotationHistory
from src.utils.img_handler import inferencer
from src.utils.logger import getUniqueLogger
from src.utils.model import Bbox, ColorPen, FileType, ModelType, Polygon, ViewMode
from src.utils.view_transform import ViewTransform

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
    # resize 控制點: 名稱 -> 在 bbox 局部座標中的位置比例 (-0.5 ~ +0.5)。
    # 這一份同時決定「畫在哪」「點得到哪」「拖了會動哪條邊界」, 三者不會走鐘。
    # 角落排在前面: 角落與邊的熱區在角附近會重疊, 角落優先才不會想拉兩個方向
    # 卻只拉到一個。
    _RESIZE_HANDLES = {
        "top_left": (-0.5, -0.5),
        "top_right": (0.5, -0.5),
        "bottom_right": (0.5, 0.5),
        "bottom_left": (-0.5, 0.5),
        "top": (0.0, -0.5),
        "bottom": (0.0, 0.5),
        "left": (-0.5, 0.0),
        "right": (0.5, 0.0),
    }

    # hover 到控制點上要顯示的游標。旋轉後的框不跟著轉游標: 那要按實際角度挑
    # 八個方向, 收益遠小於複雜度
    _HANDLE_CURSORS = {
        "top_left": Qt.CursorShape.SizeFDiagCursor,
        "bottom_right": Qt.CursorShape.SizeFDiagCursor,
        "top_right": Qt.CursorShape.SizeBDiagCursor,
        "bottom_left": Qt.CursorShape.SizeBDiagCursor,
        "top": Qt.CursorShape.SizeVerCursor,
        "bottom": Qt.CursorShape.SizeVerCursor,
        "left": Qt.CursorShape.SizeHorCursor,
        "right": Qt.CursorShape.SizeHorCursor,
    }

    def __init__(self, app_state: AppState):
        super().__init__()
        self.setMouseTracking(True)  # 即使沒按住按鍵也能追蹤滑鼠移動
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.app_state = app_state
        self.image_label = QLabel()
        self.pixmap = None
        self.bboxes: list[Bbox] = []
        self.drawing = False
        # BBOX 兩點模式進行中的兩個角 (原圖座標)。存原圖而非 widget 座標:
        # 畫到一半縮放或平移時, widget 座標會整組錯位
        self.draw_start: QPointF | None = None
        self.draw_end: QPointF | None = None

        self.idx_focus_bbox: int = -1
        self.resizing = False
        self.rotating = False  # 旋轉狀態
        self.selected_bbox: Optional[Bbox] = None
        self.resizing_handle: str | None = None  # 正在拖的控制點名稱
        self.original_bbox = None  # 拖曳前的 (x, y, width, height)
        self.original_angle = None  # 儲存原始角度
        self.rotation_start_angle = None  # 旋轉開始時的滑鼠角度
        self.current_mouse_pos = None  # 儲存滑鼠當前位置

        # Mask drawing properties
        self.drawing_mode = DrawingMode.SELECT
        self.mask_pixmap: QPixmap | None = None
        self.brush_size = 20
        self.last_pos = None

        # Polygon drawing state
        self.polygons: list[Polygon] = []
        # 進行中的 polygon 頂點 (原圖座標, 理由同 draw_start)
        self.current_polygon_points: list[QPointF] = []
        self.idx_focus_polygon: int = -1

        # SELECT mode state
        self.select_type: str | None = None  # 'bbox', 'polygon', 'multi'
        self.dragging_vertex_idx: int = -1  # 拖曳中的polygon頂點index
        # 拖曳移動選取中的標註 (整體平移, 不變形)
        self.moving: bool = False
        self.move_start_orig: QPointF | None = None  # 拖曳起點 (原圖座標)
        # 拖曳前的原始位置; 位移一律從這裡重算而非逐次累加, 免得取整誤差疊起來
        self.move_orig_boxes: list[tuple[int, tuple[int, int]]] = []
        self.move_orig_polys: list[tuple[int, list[tuple[float, float]]]] = []
        self._move_pushed = False  # 這次拖曳是否已經記過 undo

        # 框選 (multi-select) state
        self.selected_bbox_indices: set[int] = set()
        self.selected_polygon_indices: set[int] = set()
        self.selection_rect_start: QPoint | None = None  # 框選起點
        self.dragging_selection: bool = False

        self.view_mode = ViewMode.ALL
        self.list_fps = []

        # 標註的 undo / redo 歷史; 屬於目前這張影像, 換檔由 clearBboxes() 清空
        self.history = AnnotationHistory(cfg.undo_limit)

        # 檢視變換 (zoom + pan); 原圖 <-> widget 的換算全部經由這裡
        self.tf = ViewTransform()
        self._needs_fit = True  # 還沒對這張影像 fit 過
        self._panning = False
        self._pan_last = QPointF()
        # 縮小檢視時的預縮 pixmap 快取, 平移就只是 blit
        self._scaled_cache: QPixmap | None = None
        self._scaled_cache_key: tuple | None = None

        self.cv_img = None
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )  # 設定大小策略

        self.cap = None

        # Callbacks for main window communication
        self.on_mouse_press_callback = None
        self.on_wheel_event_callback = None
        self.on_view_changed_callback = None
        self.on_video_loaded_callback = None
        self.on_image_loaded_callback = None
        self.file_type = FileType.IMAGE
        self.fps = 30
        self.total_frames = 0  # 影片總幀數, 換檔時算一次
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
        on_view_changed=None,
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
        if on_view_changed:
            self.on_view_changed_callback = on_view_changed

    @property
    def scaled_width(self) -> int:
        """整張影像目前在畫面上的寬度 (px)"""
        return int(round(self.tf.span_x))

    @property
    def scaled_height(self) -> int:
        """整張影像目前在畫面上的高度 (px)"""
        return int(round(self.tf.span_y))

    def _scale_to_original(self, point):
        """widget 座標 -> 原圖座標 (取整)

        Args:
            point: widget 座標

        Returns:
            QPoint: 原圖座標; 尚未載入影像時原樣回傳
        """
        if not self.pixmap:
            return point
        o = self.tf.v2o(point.x(), point.y())
        return QPoint(int(o.x()), int(o.y()))

    def _scale_to_original_f(self, point) -> QPointF:
        """widget 座標 -> 原圖座標 (保留小數)

        繪製中的暫存點用這個: 反覆在兩個座標系之間取整會讓點慢慢漂掉。

        Args:
            point: widget 座標

        Returns:
            QPointF: 原圖座標
        """
        if not self.pixmap:
            return QPointF(point)
        return self.tf.v2o(point.x(), point.y())

    def _scale_to_widget(self, point):
        """原圖座標 -> widget 座標 (取整)

        Args:
            point: 原圖座標

        Returns:
            QPoint: widget 座標; 尚未載入影像時原樣回傳
        """
        if not self.pixmap:
            return point
        v = self.tf.o2v(point.x(), point.y())
        return QPoint(int(v.x()), int(v.y()))

    def _scale_to_widget_f(self, point) -> QPointF:
        """原圖座標 -> widget 座標 (保留小數)

        Args:
            point: 原圖座標

        Returns:
            QPointF: widget 座標
        """
        if not self.pixmap:
            return QPointF(point)
        return self.tf.o2v(point.x(), point.y())

    def _clampToImage(self, p: QPointF) -> QPointF:
        """把原圖座標夾進影像範圍內

        標註不該跑到影像外: 存進 VOC XML 會出現負座標或超過 size 的值, 下游轉
        YOLO 正規化後也會落在 [0,1] 之外。

        Args:
            p: 原圖座標

        Returns:
            QPointF: 夾進 [0, img_w] x [0, img_h] 的座標
        """
        if not self.pixmap:
            return p
        return QPointF(
            min(max(p.x(), 0.0), float(self.tf.img_w)),
            min(max(p.y(), 0.0), float(self.tf.img_h)),
        )

    def _clampBboxToImage(self, bbox: Bbox) -> None:
        """把未旋轉的 bbox 裁進影像範圍 (就地修改)

        是「裁切邊界」而不是「整個平移回來」: 框超出邊界通常代表物體被影像邊緣
        截斷, 裁掉溢出的部分後框仍貼著可見的那一半; 平移的話框會整個跑離物體。
        移動操作要的才是平移 (框不該變形), 那由 _clampMoveDelta 夾位移量處理。

        旋轉過的框不在這裡處理: 要裁的是旋轉後的四個角, 而把角拉回來會連帶改變
        角度或讓框變形 —— 那不是拖控制點時預期的結果。旋轉框改為在拖曳時夾游標
        位置, 邊界因此也不會離影像太遠。

        Args:
            bbox: 要裁的 bbox (就地修改)
        """
        if not self.pixmap or bbox.angle != 0:
            return
        x1 = min(max(bbox.x, 0), self.tf.img_w)
        y1 = min(max(bbox.y, 0), self.tf.img_h)
        x2 = min(max(bbox.x + bbox.width, 0), self.tf.img_w)
        y2 = min(max(bbox.y + bbox.height, 0), self.tf.img_h)
        bbox.x, bbox.y = x1, y1
        bbox.width = max(1, x2 - x1)
        bbox.height = max(1, y2 - y1)

    def _clampPolygonToImage(self, polygon: Polygon) -> None:
        """把 polygon 的所有頂點夾進影像範圍 (就地修改)

        Args:
            polygon: 要夾的 polygon (就地修改)
        """
        if not self.pixmap:
            return
        w, h = float(self.tf.img_w), float(self.tf.img_h)
        polygon.points = [
            (min(max(px, 0.0), w), min(max(py, 0.0), h)) for px, py in polygon.points
        ]

    def _clampAnnotationsToImage(self) -> None:
        """把目前所有標註夾進影像範圍

        畫面上不該出現超出影像的框, 不論它是怎麼來的 —— 讀進來的 XML、偵測結果、
        還是手動畫的。旋轉過的框 (OBB) 例外, 理由見 _clampBboxToImage。
        """
        for b in self.bboxes:
            self._clampBboxToImage(b)
        for p in self.polygons:
            self._clampPolygonToImage(p)

    def _min_zoom(self) -> float:
        """這張影像允許的最小 zoom

        比 fit 再小一截, 好把整張圖連邊界一起看進來。
        """
        return self.tf.fit_zoom(self.size()) * 0.25

    def _notifyViewChanged(self) -> None:
        """通知外面 (狀態列) 目前的縮放倍率"""
        if self.on_view_changed_callback:
            try:
                self.on_view_changed_callback(self.tf.zoom)
            except Exception as e:
                log.e(f"view changed callback 失敗: {e}")

    def fitView(self) -> None:
        """縮放到整張影像可見並置中"""
        if not self.pixmap:
            return
        self.tf.fit(self.size())
        self._needs_fit = False
        self._notifyViewChanged()
        self.update()

    def resizeEvent(self, event):
        """視窗尺寸變動時維持檢視合理: 還沒 fit 過就 fit, 否則只夾住 offset"""
        super().resizeEvent(event)
        if not self.pixmap:
            return
        if self._needs_fit:
            self.tf.fit(self.size())
            self._needs_fit = False
        else:
            self.tf.clamp_offset(self.size())
        self._notifyViewChanged()

    def _scaledPixmap(self) -> QPixmap:
        """縮小檢視時用的預縮 pixmap

        同一個 zoom 只縮一次, 之後平移就只是 blit; 換圖 (cacheKey 改變) 會自動失效。
        """
        key = (self.scaled_width, self.scaled_height, self.pixmap.cacheKey())
        if self._scaled_cache is None or self._scaled_cache_key != key:
            self._scaled_cache = self.pixmap.scaled(
                max(1, self.scaled_width),
                max(1, self.scaled_height),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._scaled_cache_key = key
        return self._scaled_cache

    def _hitSelectedGroup(self, pos: QPoint) -> bool:
        """游標是否落在目前多選的任一個標註上

        Args:
            pos: widget 座標

        Returns:
            bool: 是否命中
        """
        for i in self.selected_bbox_indices:
            if 0 <= i < len(self.bboxes) and self._isInBboxArea(pos, self.bboxes[i]):
                return True
        for i in self.selected_polygon_indices:
            if 0 <= i < len(self.polygons) and self._isPointInPolygon(
                pos, self.polygons[i]
            ):
                return True
        return False

    def _hitMovableTarget(self, pos: QPoint) -> bool:
        """游標是否落在「可以拖著移動」的選取標註上 (給 cursor 提示用)

        Args:
            pos: widget 座標

        Returns:
            bool: 是否命中
        """
        if self.select_type == "multi":
            return self._hitSelectedGroup(pos)
        if self.select_type == "bbox" and 0 <= self.idx_focus_bbox < len(self.bboxes):
            return self._isInBboxArea(pos, self.bboxes[self.idx_focus_bbox])
        if self.select_type == "polygon" and 0 <= self.idx_focus_polygon < len(
            self.polygons
        ):
            return self._isPointInPolygon(pos, self.polygons[self.idx_focus_polygon])
        return False

    def _beginMove(self, pos: QPoint, bbox_indices, poly_indices) -> None:
        """開始拖曳移動指定的標註

        Args:
            pos: 拖曳起點的 widget 座標
            bbox_indices: 要移動的 bbox index
            poly_indices: 要移動的 polygon index
        """
        self.move_orig_boxes = [
            (i, (self.bboxes[i].x, self.bboxes[i].y))
            for i in sorted(bbox_indices)
            if 0 <= i < len(self.bboxes)
        ]
        self.move_orig_polys = [
            (i, list(self.polygons[i].points))
            for i in sorted(poly_indices)
            if 0 <= i < len(self.polygons)
        ]
        if not self.move_orig_boxes and not self.move_orig_polys:
            return
        self.move_start_orig = self._scale_to_original_f(pos)
        self._move_pushed = False
        self.moving = True

    def _clampMoveDelta(self, dx: float, dy: float) -> tuple[float, float]:
        """把位移夾住, 讓整組標註不被拖出影像外

        對整組算一次而不是各自夾: 各自夾的話碰到邊界的那幾個會先停下、其他繼續走,
        整組的相對位置就散掉了。整組本來就超出影像時不夾, 否則一按下去就會被強制
        推回來 —— 讀檔與偵測都已經先夾過, 所以這只會發生在 OBB 上。

        Args:
            dx: 原圖座標的 x 位移
            dy: 原圖座標的 y 位移

        Returns:
            tuple: 夾過的 (dx, dy)
        """
        # 組裡有旋轉框就整組不夾: OBB 的角落本來就可能落在影像外 (貼著邊緣的斜物體),
        # 夾住會讓那種標註根本畫不出來。理由與 _clampBboxToImage 相同
        if any(
            0 <= idx < len(self.bboxes) and self.bboxes[idx].angle
            for idx, _ in self.move_orig_boxes
        ):
            return dx, dy

        lo_xs, hi_xs, lo_ys, hi_ys = [], [], [], []
        for idx, (ox, oy) in self.move_orig_boxes:
            if 0 <= idx < len(self.bboxes):
                lo_xs.append(ox)
                lo_ys.append(oy)
                hi_xs.append(ox + self.bboxes[idx].width)
                hi_ys.append(oy + self.bboxes[idx].height)
        for _, pts in self.move_orig_polys:
            for px, py in pts:
                lo_xs.append(px)
                lo_ys.append(py)
                hi_xs.append(px)
                hi_ys.append(py)
        if not lo_xs:
            return dx, dy

        lo_x, hi_x = -min(lo_xs), self.tf.img_w - max(hi_xs)
        if lo_x <= hi_x:
            dx = min(max(dx, lo_x), hi_x)
        lo_y, hi_y = -min(lo_ys), self.tf.img_h - max(hi_ys)
        if lo_y <= hi_y:
            dy = min(max(dy, lo_y), hi_y)
        return dx, dy

    def _applyMove(self, pos: QPoint) -> None:
        """把游標位移套用到所有拖曳中的標註 (整體平移, 不變形)

        Args:
            pos: 目前的 widget 座標
        """
        if self.move_start_orig is None:
            return
        now = self._scale_to_original_f(pos)
        dx, dy = self._clampMoveDelta(
            now.x() - self.move_start_orig.x(), now.y() - self.move_start_orig.y()
        )
        # 真的要動了才記 undo: 只是點一下選取的話, 連快照都不必建
        if not self._move_pushed:
            self.pushHistory()
            self._move_pushed = True
        for idx, (ox, oy) in self.move_orig_boxes:
            if 0 <= idx < len(self.bboxes):
                self.bboxes[idx].x = int(round(ox + dx))
                self.bboxes[idx].y = int(round(oy + dy))
        for idx, pts in self.move_orig_polys:
            if 0 <= idx < len(self.polygons):
                self.polygons[idx].points = [(px + dx, py + dy) for px, py in pts]
        self.update()

    def _beginRotate(self, pos: QPoint, bbox: Bbox) -> None:
        """開始拖曳旋轉握把

        Args:
            pos: 按下的 widget 座標
            bbox: 目標 bbox
        """
        self.selected_bbox = bbox
        bbox.color_pen = ColorPen.YELLOW
        self.pushHistory()
        self.rotating = True
        self.original_angle = bbox.angle
        pos_original = self._scale_to_original(pos)
        center_x = bbox.x + bbox.width / 2
        center_y = bbox.y + bbox.height / 2
        dx = pos_original.x() - center_x
        dy = pos_original.y() - center_y
        self.rotation_start_angle = math.degrees(math.atan2(dy, dx))
        self.update()

    def _beginResize(self, pos: QPoint, bbox: Bbox, handle: str) -> None:
        """開始拖曳控制點改大小

        Args:
            pos: 按下的 widget 座標
            bbox: 目標 bbox
            handle: 命中的控制點名稱 (見 _RESIZE_HANDLES)
        """
        self.selected_bbox = bbox
        bbox.color_pen = ColorPen.YELLOW
        self.pushHistory()
        self.resizing_handle = handle
        self.original_bbox = (bbox.x, bbox.y, bbox.width, bbox.height)
        self.resizing = True
        self.update()

    def _applyResize(self, pos: QPoint) -> None:
        """把游標位置套用到正在 resize 的 bbox

        角落與邊、旋轉與未旋轉走同一條路: 都在 bbox 的局部座標系 (以拖曳前的
        中心為原點、反旋轉) 裡改邊界。角落動兩條邊界、邊只動一條, 差別完全來自
        _RESIZE_HANDLES 裡的位置比例, 所以「畫在哪」與「拖了會動哪」不會走鐘。

        邊界一律從拖曳前的 original_bbox 重算而非逐次累加, 免得取整誤差疊起來。

        Args:
            pos: 目前的 widget 座標
        """
        bbox = self.selected_bbox
        if bbox is None or self.original_bbox is None:
            return
        ox, oy, ow, oh = self.original_bbox
        cx = ox + ow / 2
        cy = oy + oh / 2

        # 游標先夾進影像再轉局部座標: 邊界跟著游標走, 游標不出界框就不出界
        p = self._clampToImage(self._scale_to_original_f(pos))
        angle_rad = math.radians(bbox.angle)
        cos_i, sin_i = math.cos(-angle_rad), math.sin(-angle_rad)
        mdx, mdy = p.x() - cx, p.y() - cy
        mx = mdx * cos_i - mdy * sin_i
        my = mdx * sin_i + mdy * cos_i

        # 拖曳前的四條邊界 (局部座標), 再讓這個控制點負責的那幾條跟著游標走
        x_lo, x_hi = -ow / 2.0, ow / 2.0
        y_lo, y_hi = -oh / 2.0, oh / 2.0
        fx, fy = self._RESIZE_HANDLES.get(self.resizing_handle, (0.0, 0.0))
        if fx < 0:
            x_lo = mx
        elif fx > 0:
            x_hi = mx
        if fy < 0:
            y_lo = my
        elif fy > 0:
            y_hi = my

        # abs 而非夾住: 拖過對邊時框翻到另一側, 不會產生負的寬高寫進 XML
        new_w = max(abs(x_hi - x_lo), MIN_RESIZE_LENGTH)
        new_h = max(abs(y_hi - y_lo), MIN_RESIZE_LENGTH)
        lcx = (x_lo + x_hi) / 2.0
        lcy = (y_lo + y_hi) / 2.0

        # 新中心轉回全域座標
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        ncx = lcx * cos_a - lcy * sin_a + cx
        ncy = lcx * sin_a + lcy * cos_a + cy

        bbox.width = int(round(new_w))
        bbox.height = int(round(new_h))
        bbox.x = int(round(ncx - new_w / 2))
        bbox.y = int(round(ncy - new_h / 2))
        # 收尾再夾一次: 取整與 MIN_RESIZE_LENGTH 的下限都可能讓邊界溢出個位數 px
        self._clampBboxToImage(bbox)

    def _startPan(self, pos: QPointF) -> None:
        """開始平移檢視

        Args:
            pos: 起始的 widget 座標
        """
        self._panning = True
        self._pan_last = QPointF(pos)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _cancelInProgressDrawing(self) -> bool:
        """取消進行中的 BBOX / Polygon 繪製

        Returns:
            bool: 是否有取消掉進行中的繪製
        """
        if self.drawing_mode == DrawingMode.BBOX and self.drawing:
            self.drawing = False
            self.draw_start = None
            self.draw_end = None
            self.update()
            return True
        if self.drawing_mode == DrawingMode.POLYGON and self.current_polygon_points:
            self.current_polygon_points = []
            self.update()
            return True
        return False

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

    def _resizeHandlePoints(self, bbox: Bbox) -> dict[str, tuple[float, float]]:
        """這個 bbox 目前提供的 resize 控制點 -> 原圖座標

        繪製與命中判斷共用這一份, 兩邊才不會對不上。旋轉的框也走同一條路,
        差別只是位置先繞中心轉過。

        Args:
            bbox: 目標 bbox

        Returns:
            dict: 控制點名稱 -> (原圖 x, 原圖 y); 角落在前、邊在後
        """
        cx = bbox.x + bbox.width / 2
        cy = bbox.y + bbox.height / 2
        angle_rad = math.radians(bbox.angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        # 框在畫面上太窄/太扁時就不給那個方向的邊控制點
        wide_enough = self.tf.o2v_len(bbox.width) >= EDGE_HANDLE_MIN_SPAN
        tall_enough = self.tf.o2v_len(bbox.height) >= EDGE_HANDLE_MIN_SPAN

        pts = {}
        for name, (fx, fy) in self._RESIZE_HANDLES.items():
            if name in ("top", "bottom") and not wide_enough:
                continue
            if name in ("left", "right") and not tall_enough:
                continue
            dx = fx * bbox.width
            dy = fy * bbox.height
            pts[name] = (
                cx + dx * cos_a - dy * sin_a,
                cy + dx * sin_a + dy * cos_a,
            )
        return pts

    def _hitResizeHandle(self, pos, bbox: Bbox) -> str | None:
        """檢查滑鼠是否落在某個 resize 控制點上

        以螢幕像素判斷, 與繪製的方塊大小一致 —— 熱區不隨影像縮放脹縮, 也不會
        大於看得到的方塊。

        Args:
            pos: widget 座標
            bbox: 目標 bbox

        Returns:
            控制點名稱 (見 _RESIZE_HANDLES), 沒命中則 None
        """
        for name, (ox, oy) in self._resizeHandlePoints(bbox).items():
            wpt = self._scale_to_widget(QPoint(int(ox), int(oy)))
            rect = QRect(
                wpt.x() - CORNER_SIZE,
                wpt.y() - CORNER_SIZE,
                CORNER_SIZE * 2,
                CORNER_SIZE * 2,
            )
            if rect.contains(pos):
                return name
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
        self.pushHistory()

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
        else:
            # 沒選到東西時不留下空的 undo 步驟
            self.history.drop_last()
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

            # 讀進來的標註也夾: 畫面上不該出現超出影像的框, 不論它是怎麼來的。
            # 這裡刻意不設 user_labeling —— 只是翻過去看一眼就把檔案改掉不合理;
            # 一旦真的動過標註, 存檔時寫出去的自然就是夾過的值
            self._clampAnnotationsToImage()

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
        if inferencer.is_loading or not inferencer.is_loaded(model_type):
            return

        self.bboxes = []
        self.polygons = []

        if cfg.show_fps:
            t1 = time.time()

        if model_type == ModelType.YOLO:
            bboxes, polygons = inferencer.infer_yolo(self.cv_img)
            mode = settings.models.yolo_label_mode or "bbox"
            if mode == "seg":
                self.polygons = polygons
            elif mode == "bbox":
                self.bboxes = bboxes
            else:  # "all"
                self.bboxes = bboxes
                self.polygons = polygons
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

        # 偵測結果偶爾會溢出影像邊界, 一併夾回來 —— 那些框會直接被存成 XML
        self._clampAnnotationsToImage()

        # 過濾掉太小的偵測結果
        min_len = cfg.minimal_bbox_length
        self.bboxes = [
            b for b in self.bboxes
            if b.width >= min_len and b.height >= min_len
        ]
        self.polygons = [
            p for p in self.polygons
            if self._polygon_bbox_size(p) >= min_len
        ]

        if cfg.show_fps:
            self.list_fps.append(1 / (time.time() - t1))
            if len(self.list_fps) > 10:
                self.list_fps.pop(0)
            log.i(f"Inference avg fps: {sum(self.list_fps) / len(self.list_fps):.0f}")

        self.update()

    def get_total_msec(self) -> int:
        """取得影片總毫秒數; frame_count 不可靠時 seek 到尾端探測真實長度"""
        fps = self.fps or 30
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if total_frames and total_frames > 0:
            return int(total_frames * 1000 / fps)

        # 部分編碼/串流的 CAP_PROP_FRAME_COUNT 會回傳 0 或負值(偵測失敗),
        # 此時 seek 到影片尾端讀 POS_MSEC 取得真實長度, 完成後還原播放位置
        try:
            pos_frames = self.cap.get(cv2.CAP_PROP_POS_FRAMES)  # 記住目前位置
            self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            total_msec = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frames)  # 還原位置
        except Exception as e:
            log.error(f"probe video duration failed: {e}")
            total_msec = 0
        return total_msec

    def get_total_frames(self) -> int:
        """取得影片總幀數

        Returns:
            int: 總幀數; 非影片或探測失敗時回傳 0
        """
        if not self.cap or self.file_type != FileType.VIDEO:
            return 0
        try:
            total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if total and total > 0:
                return int(total)
            # 部分編碼/串流的 CAP_PROP_FRAME_COUNT 會回傳 0 或負值, 此時用
            # get_total_msec() 探測出來的長度換算 (它會 seek 到尾端再還原位置)
            fps = self.fps or 30
            return int(self.get_total_msec() * fps / 1000)
        except Exception as e:
            log.e(f"probe video frame count failed: {e}")
            return 0

    def current_frame_index(self) -> int:
        """取得目前的幀序號 (1-based)

        CAP_PROP_POS_FRAMES 是「下一個要讀的幀」, read() 之後剛好等於已讀到的
        那一幀的 1-based 編號, 與存檔檔名用的編號一致。

        Returns:
            int: 幀序號; 非影片或取不到時回傳 0
        """
        if not self.cap or self.file_type != FileType.VIDEO:
            return 0
        try:
            return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        except Exception as e:
            log.e(f"read video frame position failed: {e}")
            return 0

    def set_drawing_mode(self, mode: DrawingMode):
        """切換繪圖模式"""
        self.drawing_mode = mode
        self.setCursor(Qt.CursorShape.ArrowCursor)
        # 切換模式時重置所有 focus / 選取狀態
        self._resetSelection()
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

            # 換片前先關掉舊的解碼器, 否則連續切換影片會一直累積沒釋放的 cap
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            ret, self.cv_img = self.cap.read()
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            # log.info(f"Video FPS: {self.fps}")
            # on_video_loaded_callback 移到下方 cv_img 檢查之後才呼叫
        else:
            self.file_type = FileType.IMAGE
            # imread_unicode 支援中文路徑（cv2.imread 在 Windows 走 ANSI code page，中文會回 None）
            self.cv_img = imread_unicode(file_path)

            if self.on_image_loaded_callback:
                self.on_image_loaded_callback()
        if self.cv_img is None:
            log.w(f"load {file_path} failed")
            QMessageBox.critical(self, "Error", f"Failed to load file `{file_path}`")
            self.pixmap = None
            self.update()
            return

        # 總幀數在這裡算一次就好: 探測失敗時要 seek 到尾端, 不適合每幀重算
        self.total_frames = self.get_total_frames() if self.file_type == FileType.VIDEO else 0

        # 影片: 確認第一幀真的讀到了才通知外面。callback 會設定 progress bar 範圍,
        # 而它的副作用會覆寫 cv_img; 若擺在上面的檢查之前, 就分不清是影片真的載入失敗
        # 還是只是被 callback 改掉
        if self.file_type == FileType.VIDEO and self.on_video_loaded_callback:
            self.on_video_loaded_callback(self.get_total_msec())

        height, width, channel = self.cv_img.shape
        bytesPerLine = 3 * width
        qImg = QImage(
            self.cv_img.data, width, height, bytesPerLine, QImage.Format.Format_RGB888
        ).rgbSwapped()
        self.pixmap = QPixmap.fromImage(qImg)
        # 影像尺寸相同就保留 zoom/pan: 逐張比對同一個區域是這工具的主要用法,
        # 每換一張都跳回 fit 會讓人重新找一次位置。
        # 真正的 fit 延到 paintEvent: 這裡的 self.size() 可能還是 layout 前的
        # 暫時尺寸, 現在就 fit 會縮到錯的倍率
        if self.tf.set_image_size(self.pixmap.width(), self.pixmap.height()):
            self._needs_fit = True
        else:
            self.tf.clamp_offset(self.size())
        self._notifyViewChanged()
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

    def _resetSelection(self):
        """清掉所有 focus / 多選 / 拖曳中的狀態, 但不動標註本身

        undo/redo 與換檔都會讓既有 index 失效, 兩邊共用這一份重置。
        """
        self.idx_focus_bbox = -1
        self.idx_focus_polygon = -1
        self.select_type = None
        self.selected_bbox_indices = set()
        self.selected_polygon_indices = set()
        self.selected_bbox = None
        self.selection_rect_start = None
        self.dragging_selection = False
        self.dragging_vertex_idx = -1
        self.moving = False
        self.move_start_orig = None
        self.move_orig_boxes = []
        self.move_orig_polys = []
        self._move_pushed = False
        self.resizing = False
        self.rotating = False
        self.resizing_handle = None
        self.original_bbox = None

    def clearBboxes(self):
        """重置 Bounding Box 與 Polygon 資訊 (換檔 / 換幀), 並清空 undo 歷史

        歷史刻意不跨檔保留: 換檔會重讀該檔的 XML, 留著上一張的快照會讓 undo
        把別張圖的框寫進這一張。
        """
        self.bboxes = []
        self.polygons = []
        self.current_polygon_points = []
        self.draw_start = None
        self.draw_end = None
        self.drawing = False
        self._resetSelection()
        self.history.clear()

    def pushHistory(self):
        """在改動標註「之前」記下目前狀態

        所有會改到 bboxes / polygons 的入口都要先呼叫這個, undo 才不會漏步。
        """
        self.history.push(self.bboxes, self.polygons)

    def dropHistoryIfUnchanged(self):
        """連續操作結束時, 若標註其實沒變就撤掉開始時記的快照

        例如點到 resize 控制點卻沒有拖動; 不撤掉的話 undo 會出現按了沒反應的空步驟。
        """
        if self.history.matches_last(self.bboxes, self.polygons):
            self.history.drop_last()

    def _applySnapshot(self, restored: tuple) -> None:
        """把 undo / redo 取回的快照套用到畫布

        Args:
            restored: (bbox 清單, polygon 清單)
        """
        self.bboxes, self.polygons = restored
        self.current_polygon_points = []
        self._resetSelection()
        # 還原本身也是一次變更, 必須讓切檔流程把它寫回 XML
        g_param.user_labeling = True
        self.update()

    def undo(self) -> bool:
        """還原上一步標註變更

        Returns:
            bool: 是否有還原
        """
        restored = self.history.undo(self.bboxes, self.polygons)
        if restored is None:
            return False
        self._applySnapshot(restored)
        return True

    def redo(self) -> bool:
        """重做被還原的標註變更

        Returns:
            bool: 是否有重做
        """
        restored = self.history.redo(self.bboxes, self.polygons)
        if restored is None:
            return False
        self._applySnapshot(restored)
        return True

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if not self.pixmap:
            return
        # 第一次畫這張影像時才 fit: 到這裡 widget 已經是最終尺寸
        if self._needs_fit:
            self.tf.fit(self.size())
            self._needs_fit = False
            self._notifyViewChanged()

        # 依 zoom/pan 繪製影像。縮小時 (一般檢視狀態) 用預縮好的 pixmap, 平移
        # 只是 blit; 放大時只畫可見區域。兩者都不會重新解碼原圖
        img_rect = self.tf.image_rect()
        if self.tf.zoom < 1.0:
            painter.drawPixmap(
                int(img_rect.x()), int(img_rect.y()), self._scaledPixmap()
            )
            if self.mask_pixmap:
                painter.drawPixmap(
                    int(img_rect.x()),
                    int(img_rect.y()),
                    self.mask_pixmap.scaled(
                        max(1, self.scaled_width),
                        max(1, self.scaled_height),
                        Qt.AspectRatioMode.KeepAspectRatio,
                    ),
                )
        else:
            visible = QRectF(self.rect()).intersected(img_rect)
            if visible.isEmpty():
                return
            src = QRectF(
                self.tf.v2o_len(visible.x() - img_rect.x()),
                self.tf.v2o_len(visible.y() - img_rect.y()),
                self.tf.v2o_len(visible.width()),
                self.tf.v2o_len(visible.height()),
            )
            painter.drawPixmap(visible, self.pixmap, src)
            if self.mask_pixmap:
                painter.drawPixmap(visible, self.mask_pixmap, src)

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

        # 繪製選中 bbox 的控制點。條件跟 mousePressEvent 的熱區判斷一致 (SELECT 模式
        # 且 select_type == "bbox"), 否則會在拖不動的地方畫出控制點誤導人 ——
        # 例如 BBOX 模式畫完一個框後, 那個框仍是 focus 但該模式並不處理 resize
        if (
            _bboxes_to_draw
            and self.drawing_mode == DrawingMode.SELECT
            and self.select_type == "bbox"
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

            # SELECT模式下繪製 resize 控制點 (四角 + 四邊)
            if self.drawing_mode == DrawingMode.SELECT:
                painter.setPen(QPen(QColor(255, 255, 0), 1))
                painter.setBrush(QColor(255, 255, 255, 200))
                # 與命中判斷共用同一份幾何, 畫得到的就點得到
                for ox, oy in self._resizeHandlePoints(focused_bbox).values():
                    wpt = self._scale_to_widget(QPoint(int(ox), int(oy)))
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
            # 頂點存的是原圖座標, 畫之前先換算到 widget
            wpts = [self._scale_to_widget_f(pt) for pt in self.current_polygon_points]
            # Draw lines between existing points
            for i in range(len(wpts) - 1):
                painter.drawLine(wpts[i], wpts[i + 1])

            # Draw vertex dots
            for i, pt in enumerate(wpts):
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
            if self.current_mouse_pos and wpts:
                painter.setPen(QPen(QColor(255, 0, 0, 128), 1, Qt.PenStyle.DashLine))
                painter.drawLine(wpts[-1], QPointF(self.current_mouse_pos))

        # BBOX兩點模式：繪製中的黃色矩形預覽
        if (
            self.drawing
            and self.drawing_mode == DrawingMode.BBOX
            and self.draw_start is not None
            and self.draw_end is not None
        ):
            painter.setPen(ColorPen.YELLOW)
            rect = QRectF(
                self._scale_to_widget_f(self.draw_start),
                self._scale_to_widget_f(self.draw_end),
            ).normalized()
            painter.drawRect(rect)

            # 寬高與面積直接由原圖座標算, 不必再從畫面反推一次
            box_w = int(abs(self.draw_end.x() - self.draw_start.x()))
            box_h = int(abs(self.draw_end.y() - self.draw_start.y()))
            box_text = f"{box_w}x{box_h}={box_w * box_h}"
            fm = painter.fontMetrics()
            box_text_w = fm.horizontalAdvance(box_text)
            box_text_h = fm.height()
            # label 顯示在右下角點的右上, 與框選一致, 以免拉到畫面底部被截斷
            box_text_pos = rect.bottomRight().toPoint() + QPoint(
                5, -(box_text_h + 5)
            )
            box_bg = QRect(
                box_text_pos,
                QPoint(
                    box_text_pos.x() + box_text_w + 4,
                    box_text_pos.y() + box_text_h,
                ),
            )
            painter.fillRect(box_bg, QColor(0, 0, 0, 150))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                box_text_pos + QPoint(2, box_text_h - fm.descent()), box_text
            )

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
            # label 顯示在右下角這個點的右上, 以免框選拉到畫面底部時 label 被截在畫面外看不到
            sel_text_pos = sel_rect.bottomRight() + QPoint(5, -(sel_text_h + 5))
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
                # label 顯示在右下角點的右上, 與框選一致, 以免拉到畫面底部被截斷
                text_pos = info_pos + QPoint(5, -(text_height + 5))
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

        # 中鍵與右鍵都是平移。右鍵只做這件事: 先前還兼「原地 click 取消進行中的
        # 繪製」, 但畫 polygon 畫到一半誤點右鍵就整個重來, 代價太大 —— 取消一律
        # 走 Esc
        if self.pixmap and event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
        ):
            self._startPan(event.position())
            return

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
                            self.pushHistory()
                            self.dragging_vertex_idx = vtx_idx
                            self.update()
                            return

                # 2. 只有「選取中」的那個 bbox 才吃旋轉握把與角落 resize。
                #    控制點只畫給選取中的框, 熱區跟著限縮才不會「看不到卻踩得到」
                #    —— 框重疊時最容易誤把旁邊那個拖變形
                focused_bbox = None
                if (
                    can_select_bbox
                    and self.select_type == "bbox"
                    and 0 <= self.idx_focus_bbox < len(self.bboxes)
                ):
                    focused_bbox = self.bboxes[self.idx_focus_bbox]
                if focused_bbox is not None:
                    if cfg.enable_obb and self._isOnRotationHandle(pos, focused_bbox):
                        self._beginRotate(pos, focused_bbox)
                        return
                    handle = self._hitResizeHandle(pos, focused_bbox)
                    if handle:
                        self._beginResize(pos, focused_bbox, handle)
                        return

                # 3. 多選狀態下點在任一選取項上 → 整批一起移動, 不打散選取
                if self.select_type == "multi" and self._hitSelectedGroup(pos):
                    self._beginMove(
                        pos, self.selected_bbox_indices, self.selected_polygon_indices
                    )
                    self.update()
                    return

                # 4. 嘗試選取bbox（點擊內部）, 選到後可直接拖著移動
                if can_select_bbox:
                    for idx, bbox in enumerate(self.bboxes):
                        if self._isInBboxArea(pos, bbox):
                            self.idx_focus_bbox = idx
                            self.idx_focus_polygon = -1
                            self.select_type = "bbox"
                            self.selected_bbox_indices = set()
                            self.selected_polygon_indices = set()
                            self._beginMove(pos, [idx], [])
                            self.update()
                            return

                # 5. 嘗試選取polygon（含邊緣padding範圍）, 同樣可直接拖著移動
                if can_select_polygon:
                    for idx, polygon in enumerate(self.polygons):
                        if self._isPointInPolygon(pos, polygon):
                            self.idx_focus_polygon = idx
                            self.idx_focus_bbox = -1
                            self.select_type = "polygon"
                            self.selected_bbox_indices = set()
                            self.selected_polygon_indices = set()
                            self._beginMove(pos, [], [idx])
                            self.update()
                            return

                # 6. 沒有點到任何物件，開始框選（或取消選取）
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
                # 閉合判定用螢幕距離 (門檻是螢幕 px), 所以先把第一點換到 widget
                if (
                    len(self.current_polygon_points) >= 3
                    and self._distanceBetweenPoints(
                        pos, self._scale_to_widget_f(self.current_polygon_points[0])
                    )
                    < POLYGON_CLOSE_THRESHOLD
                ):
                    # 頂點本來就是原圖座標, 直接落檔
                    points = [
                        (float(pt.x()), float(pt.y()))
                        for pt in self.current_polygon_points
                    ]

                    # 檢查 polygon bounding box 是否達到最小尺寸 (以原圖像素為準)
                    xs = [x for x, _ in points]
                    ys = [y for _, y in points]
                    poly_w = max(xs) - min(xs)
                    poly_h = max(ys) - min(ys)
                    if poly_w < cfg.minimal_bbox_length or poly_h < cfg.minimal_bbox_length:
                        self.current_polygon_points = []
                        self.update()
                        return

                    self.pushHistory()
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
                    # Add vertex (存原圖座標, 夾在影像內)
                    self.current_polygon_points.append(
                        self._clampToImage(self._scale_to_original_f(pos))
                    )
                self.update()
                return
            elif self.drawing_mode in [DrawingMode.MASK_DRAW, DrawingMode.MASK_ERASE]:
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
                    self.draw_end = self._clampToImage(
                        self._scale_to_original_f(event.pos())
                    )
                    self.drawing = False
                    self._finalizeBbox()
                else:
                    # 第一次click：記錄起點 (原圖座標, 夾在影像內)。在記錄時就夾,
                    # 拖曳中的預覽框與面積數字才會跟最後建出來的框一致
                    self.draw_start = self._clampToImage(
                        self._scale_to_original_f(event.pos())
                    )
                    self.draw_end = QPointF(self.draw_start)
                    self.drawing = True

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self._cancelInProgressDrawing():
                return
            # 沒有進行中的繪製時, Esc 用來取消選取
            if self.select_type is not None:
                self._resetSelection()
                self.update()
                return
        # 其他按鍵交給 parent (MainWindow) 處理
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event):
        self.current_mouse_pos = event.pos()

        # 平移中: 只動檢視, 完全不碰標註
        if self._panning:
            pos = event.position()
            self.tf.pan_by(
                pos.x() - self._pan_last.x(), pos.y() - self._pan_last.y()
            )
            self._pan_last = QPointF(pos)
            self.tf.clamp_offset(self.size())
            self._notifyViewChanged()
            self.update()
            return

        # SELECT模式: 拖曳移動選取中的標註
        if self.moving:
            self._applyMove(event.pos())
            return

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
                orig_pos = self._clampToImage(
                    self._scale_to_original_f(event.pos())
                )
                self.polygons[self.idx_focus_polygon].points[
                    self.dragging_vertex_idx
                ] = (orig_pos.x(), orig_pos.y())
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
            idle = not self.resizing and not self.rotating and not self.moving
            if idle and self.view_mode in (ViewMode.BBOX, ViewMode.ALL):
                # 只看選取中的那個框: 控制點熱區已限縮到它, cursor 要跟著一致,
                # 否則會在沒有控制點的地方冒出縮放游標
                focused_bbox = (
                    self.bboxes[self.idx_focus_bbox]
                    if self.select_type == "bbox"
                    and 0 <= self.idx_focus_bbox < len(self.bboxes)
                    else None
                )
                if focused_bbox is not None:
                    if cfg.enable_obb and self._isOnRotationHandle(
                        event.pos(), focused_bbox
                    ):
                        self.setCursor(Qt.CursorShape.CrossCursor)
                        cursor_changed = True
                    else:
                        handle = self._hitResizeHandle(event.pos(), focused_bbox)
                        shape = self._HANDLE_CURSORS.get(handle)
                        if shape is not None:
                            self.setCursor(shape)
                            cursor_changed = True
            # 落在選取中的標註上 → 提示可以拖著整體移動
            if not cursor_changed and idle and self._hitMovableTarget(event.pos()):
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                cursor_changed = True
            if not cursor_changed and idle:
                self.setCursor(Qt.CursorShape.ArrowCursor)

        if self.drawing and self.drawing_mode == DrawingMode.BBOX:
            # BBOX兩點模式：只在滑鼠按鍵未按住時（第一點已釋放後移動）才顯示預覽
            if not (event.buttons() & Qt.MouseButton.LeftButton):
                self.draw_end = self._clampToImage(
                    self._scale_to_original_f(event.pos())
                )
        elif self.resizing:
            self._applyResize(event.pos())

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
        # 平移結束 (中鍵或右鍵拖曳)
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            # SELECT模式: 拖曳移動結束
            if self.moving:
                self.moving = False
                self.move_start_orig = None
                self.move_orig_boxes = []
                self.move_orig_polys = []
                if self._move_pushed:
                    # 拖出去又拖回原位時不留下空的 undo 步驟
                    self.dropHistoryIfUnchanged()
                    g_param.user_labeling = True
                self._move_pushed = False
                self.update()
                return

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
                self.dropHistoryIfUnchanged()
                g_param.user_labeling = True
                self.update()
                return

            if self.resizing:
                self.resizing = False
                self.selected_bbox = None
                self.resizing_handle = None
                self.original_bbox = None
                self.dropHistoryIfUnchanged()

            elif self.rotating:
                self.rotating = False
                self.selected_bbox = None
                self.original_angle = None
                self.rotation_start_angle = None
                self.dropHistoryIfUnchanged()

            elif self.drawing:
                if self.drawing_mode == DrawingMode.BBOX:
                    # BBOX兩點模式：release不建立bbox，等待第二次click
                    return
                self.drawing = False

            self.completeMouseAction()

    def _finalizeBbox(self):
        """從 draw_start / draw_end 建立 Bbox（兩點模式）

        兩個端點本來就是原圖座標, 這裡不再做座標換算 —— 少一次換算就少一個
        「畫到一半縮放」會錯位的機會。
        """
        if self.draw_start is None or self.draw_end is None:
            self.completeMouseAction()
            return

        x1 = int(min(self.draw_start.x(), self.draw_end.x()))
        y1 = int(min(self.draw_start.y(), self.draw_end.y()))
        x2 = int(max(self.draw_start.x(), self.draw_end.x()))
        y2 = int(max(self.draw_start.y(), self.draw_end.y()))
        width = x2 - x1
        height = y2 - y1

        # 檢查寬高是否大於最小限制 (以原圖像素為準, 不受顯示縮放影響)
        if width < cfg.minimal_bbox_length or height < cfg.minimal_bbox_length:
            self.draw_start = None
            self.draw_end = None
            self.completeMouseAction()
            return

        self.pushHistory()
        self.bboxes.append(
            Bbox(
                x1,
                y1,
                width,
                height,
                self.app_state.last_used_label,
                1.0,
            )
        )
        self.idx_focus_bbox = len(self.bboxes) - 1
        self.draw_start = None
        self.draw_end = None
        self.completeMouseAction()

    def completeMouseAction(self):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        for bbox in self.bboxes:
            bbox.color_pen = ColorPen.GREEN
        g_param.user_labeling = True
        self.update()

    @staticmethod
    def _polygon_bbox_size(polygon: Polygon) -> float:
        """回傳 polygon bounding box 的較短邊長度（原始影像座標）"""
        xs = [x for x, _ in polygon.points]
        ys = [y for _, y in polygon.points]
        return min(max(xs) - min(xs), max(ys) - min(ys))

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
        """滾輪以游標為錨點縮放; Ctrl+滾輪 則交給外面切換檔案"""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        # Ctrl+滾輪保留原本「滾輪翻檔」的手感, 熟手的肌肉記憶不會白費
        # (delta > 0 代表往上滾 = 上一個檔案)
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if self.on_wheel_event_callback:
                self.on_wheel_event_callback(delta > 0)
            event.accept()
            return
        if not self.pixmap:
            return
        # 1.0015**120 ~= 1.20: 一個滑鼠刻度約 20%, 觸控板的小刻度也能平滑縮放
        if self.tf.zoom_by(
            1.0015**delta, QPointF(event.position()), self._min_zoom()
        ):
            self.tf.clamp_offset(self.size())
            self._notifyViewChanged()
            self.update()
        event.accept()
