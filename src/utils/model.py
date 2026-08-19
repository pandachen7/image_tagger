# 標註與播放狀態的資料模型。Bbox / Polygon 另提供 snapshot / from_snapshot,
# 供 undo 歷史以「不含 Qt 物件的純資料」保存與還原 (見 src/utils/history.py)。
# 更新日期: 2026-08-18
from PyQt6.QtGui import QColor, QPen


class ColorPen:
    RED = QPen(QColor(255, 0, 0), 2)
    GREEN = QPen(QColor(0, 255, 0), 1)
    # GREEN_BOLD = QPen(QColor(0, 255, 0), 2)
    ORANGE = QPen(QColor(255, 128, 0), 1)
    YELLOW = QPen(QColor(255, 255, 0), 1)


class Bbox:
    """單一矩形標註 (原始影像座標)"""

    def __init__(self, x, y, width, height, label, confidence=-1.0, angle=0.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.confidence = confidence
        self.angle = angle  # 旋轉角度（順時針，單位：度）
        self.color_pen = ColorPen.GREEN

    def snapshot(self) -> tuple:
        """轉成純資料 tuple, 供 undo 歷史保存

        color_pen 刻意不入快照: 它是「選取中/拖曳中」的顯示狀態而非標註內容,
        還原時一律回到預設色, 免得把拖曳中的黃色一起還原回來。

        Returns:
            tuple: (x, y, width, height, label, confidence, angle)
        """
        return (
            self.x,
            self.y,
            self.width,
            self.height,
            self.label,
            self.confidence,
            self.angle,
        )

    @classmethod
    def from_snapshot(cls, snap: tuple) -> "Bbox":
        """由 snapshot() 的 tuple 重建 Bbox

        Args:
            snap: snapshot() 產生的 tuple

        Returns:
            Bbox: 重建的物件 (color_pen 為預設色)
        """
        return cls(*snap)


class Polygon:
    """單一多邊形標註 (原始影像座標)"""

    def __init__(self, points, label, confidence=-1.0):
        self.points = points  # list[(float, float)] in original image coords
        self.label = label
        self.confidence = confidence
        self.color_pen = ColorPen.ORANGE

    def snapshot(self) -> tuple:
        """轉成純資料 tuple, 供 undo 歷史保存

        points 轉成 tuple 而非直接共用同一個 list: 頂點拖曳是就地改 points[i],
        共用容器的話快照會跟著被改掉, undo 就還原不回拖曳前的位置。

        Returns:
            tuple: (points tuple, label, confidence)
        """
        return (tuple(self.points), self.label, self.confidence)

    @classmethod
    def from_snapshot(cls, snap: tuple) -> "Polygon":
        """由 snapshot() 的 tuple 重建 Polygon

        Args:
            snap: snapshot() 產生的 tuple

        Returns:
            Polygon: 重建的物件 (color_pen 為預設色)
        """
        points, label, confidence = snap
        return cls(list(points), label, confidence)


class FileType:
    VIDEO = "video"
    IMAGE = "image"


class ShowImageCmd:
    NEXT = "next"
    PREV = "prev"
    FIRST = "first"
    LAST = "last"

    SAME_INDEX = "same_index"


class PlayState:
    PLAY = "play"
    PAUSE = "pause"
    STOP = "stop"


class ViewMode:
    ALL = "all"
    BBOX = "bbox"
    SEG = "seg"


class ModelType:
    NONE = "none"
    YOLO = "yolo"
    SAM3 = "sam3"
