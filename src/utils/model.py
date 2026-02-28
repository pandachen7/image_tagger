from PyQt6.QtGui import QColor, QPen


class ColorPen:
    RED = QPen(QColor(255, 0, 0), 2)
    GREEN = QPen(QColor(0, 255, 0), 1)
    # GREEN_BOLD = QPen(QColor(0, 255, 0), 2)
    ORANGE = QPen(QColor(255, 128, 0), 1)
    YELLOW = QPen(QColor(255, 255, 0), 1)


class Bbox:
    def __init__(self, x, y, width, height, label, confidence=-1.0, angle=0.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.confidence = confidence
        self.angle = angle  # 旋轉角度（順時針，單位：度）
        self.color_pen = ColorPen.GREEN


class Polygon:
    def __init__(self, points, label, confidence=-1.0):
        self.points = points  # list[(float, float)] in original image coords
        self.label = label
        self.confidence = confidence
        self.color_pen = ColorPen.ORANGE


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
