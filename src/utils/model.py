from PyQt6.QtGui import QColor, QPen


class ColorPen:
    RED = QPen(QColor(255, 0, 0), 2)
    GREEN = QPen(QColor(0, 255, 0), 1)
    # GREEN_BOLD = QPen(QColor(0, 255, 0), 2)
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
