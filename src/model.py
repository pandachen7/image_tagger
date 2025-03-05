class Bbox:
    def __init__(self, x, y, width, height, label, confidence=-1.0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.confidence = confidence
        self.label_color = "green"


class FileType:
    VIDEO = 'video'
    IMAGE = 'image'
