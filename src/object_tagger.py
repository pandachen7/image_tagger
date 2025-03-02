import cv2
import os
import sys

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QFileDialog,
                             QMenuBar, QToolBar, QStatusBar, QVBoxLayout,
                             QListView, QMessageBox, QInputDialog, QSizePolicy)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QAction, QEnterEvent
from PyQt6.QtCore import Qt, QAbstractListModel, QTimer, QRect, QPoint
from ultralytics import YOLO

from src.model import Bbox

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Object Tagger")

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
        self.addToolBar(self.toolbar)

        # 狀態列
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")

        # 中央 Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.image_widget = ImageWidget()
        self.main_layout.addWidget(self.image_widget)

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
            file_path = self.file_handler.current_image_path().replace(".jpg", ".xml") # 假設都是 .jpg
            bboxes = self.image_widget.bboxes
            xml_content = self.file_handler.generate_voc_xml(bboxes, self.file_handler.current_image_path())
            with open(file_path, "w") as f:
                f.write(xml_content)
            if self.is_auto_save():
                self.statusbar.showMessage(f"Annotations auto saved to {file_path}")
            else:
                self.statusbar.showMessage(f"Annotations saved to {file_path}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Right or event.key() == Qt.Key.Key_PageDown:
            self.save_annotations()
            self.next_image()
        elif event.key() == Qt.Key.Key_Left or event.key() == Qt.Key.Key_PageUp:
            self.save_annotations()
            self.prev_image()
        elif event.key() == Qt.Key.Key_Q:
            self.close()
        elif event.key() == Qt.Key.Key_A:
            self.toggle_auto_save()

class ImageWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image_label = QLabel()
        self.pixmap = None
        self.bboxes: list[Bbox] = []
        self.start_pos = None
        self.end_pos = None
        self.drawing = False

        self.model: None|YOLO = None

        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # 設定大小策略

    def _scale_to_original(self, point):
        if self.pixmap:
            scale_x = self.pixmap.width() / self.width()
            scale_y = self.pixmap.height() / self.height()
            return QPoint(int(point.x() * scale_x), int(point.y() * scale_y))
        else:
            return point

    def _scale_to_widget(self, point):
        if self.pixmap:
            scale_x = self.width() / self.pixmap.width()
            scale_y = self.height() / self.pixmap.height()
            return QPoint(int(point.x() * scale_x), int(point.y() * scale_y))
        else:
            return point

    def load_image(self, image_path):
        self.image = cv2.imread(image_path)
        height, width, channel = self.image.shape
        bytesPerLine = 3 * width
        qImg = QImage(self.image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888).rgbSwapped()
        self.pixmap = QPixmap.fromImage(qImg)
        self.bboxes = []  # 清空 Bounding Box

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
        self.update() # 觸發 paintEvent

    def paintEvent(self, event):
        if self.pixmap:
            painter = QPainter(self)
            # 計算繪製區域，將縮放後的影像置於左上
            scaled_pixmap = self.pixmap.scaled(self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatio)
            painter.drawPixmap(0, 0, scaled_pixmap)

            # 繪製 Bounding Box
            pen = QPen(QColor(0, 255, 0), 2)  # 綠色，寬度 2
            painter.setPen(pen)
            for bbox in self.bboxes:
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
                                         + int(text_width * self.width() / self.pixmap.width()),
                                        self._scale_to_widget(qpt_text).y()
                                         + int(text_height * self.height() / self.pixmap.height())))
                painter.fillRect(bg_rect, QColor(0, 0, 0, 127))  # 黑色半透明底色

                # 繪製文字 (調整位置)
                painter.drawText(self._scale_to_widget(QPoint(bbox.x, bbox.y - text_height)), text)

            if self.drawing:
                pen = QPen(QColor(255, 0, 0), 2)  # 繪製中的 Bounding Box 用紅色
                painter.setPen(pen)
                rect = QRect(self.start_pos, self.end_pos)
                painter.drawRect(rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
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
        if self.drawing:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            # 取得座標 (視窗座標)
            x1, y1 = self.start_pos.x(), self.start_pos.y()
            x2, y2 = self.end_pos.x(), self.end_pos.y()

            # 取得寬高 (視窗座標)
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            # 檢查寬高是否大於最小限制
            if width > 5 and height > 5:  # 最小 5x5
                # 取得標籤
                label, ok = QInputDialog.getText(self, 'Input', 'Enter label name:', text="object")
                if ok:
                    # 將視窗座標轉換為原始影像座標
                    x1_original, y1_original = self._scale_to_original(QPoint(x1, y1)).x(), self._scale_to_original(QPoint(x1, y1)).y()
                    width_original, height_original = int(width * self.pixmap.width() / self.width()), int(height * self.pixmap.height() / self.height())
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
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff')):
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
