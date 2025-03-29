import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
from PyQt6.QtCore import QAbstractListModel, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHeaderView,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStatusBar,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from ruamel.yaml import YAML
from ultralytics import YOLO

from src.const import ALL_EXTS
from src.func import getXmlPath
from src.image_widget import ImageWidget
from src.model import FileType, ShowImageCmd

yaml = YAML()

YOLO_LABELS_FOLDER = "labels"


class Settings:
    """
    程式正常關閉後自動儲存
    """

    data = {"model_path": None, "folder_path": None, "file_index": 0, "categories": {}}
    try:
        with open("config/settings.yaml", "r", encoding="utf-8") as f:
            yaml_settings = yaml.load(f)
        data["model_path"] = yaml_settings.get("model_path", None)
        data["folder_path"] = yaml_settings.get("folder_path", None)
        data["file_index"] = int(yaml_settings.get("file_index", 0))
        data["categories"] = yaml_settings.get("categories", {})

        # check validation
        if data["model_path"] is None or Path(data["model_path"]).is_file() is False:
            data["model_path"] = None

        if data["folder_path"] is None or Path(data["folder_path"]).is_dir() is False:
            data["folder_path"] = None

        if not isinstance(data["file_index"], int) or data["file_index"] < 0:
            data["file_index"] = 0

    except Exception as e:
        print(f"Error parsing config/label.yaml: {e}")


class CategorySettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Categories")
        self.categories = Settings.data["categories"]

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["Category Name", "Index"])
        self.table_widget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_category)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_category)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_categories)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.table_widget)
        main_layout.addLayout(button_layout)

        self.load_categories()

    def load_categories(self):
        self.table_widget.setRowCount(0)
        for name, index in self.categories.items():
            self.add_row(name, str(index))

    def add_row(self, name="", index=""):
        row_count = self.table_widget.rowCount()
        self.table_widget.insertRow(row_count)
        name_item = QTableWidgetItem(name)
        index_item = QTableWidgetItem(index)
        self.table_widget.setItem(row_count, 0, name_item)
        self.table_widget.setItem(row_count, 1, index_item)

    def add_category(self):
        self.add_row()

    def delete_category(self):
        selected_row = self.table_widget.currentRow()
        if selected_row >= 0:
            self.table_widget.removeRow(selected_row)

    def save_categories(self):
        categories = {}
        for row in range(self.table_widget.rowCount()):
            name_item = self.table_widget.item(row, 0)
            index_item = self.table_widget.item(row, 1)
            if name_item and index_item:
                name = name_item.text()
                index_str = index_item.text()
                if name and index_str.isdigit():
                    categories[name] = int(index_str)
        Settings.data["categories"] = categories
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self):

        super().__init__()

        self.setWindowTitle("Image Tagger")

        # 設定最小尺寸
        self.setMinimumSize(500, 500)

        # 自動
        self.auto_save = False
        self.auto_detect = False

        # 自動儲存
        self.auto_save_action = QAction("&Auto Save", self)
        self.auto_save_action.setCheckable(True)
        self.auto_save_action.triggered.connect(self.toggle_auto_save)

        # 自動儲存間隔 (秒)
        self.auto_save_per_second = -1

        # 自動使用偵測 (GPU不好速度就會慢)
        self.auto_detect_action = QAction("&Auto Detect", self)
        self.auto_detect_action.setCheckable(True)
        self.auto_detect_action.triggered.connect(self.toggle_auto_detect)

        # 檔案相關動作
        self.open_folder_action = QAction("&Open Folder", self)
        self.open_folder_action.triggered.connect(self.open_folder)

        # 工具列
        self.toolbar = QToolBar()
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, self.toolbar)

        self.toolbar_auto_save = QAction("&Auto Save", self)
        self.toolbar_auto_save.triggered.connect(self.toggle_auto_save)
        self.toolbar.addAction(self.auto_save_action)

        self.toolbar_auto_detect = QAction("&Auto Detect", self)
        self.toolbar_auto_detect.triggered.connect(self.toggle_auto_detect)
        self.toolbar.addAction(self.auto_detect_action)

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

        # 建立 Bounding Box 列表
        # self.bbox_list_view = QListView()
        # self.bbox_list_model = BboxListModel([])
        # self.bbox_list_view.setModel(self.bbox_list_model)
        # self.main_layout.addWidget(self.bbox_list_view)

        self.save_action = QAction("&Save", self)
        self.save_action.triggered.connect(self.save_annotations)
        self.toolbar.addAction(self.save_action)

        # 播放控制
        self.play_pause_action = QAction("", self)
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.play_pause_action.setIcon(icon)
        self.play_pause_action.triggered.connect(self.toggle_play_pause)
        self.toolbar.addAction(self.play_pause_action)
        self.refresh_interval = 30

        self.progress_bar = QSlider(Qt.Orientation.Horizontal)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.progress_bar.valueChanged.connect(self.set_media_position)
        self.toolbar.addWidget(self.progress_bar)

        self.speed_control = QComboBox()
        self.speed_control.addItem("0.5x", 0.5)
        self.speed_control.addItem("1.0x", 1.0)
        self.speed_control.addItem("1.5x", 1.5)
        self.speed_control.addItem("2.0x", 2.0)
        self.speed_control.setCurrentIndex(1)  # 預設為 1.0x
        self.speed_control.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.speed_control.currentIndexChanged.connect(self.set_playback_speed)
        self.toolbar.addWidget(self.speed_control)

        self.play_pause_action.setEnabled(False)
        self.progress_bar.setEnabled(False)
        self.speed_control.setEnabled(False)
        # ^播放控制

        # 退出
        self.quit_action = QAction("&Quit", self)
        self.quit_action.triggered.connect(self.close)

        # 偵測
        self.detect_action = QAction("&Detect", self)
        self.detect_action.triggered.connect(self.image_widget.detectImage)

        # 主選單
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("&File")
        self.edit_menu = self.menu.addMenu("&Edit")
        self.ai_menu = self.menu.addMenu("&Ai")
        self.convert_menu = self.menu.addMenu("&Convert")
        # self.view_menu = self.menu.addMenu("&View")
        # self.help_menu = self.menu.addMenu("&Help")

        self.edit_categories_action = QAction("&Edit Categories", self)
        self.edit_categories_action.triggered.connect(self.edit_categories)

        self.convert_voc_yolo_action = QAction("&VOC to YOLO", self)
        self.convert_voc_yolo_action.triggered.connect(self.convert_voc_to_yolo)

        self.convert_menu.addAction(self.edit_categories_action)
        self.convert_menu.addAction(self.convert_voc_yolo_action)

        self.open_file_by_index_action = QAction("Open File by &Index", self)
        self.open_file_by_index_action.triggered.connect(self.open_file_by_index)

        self.file_menu.addAction(self.open_folder_action)
        self.file_menu.addAction(self.open_file_by_index_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.save_action)
        self.file_menu.addAction(self.auto_save_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.quit_action)

        # 變更標籤
        self.edit_label_action = QAction("&Edit Label", self)
        self.edit_label_action.triggered.connect(self.promptInputLabel)

        self.edit_menu.addAction(self.edit_label_action)

        self.select_model_action = QAction("&Select Model", self)
        self.select_model_action.triggered.connect(self.select_model)

        self.use_default_model_action = QAction("&Use default Model", self)
        self.use_default_model_action.triggered.connect(self.use_default_model)

        self.ai_menu.addAction(self.use_default_model_action)
        self.ai_menu.addAction(self.select_model_action)
        self.ai_menu.addSeparator()
        self.ai_menu.addAction(self.detect_action)
        self.ai_menu.addAction(self.auto_detect_action)

        # 檔案處理器
        self.file_handler = FileHandler()

        # 讀取預設標籤和上次使用的標籤
        self.preset_labels = {}
        self.last_used_label = "object"  # 預設值

        if Settings.data["model_path"]:
            self.load_model(Settings.data["model_path"])
        self.choose_folder(Settings.data["folder_path"], Settings.data["file_index"])

        try:
            with open("config/cfg.yaml", "r", encoding="utf-8") as f:
                config = yaml.load(f)
                yaml_labels = config.get("labels", {})
                if yaml_labels:
                    self.preset_labels = {
                        str(key): value
                        for key, value in yaml_labels.items()
                        if isinstance(key, (int, str))
                    }
                self.last_used_label = config.get("default_label", "object")
                self.auto_save_per_second = config.get("auto_save_per_second", -1)

        except FileNotFoundError:
            QMessageBox.warning(self, "Warning", "config/cfg.yaml not found.")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error parsing config/cfg.yaml: {e}")

    def update_dynamic_config(self):
        with open("config/settings.yaml", "w", encoding="utf-8") as f:
            yaml.dump(Settings.data, f)

    def use_default_model(self):
        self.load_model("yolov8n.pt")

    def select_model(self):
        model_path, _ = QFileDialog.getOpenFileName(
            self, "Open Ultralytics Model File", "", "Model Files (*.pt)"
        )
        if model_path:
            self.load_model(model_path)
            Settings.data["model_path"] = model_path

    def load_model(self, model_path):
        try:
            if self.image_widget.model:
                del self.image_widget.model
            self.image_widget.model = YOLO(model_path)
            self.statusbar.showMessage(f"Model loaded: {model_path}")
            self.image_widget.use_model = True
            self.auto_detect = True
            self.auto_detect_action.setChecked(self.auto_detect)
            self.image_widget.detectImage()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")

    def open_file_by_index(self):
        if not self.file_handler.folder_path:
            QMessageBox.warning(self, "Warning", "No folder opened.")
            return

        if not self.file_handler.image_files:
            QMessageBox.warning(self, "Warning", "No files in the folder.")
            return

        num_files = len(self.file_handler.image_files)
        index, ok = QInputDialog.getInt(
            self,
            "Open File by Index",
            f"Enter file index (1-{num_files}):",
            1,  # 預設值
            1,  # 最小值
            num_files,  # 最大值
        )

        if ok:
            self.file_handler.current_index = index - 1
            self.image_widget.load_image(self.file_handler.current_image_path())
            self.statusbar.showMessage(
                f"Image: {self.file_handler.current_image_path()} "
                f"[{self.file_handler.current_index + 1} / {len(self.file_handler.image_files)}]"
            )

    def open_folder(self):
        """
        用pyqt瀏覽並選定資料夾
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Open Folder")  # PyQt6
        self.choose_folder(folder_path)
        Settings.data["folder_path"] = folder_path
        Settings.data["file_index"] = 0

    def choose_folder(self, folder_path: str, file_index: int = 0):
        """
        開啟資料夾的檔案
        """
        if folder_path:
            self.file_handler.load_folder(folder_path)
            if self.file_handler.image_files:
                self.file_handler.current_index = min(
                    file_index, len(self.file_handler.image_files) - 1
                )
                self.image_widget.load_image(self.file_handler.current_image_path())
                self.statusbar.showMessage(
                    f"Opened folder: {folder_path} "
                    f"[{self.file_handler.current_index + 1} / {len(self.file_handler.image_files)}]"
                )
            else:
                QMessageBox.information(self, "Info", "No files in the folder.")

    def show_image(self, cmd: str):
        if self.is_auto_save():
            self.save_annotations()
        if self.file_handler.show_image(cmd):
            self.image_widget.load_image(self.file_handler.current_image_path())
            self.statusbar.showMessage(
                f"Image: {self.file_handler.current_image_path()}"
                f"[{self.file_handler.current_index + 1} / {len(self.file_handler.image_files)}]"
            )
            Settings.data["file_index"] = self.file_handler.current_index

    def toggle_auto_save(self):
        self.auto_save = not self.auto_save
        self.auto_save_action.setChecked(self.auto_save)
        self.statusbar.showMessage(f"Auto save: {'on' if self.auto_save else 'off'}")

    def toggle_auto_detect(self):
        self.auto_detect = not self.auto_detect
        self.auto_detect_action.setChecked(self.auto_detect)
        self.statusbar.showMessage(
            f"Auto detect: {'on' if self.auto_detect else 'off'}"
        )
        if self.auto_detect:
            self.image_widget.detectImage()

    def is_auto_save(self):
        return self.auto_save

    def is_auto_detect(self):
        return self.auto_detect

    def save_annotations(self):
        """
        儲存標記, 注意就算沒有bbox也是要儲存, 表示有處理過, 也能讓trainer知道這是在訓練背景
        """
        if self.file_handler.current_image_path():
            if self.image_widget.file_type == FileType.VIDEO:
                # 儲存影片當前幀和對應的 XML
                frame = self.image_widget.pixmap.toImage()  # 從 pixmap 取得
                original_filename = os.path.basename(
                    self.file_handler.current_image_path()
                )
                filename_no_ext = os.path.splitext(original_filename)[0]
                frame_number = int(self.image_widget.cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame_filename = f"{filename_no_ext}_frame{frame_number}.jpg"
                frame_path = os.path.join(self.file_handler.folder_path, frame_filename)
                frame.save(frame_path)

                xml_path = getXmlPath(frame_path)
                bboxes = self.image_widget.bboxes
                xml_content = self.file_handler.generate_voc_xml(bboxes, frame_path)
                with open(xml_path, "w", encoding="utf-8") as f:
                    f.write(xml_content)

                if self.is_auto_save():
                    self.statusbar.showMessage(f"Annotations auto saved to {xml_path}")
                else:
                    self.statusbar.showMessage(f"Annotations saved to {xml_path}")

            else:
                # 儲存圖片的 XML
                file_path = getXmlPath(self.file_handler.current_image_path())
                bboxes = self.image_widget.bboxes
                xml_content = self.file_handler.generate_voc_xml(
                    bboxes, self.file_handler.current_image_path()
                )
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(xml_content)
                if self.is_auto_save():
                    self.statusbar.showMessage(f"Annotations auto saved to {file_path}")
                else:
                    self.statusbar.showMessage(f"Annotations saved to {file_path}")

    def toggle_play_pause(self):
        if self.image_widget.file_type != FileType.VIDEO:
            self.image_widget.is_playing = False
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            self.play_pause_action.setIcon(icon)
            return
        self.image_widget.is_playing = not self.image_widget.is_playing
        if self.image_widget.is_playing:
            # 播放前先儲存標籤資訊
            if self.is_auto_save():
                self.save_annotations()

            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            self.play_pause_action.setIcon(icon)
            self.image_widget.timer.start(self.refresh_interval)
        else:
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            self.play_pause_action.setIcon(icon)
            self.image_widget.timer.stop()

    def set_media_position(self, position):
        # 設定影片播放位置 (以毫秒為單位)
        if self.image_widget.cap:
            self.progress_bar.blockSignals(True)
            self.image_widget.cap.set(cv2.CAP_PROP_POS_MSEC, position)
            self.progress_bar.blockSignals(False)

    def set_playback_speed(self, index):
        # 設定播放速度
        if self.image_widget.file_type != FileType.VIDEO:
            return
        speed = self.speed_control.itemData(index)
        self.image_widget.timer.stop()
        if self.image_widget.fps:
            fps = self.image_widget.fps
        else:
            fps = 30
        self.refresh_interval = int(fps / speed)
        if self.image_widget.is_playing:
            self.image_widget.timer.start(self.refresh_interval)  # 重新啟動定時器

    def updateFocusOrLastBbox(self):
        """
        更新focus或最後使用的標籤
        """
        idx = (
            self.image_widget.idx_focus_bbox
            if self.image_widget.idx_focus_bbox >= 0
            and self.image_widget.idx_focus_bbox < len(self.image_widget.bboxes)
            else -1
        )
        if self.image_widget.bboxes:
            self.image_widget.bboxes[idx].label = self.last_used_label
            self.image_widget.update()

    def promptInputLabel(self):
        # 彈出輸入框，讓使用者輸入標籤
        label, ok = QInputDialog.getText(
            self, "Input", "Enter label name:", text=self.last_used_label
        )
        if ok and label.strip():
            self.last_used_label = label.strip()  # 更新上次使用的標籤
            self.updateFocusOrLastBbox()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Right or event.key() == Qt.Key.Key_PageDown:
            self.show_image(ShowImageCmd.NEXT)
        elif event.key() == Qt.Key.Key_Left or event.key() == Qt.Key.Key_PageUp:
            self.show_image(ShowImageCmd.PREV)
        elif event.key() == Qt.Key.Key_Home:
            self.show_image(ShowImageCmd.FIRST)
        elif event.key() == Qt.Key.Key_End:
            self.show_image(ShowImageCmd.LAST)
        elif event.key() == Qt.Key.Key_Q:
            self.close()
        elif event.key() == Qt.Key.Key_A:
            self.toggle_auto_save()
        elif event.key() == Qt.Key.Key_Space:
            self.toggle_play_pause()
        elif event.key() == Qt.Key.Key_L:
            self.promptInputLabel()

        elif event.key() in [
            Qt.Key.Key_1,
            Qt.Key.Key_2,
            Qt.Key.Key_3,
            Qt.Key.Key_4,
            Qt.Key.Key_5,
            Qt.Key.Key_6,
            Qt.Key.Key_7,
            Qt.Key.Key_8,
            Qt.Key.Key_9,
            Qt.Key.Key_0,
        ]:
            key_val = event.key() - Qt.Key.Key_1 + 1
            self.last_used_label = self.preset_labels.get(
                str(key_val), self.last_used_label
            )
            self.updateFocusOrLastBbox()
            self.statusBar().showMessage(f"labels: {self.preset_labels}")

    def closeEvent(self, event):
        """
        關閉前需要儲存最後的標籤資訊, 並更新動態設定檔
        """
        if self.is_auto_save():
            self.save_annotations()
        self.update_dynamic_config()

    def convert_voc_to_yolo(self):
        """
        將 VOC XML 格式的標註檔案轉換為 YOLO TXT 標籤檔
        """
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select folder containing VOC XML files"
        )
        if folder_path:
            output_folder = Path(
                folder_path, YOLO_LABELS_FOLDER
            )  # 在同資料夾下建立 yolo labels 資料夾
            output_folder.mkdir(parents=True, exist_ok=True)
            self.file_handler.convertVocInFolder(folder_path, output_folder)
            self.statusbar.showMessage(
                f"VOC to YOLO conversion completed in folder: {output_folder}"
            )

    def edit_categories(self):
        dialog = CategorySettingsDialog(self)
        if dialog.exec():
            self.statusbar.showMessage("Categories設定已儲存")
            self.update_dynamic_config()


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
        self.folder_path = None
        self.image_files = []
        self.current_index = 0

    def load_folder(self, folder_path):
        self.folder_path = folder_path
        self.image_files = []
        self.current_index = 0
        for file in os.listdir(folder_path):
            if file.lower().endswith(ALL_EXTS):
                self.image_files.append(file)
        self.image_files.sort()  # 排序

    def current_image_path(self):
        if not self.image_files:
            return None
        return os.path.join(self.folder_path, self.image_files[self.current_index])

    def show_image(self, cmd: str):
        """
        show [next, prev, first, last] image
        Args:
            cmd: one of "next", "prev", "first", "last"

        Returns:
            True if file is changed
        """
        if cmd == ShowImageCmd.NEXT:
            if self.current_index < len(self.image_files) - 1:
                self.current_index += 1
                return True
        elif cmd == ShowImageCmd.PREV:
            if self.current_index > 0:
                self.current_index -= 1
                return True
        elif cmd == ShowImageCmd.FIRST:
            if self.current_index != 0:
                self.current_index = 0
                return True
        elif cmd == ShowImageCmd.LAST:
            if self.current_index != len(self.image_files) - 1:
                self.current_index = len(self.image_files) - 1
                return True
        return False

    def generate_voc_xml(self, bboxes, image_path):
        """
        基於現有的bbox 產生符合voc格式的xml檔案
        """
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

    def convertVocInFolder(self, folder_path, output_folder: Optional[Path] = None):
        """
        將指定資料夾下的所有 VOC XML 檔案轉換為 YOLO 格式
        """
        if output_folder is None:
            output_folder = folder_path  # 預設輸出到同一個資料夾

        for xml_file in Path(folder_path).glob("*.xml"):
            self.convert_voc_xml_to_yolo_txt(xml_file, output_folder)

    def convert_voc_xml_to_yolo_txt(self, xml_path, output_folder):
        """
        轉換單個 VOC XML 檔案到 YOLO 格式
        """

        # root = ET.parse(Path(xml_path).as_posix())
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_element = root.find("size")
        if size_element is None:
            print(f"Warning: No size element found in {xml_path}, skipping")
            return
        width = int(size_element.find("width").text)
        height = int(size_element.find("height").text)
        yolo_lines = []

        for object_element in root.findall("object"):
            label_name = object_element.find("name").text
            if label_name not in Settings.data["categories"]:
                print(f"Warning: Label '{label_name}' not in categories,")
                continue  # Skip to the next object if label is not in categories

            category_id = Settings.data["categories"].get(label_name)
            if category_id is None or not isinstance(category_id, int):
                print(
                    f"Warning: Category ID not found for label '{label_name}', skipping"
                )
                continue

            bndbox_element = object_element.find("bndbox")
            xmin = int(bndbox_element.find("xmin").text)
            ymin = int(bndbox_element.find("ymin").text)
            xmax = int(bndbox_element.find("xmax").text)
            ymax = int(bndbox_element.find("ymax").text)

            # Convert VOC bbox to YOLO normalized coordinates
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            yolo_line = f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            yolo_lines.append(yolo_line)

        # Save YOLO txt file
        output_file = output_folder / Path(xml_path).with_suffix(".txt").name
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        print(f"Converted {xml_path} to {output_file}")


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
