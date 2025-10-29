from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.core import AppState
from src.utils.dynamic_settings import settings


class CategorySettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Categories, 這是給VOC轉yolo用的")
        self.categories = settings.categories

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
        settings.categories = categories
        self.accept()


class ConvertSettingsDialog(QDialog):
    """轉換設定對話框"""

    def __init__(self, parent=None, app_state: AppState = None):
        super().__init__(parent)
        self.app_state = app_state
        self.setWindowTitle("轉換設定 (Convert Settings)")
        self.setMinimumWidth(400)

        # 主佈局
        main_layout = QVBoxLayout(self)

        # 格式選擇群組
        format_group = QGroupBox("輸出格式 (Output Format)")
        format_layout = QFormLayout()

        # 格式選擇下拉選單
        self.format_combo = QComboBox()
        self.format_combo.addItem("YOLO", "yolo")  # 顯示文字, 資料值
        # 將來可以在這裡添加更多格式
        # self.format_combo.addItem("COCO", "coco")
        # self.format_combo.addItem("Pascal VOC", "voc")

        format_layout.addRow(QLabel("格式 (Format):"), self.format_combo)
        format_group.setLayout(format_layout)
        main_layout.addWidget(format_group)

        # YOLO 選項群組
        yolo_group = QGroupBox("YOLO 選項 (YOLO Options)")
        yolo_layout = QVBoxLayout()

        # OBB 格式選項
        self.obb_checkbox = QCheckBox(
            "使用 OBB 格式 (Use OBB format for rotated bounding boxes)"
        )
        self.obb_checkbox.setToolTip(
            "啟用後，將輸出旋轉邊界框的四個角點座標\n"
            "格式: class_id x1 y1 x2 y2 x3 y3 x4 y4 (左上右上右下左下)\n"
            "適用於 YOLOv8 OBB 訓練"
        )

        yolo_layout.addWidget(self.obb_checkbox)
        yolo_group.setLayout(yolo_layout)
        main_layout.addWidget(yolo_group)

        # 按鈕
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("儲存 (Save)")
        self.save_button.clicked.connect(self.save_settings)
        self.cancel_button = QPushButton("取消 (Cancel)")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        # 載入當前設定
        self.load_settings()

    def load_settings(self):
        """從 app_state 載入當前設定"""
        if self.app_state:
            # 設定格式選擇
            index = self.format_combo.findData(self.app_state.convert_format)
            if index >= 0:
                self.format_combo.setCurrentIndex(index)

            # 設定 OBB 選項
            self.obb_checkbox.setChecked(self.app_state.yolo_obb_format)

    def save_settings(self):
        """儲存設定到 app_state"""
        if self.app_state:
            self.app_state.convert_format = self.format_combo.currentData()
            self.app_state.yolo_obb_format = self.obb_checkbox.isChecked()
        self.accept()
