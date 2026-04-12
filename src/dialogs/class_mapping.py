# Class Mapping 對話框：編輯 class_name → class_id 的對應關係
# 更新日期: 2026-04-12
from PyQt6.QtWidgets import (
    QDialog,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.utils.dynamic_settings import settings


class ClassMappingDialog(QDialog):
    """編輯 class_name → class_id 的對應關係"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Class Mapping")
        self.categories = settings.class_names.categories

        hint = QLabel("設定 VOC → YOLO 轉換時 class name 與 class id 的對應關係")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["class_name", "class_id"])
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
        main_layout.addWidget(hint)
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

    def _next_class_id(self) -> int:
        """找出目前表格中最大的 class_id + 1"""
        max_id = -1
        for row in range(self.table_widget.rowCount()):
            item = self.table_widget.item(row, 1)
            if item and item.text().isdigit():
                max_id = max(max_id, int(item.text()))
        return max_id + 1

    def add_category(self):
        self.add_row("", str(self._next_class_id()))

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
        settings.class_names.categories = categories
        self.accept()
