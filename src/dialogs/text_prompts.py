# Text Prompts 對話框：編輯 SAM3 的文字提示
# 更新日期: 2026-04-12
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.utils.dynamic_settings import settings


class TextPromptsDialog(QDialog):
    """編輯 SAM3 Text Prompts 的對話框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Text Prompts")
        self.setMinimumWidth(350)

        self.prompts = list(settings.class_names.text_prompts or [])

        hint = QLabel("提供給 SAM3 / Segmentation model 的文字提示，用於引導分割目標")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)

        self.list_widget = QTableWidget()
        self.list_widget.setColumnCount(1)
        self.list_widget.setHorizontalHeaderLabels(["Prompt"])
        self.list_widget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_prompt)
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_prompt)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_prompts)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)

        content_layout = QHBoxLayout()
        content_layout.addWidget(self.list_widget)
        content_layout.addLayout(button_layout)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(hint)
        main_layout.addLayout(content_layout)

        self.load_prompts()

    def load_prompts(self):
        self.list_widget.setRowCount(0)
        for prompt in self.prompts:
            row = self.list_widget.rowCount()
            self.list_widget.insertRow(row)
            self.list_widget.setItem(row, 0, QTableWidgetItem(prompt))

    def add_prompt(self):
        row = self.list_widget.rowCount()
        self.list_widget.insertRow(row)
        self.list_widget.setItem(row, 0, QTableWidgetItem(""))
        self.list_widget.editItem(self.list_widget.item(row, 0))

    def delete_prompt(self):
        selected_row = self.list_widget.currentRow()
        if selected_row >= 0:
            self.list_widget.removeRow(selected_row)

    def save_prompts(self):
        prompts = []
        for row in range(self.list_widget.rowCount()):
            item = self.list_widget.item(row, 0)
            if item and item.text().strip():
                prompts.append(item.text().strip())
        settings.class_names.text_prompts = prompts
        self.accept()
