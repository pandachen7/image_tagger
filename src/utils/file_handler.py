import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2

from src.utils.const import ALL_EXTS
from src.utils.dynamic_settings import settings
from src.utils.loglo import getUniqueLogger
from src.utils.model import ShowImageCmd

log = getUniqueLogger(__file__)


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
        elif cmd == ShowImageCmd.SAME_INDEX:
            # 用於刪除時
            if self.current_index > len(self.image_files) - 1:
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

        ct = 0
        for xml_file in Path(folder_path).glob("*.xml"):
            self.convert_voc_xml_to_yolo_txt(xml_file, output_folder)
            ct += 1
        log.i(f"converted {ct} xml files")

    def convert_voc_xml_to_yolo_txt(self, xml_path, output_folder):
        """
        轉換單個 VOC XML 檔案到 YOLO 格式
        """

        # root = ET.parse(Path(xml_path).as_posix())
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_element = root.find("size")
        if size_element is None:
            log.w(f"Warning: No size element found in {xml_path}, skipping")
            return
        width = int(size_element.find("width").text)
        height = int(size_element.find("height").text)
        yolo_lines = []

        for object_element in root.findall("object"):
            label_name = object_element.find("name").text
            if label_name not in settings.categories:
                log.w(f"Warning: Label '{label_name}' not in categories,")
                continue  # Skip to the next object if label is not in categories

            category_id = settings.categories.get(label_name)
            if category_id is None or not isinstance(category_id, int):
                log.w(
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

        # log.d(f"Converted {xml_path} to {output_file}")


file_h = FileHandler()
