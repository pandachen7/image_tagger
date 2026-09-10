# 檔案讀寫、清單維護、VOC XML 產生與 VOC→YOLO 格式轉換 (含 train/val 分組切分)
# 更新日期: 2026-09-10
import math
import os
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core import AppState
from src.utils.const import ALL_EXTS
from src.utils.dynamic_settings import settings
from src.utils.func import imread_unicode
from src.utils.logger import getUniqueLogger
from src.utils.model import Bbox, Polygon, ShowImageCmd

log = getUniqueLogger(__file__)

# 同一張原圖切出來的 cropped (_crop0, _crop1...) 與同一段影片的幀 (_frame123) 內容
# 高度相似, 分到 train / val 兩邊會讓驗證分數虛高。切分時用這個 pattern 還原出共同
# 來源當分組依據。cropped 影片的檔名是 <stem>_frame123_crop0, 兩段都要吃掉。
_SOURCE_SUFFIX_RE = re.compile(r"(?:_frame\d+)?(?:_crop\d+)?$")

# seg 輸出時, 同一物件的 bndbox 與 polygon 都會被寫進 VOC XML (label mode = all),
# 兩者的外接框重疊到這個 IoU 以上就視為同一物件, 只保留 polygon。
_SEG_DEDUP_IOU = 0.5


@dataclass
class ConvertStats:
    """VOC → YOLO 轉換過程的統計，供轉換後的摘要與警告使用"""

    # 未對應到 categories 的 (圖檔名, class_name)
    not_matched: list[tuple[str, str]] = field(default_factory=list)
    # 原本有框、但所有框的 class 都對不上 → 刻意不產生 txt 的 xml 檔名
    unmatched_only: list[str] = field(default_factory=list)
    # 原本就沒有框的背景樣本數 (產生空 txt, 這是刻意的)
    background: int = 0
    # seg 模式下, 與 polygon 重複而被丟掉的 bndbox 筆數
    seg_dedup: int = 0


def source_group_key(stem: str) -> str:
    """取得檔名所屬的「來源」，同一來源的檔案必須落在同一個 split

    Args:
        stem: 不含副檔名的檔名

    Returns:
        str: 去掉 _frameN / _cropN 後的來源名稱
    """
    return _SOURCE_SUFFIX_RE.sub("", stem)


def split_train_val(
    files: list[Path], train_ratio: float
) -> tuple[list[Path], list[Path], bool]:
    """依比例切成 train / val，並保證同一來源的檔案不會被拆到兩邊

    val 一定至少有一個檔案 (檔案數 >= 2 時): val 是空的或與 train 重疊時, early
    stopping 與 best.pt 的挑選都會失去意義 (fitness 反映的是訓練集表現)。

    Args:
        files: 已配對好標籤的圖片清單
        train_ratio: train 佔的比例 (0 < ratio < 1)

    Returns:
        train_files, val_files, grouped:
            grouped 為 False 表示來源只有一組 (例如只標了一段影片), 無法分組切分,
            已退回逐檔隨機切; 此時 val 與 train 高度相似, 分數會偏樂觀。
    """
    if len(files) < 2:
        return list(files), [], False

    groups: dict[str, list[Path]] = {}
    for f in files:
        groups.setdefault(source_group_key(f.stem), []).append(f)

    # 來源只有一組時無法分組切 (整組都進 train 的話 val 就空了), 退回逐檔切
    if len(groups) < 2:
        shuffled = list(files)
        random.shuffle(shuffled)
        idx = min(max(1, int(len(shuffled) * train_ratio)), len(shuffled) - 1)
        return shuffled[:idx], shuffled[idx:], False

    keys = list(groups)
    random.shuffle(keys)
    target = len(files) * train_ratio
    train_files: list[Path] = []
    val_files: list[Path] = []
    for key in keys:
        if len(train_files) < target:
            train_files.extend(groups[key])
        else:
            val_files.extend(groups[key])

    # 比例算下來可能整份都落在 train (例如 ratio 0.95 但只有兩組), 至少挪一組給 val
    if not val_files:
        smallest = min(keys, key=lambda k: len(groups[k]))
        val_files = groups[smallest]
        train_files = [f for f in train_files if f not in set(val_files)]
    return train_files, val_files, True


def _aabb_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """兩個 (xmin, ymin, xmax, ymax) 的 IoU

    Args:
        a: 矩形 A
        b: 矩形 B

    Returns:
        float: IoU, 沒有交集時為 0.0
    """
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


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

    def current_image_path(self) -> str:
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
            if not self.image_files:
                # 全部刪光了, 否則下面會把索引設成 -1 而讀到最後一個不存在的項目
                self.current_index = 0
                return False
            if self.current_index > len(self.image_files) - 1:
                self.current_index = len(self.image_files) - 1
            return True
        return False

    def drop_current(self) -> bool:
        """把目前檔案從清單移除 (實體檔案已刪除後才呼叫)。

        索引留在原位, 因此接著會顯示原本的下一張; 刪到最後一張時退回新的最後一張。

        Returns:
            bool: 移除後清單是否還有檔案
        """
        if not self.image_files:
            return False
        try:
            self.image_files.pop(self.current_index)
        except IndexError as e:
            log.e(f"移除清單項目失敗 (index={self.current_index}): {e}")
            return bool(self.image_files)
        # 索引修正沿用 SAME_INDEX 那一套: 停在原位以顯示下一張, 超出範圍才退回最後一張
        return self.show_image(ShowImageCmd.SAME_INDEX)

    def generate_voc_xml(
        self, bboxes: list[Bbox], image_path, polygons: list[Polygon] = None
    ):
        """
        基於現有的bbox與polygon產生符合voc格式的xml檔案
        """
        if polygons is None:
            polygons = []

        image_filename = os.path.basename(image_path)
        folder_name = os.path.basename(os.path.dirname(image_path))

        xml_str = "<annotation>\n"
        xml_str += f"    <folder>{folder_name}</folder>\n"
        xml_str += f"    <filename>{image_filename}</filename>\n"

        # 讀取圖片大小（imread_unicode 支援中文路徑）
        img = imread_unicode(image_path)
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
            xml_str += f"            <angle>{int(bbox.angle)}</angle>\n"
            xml_str += "        </bndbox>\n"
            xml_str += "    </object>\n"

        for polygon in polygons:
            xml_str += "    <object>\n"
            xml_str += f"        <name>{polygon.label}</name>\n"
            xml_str += "        <polygon>\n"
            xml_str += f"            <confidence>{polygon.confidence}</confidence>\n"
            for px, py in polygon.points:
                xml_str += (
                    f"            <point><x>{px:.1f}</x><y>{py:.1f}</y></point>\n"
                )
            xml_str += "        </polygon>\n"
            xml_str += "    </object>\n"

        xml_str += "</annotation>\n"
        return xml_str

    def convertVocInFolder(
        self,
        folder_path,
        output_folder: Optional[Path] = None,
        app_state: AppState = None,
        progress_callback: Optional[callable] = None,
    ) -> ConvertStats:
        """
        將指定資料夾下的所有 VOC XML 檔案轉換為 YOLO 格式
        Args:
            progress_callback: 回呼函式 (current, total) -> None，用於更新進度條
        Returns:
            ConvertStats: 未對應的 class、被跳過的 xml、背景樣本數等統計
        """
        if output_folder is None:
            output_folder = folder_path  # 預設輸出到同一個資料夾

        output_mode = app_state.yolo_output_mode if app_state else "bbox"
        xml_files = list(Path(folder_path).glob("*.xml"))
        total = len(xml_files)
        stats = ConvertStats()

        for i, xml_file in enumerate(xml_files):
            if output_mode == "seg":
                self.convert_voc_xml_to_yolo_seg_txt(
                    xml_file, output_folder, app_state, stats
                )
            else:
                self.convert_voc_xml_to_yolo_txt(
                    xml_file, output_folder, app_state, stats
                )
            if progress_callback:
                progress_callback(i + 1, total)

        log.i(
            f"converted {total} xml files (mode={output_mode}), "
            f"背景樣本={stats.background}, 全數對不上而跳過={len(stats.unmatched_only)}, "
            f"seg 去重={stats.seg_dedup}"
        )
        return stats

    def count_label_classes(self, labels_dir: Path) -> dict[int, int]:
        """統計 YOLO txt 內每個 class id 的標註筆數

        Args:
            labels_dir: 放 YOLO txt 的資料夾

        Returns:
            dict: {class_id: 標註筆數}; 空檔 (背景圖) 不計入
        """
        counts: dict[int, int] = {}
        for txt in labels_dir.glob("*.txt"):
            try:
                for line in txt.read_text(encoding="utf-8").splitlines():
                    cols = line.split()
                    if not cols:
                        continue
                    cid = int(float(cols[0]))
                    counts[cid] = counts.get(cid, 0) + 1
            except Exception as e:
                log.error(f"讀取標籤失敗 ({txt}): {e}")
        return counts

    def remap_label_classes(self, labels_dir: Path, id_map: dict[int, int]) -> int:
        """依 id_map 就地改寫 YOLO txt 的 class id

        用於「只輸出實際出現的類別」: dataset.yaml 的 nc 必須等於 names 的數量且
        class id 要從 0 連號, 而 class mapping 裡沒用到的類別除了讓模型多學幾個空
        類別, 還會讓 optimizer=auto 依 nc 算出更小的學習率。

        Args:
            labels_dir: 放 YOLO txt 的資料夾
            id_map: {舊 class_id: 新 class_id}

        Returns:
            int: 實際被改寫的檔案數
        """
        changed = 0
        for txt in labels_dir.glob("*.txt"):
            try:
                new_lines = []
                dirty = False
                for line in txt.read_text(encoding="utf-8").splitlines():
                    cols = line.split()
                    if not cols:
                        continue
                    old_id = int(float(cols[0]))
                    new_id = id_map.get(old_id, old_id)
                    if new_id != old_id:
                        cols[0] = str(new_id)
                        dirty = True
                    new_lines.append(" ".join(cols))
                if dirty:
                    txt.write_text("\n".join(new_lines), encoding="utf-8")
                    changed += 1
            except Exception as e:
                log.error(f"改寫標籤 class id 失敗 ({txt}): {e}")
        return changed

    def _write_yolo_txt(
        self,
        xml_path,
        output_folder: Path,
        yolo_lines: list[str],
        object_count: int,
        stats: ConvertStats,
    ) -> bool:
        """寫出 YOLO txt；原本有框卻一行都轉不出來時刻意不寫

        ultralytics 對「缺少 txt」與「空 txt」都當成背景圖 (verify_image_label 的
        nm / ne), 兩者都會被拿去當負樣本訓練。所以只有原本就沒框的 XML 才可以留下
        空檔 (那是刻意存的背景樣本); 有框但 class 全對不上的話, 一旦寫出空檔, 圖裡
        的物件就會被當成「背景」教給模型, 比整張圖不收進 dataset 還糟。

        Args:
            xml_path: 來源 XML 路徑
            output_folder: txt 輸出資料夾
            yolo_lines: 轉出來的標籤行
            object_count: XML 內原本的 object 數量
            stats: 轉換統計 (就地更新)

        Returns:
            bool: 是否寫出了 txt
        """
        if not yolo_lines and object_count > 0:
            log.w(f"{Path(xml_path).name}: {object_count} 個框全數對不上 categories, 不產生 txt")
            stats.unmatched_only.append(Path(xml_path).name)
            return False
        if not yolo_lines:
            stats.background += 1

        output_file = output_folder / Path(xml_path).with_suffix(".txt").name
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
        except Exception as e:
            log.error(f"寫出 YOLO 標籤失敗 ({output_file}): {e}")
            return False
        return True

    def convert_voc_xml_to_yolo_txt(
        self, xml_path, output_folder, app_state=None, stats: ConvertStats = None
    ) -> ConvertStats:
        """
        轉換單個 VOC XML 檔案到 YOLO 格式
        支援 OBB (Oriented Bounding Box) 格式，輸出四個角點座標
        Args:
            stats: 轉換統計 (就地更新); 未提供時自行建立
        Returns:
            ConvertStats: 更新後的統計
        """
        if stats is None:
            stats = ConvertStats()

        # root = ET.parse(Path(xml_path).as_posix())
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_element = root.find("size")
        if size_element is None:
            log.w(f"Warning: No size element found in {xml_path}, skipping")
            return stats
        img_width = int(size_element.find("width").text)
        img_height = int(size_element.find("height").text)
        yolo_lines = []

        # 取得對應的圖檔名
        filename_element = root.find("filename")
        image_filename = filename_element.text if filename_element is not None else Path(xml_path).stem

        objects = root.findall("object")
        for object_element in objects:
            label_name = object_element.find("name").text
            if label_name not in settings.class_names.categories:
                log.w(f"Warning: Label '{label_name}' not in categories")
                stats.not_matched.append((image_filename, label_name))
                continue  # Skip to the next object if label is not in categories

            category_id = settings.class_names.categories.get(label_name)
            if category_id is None or not isinstance(category_id, int):
                log.w(
                    f"Warning: Category ID not found for label '{label_name}', skipping"
                )
                stats.not_matched.append((image_filename, label_name))
                continue

            bndbox_element = object_element.find("bndbox")
            if bndbox_element is None:
                log.w(f"Warning: No bndbox element for '{label_name}' in {xml_path}, skipping")
                continue
            xmin = int(bndbox_element.find("xmin").text)
            ymin = int(bndbox_element.find("ymin").text)
            xmax = int(bndbox_element.find("xmax").text)
            ymax = int(bndbox_element.find("ymax").text)

            # 讀取角度（如果存在）
            angle_element = bndbox_element.find("angle")
            angle = float(angle_element.text) if angle_element is not None else 0.0

            # 判斷是否使用 OBB 格式
            output_mode = app_state.yolo_output_mode if app_state else "bbox"
            if output_mode == "obb" and angle != 0:
                # OBB 格式：輸出四個角點的歸一化座標
                # 計算 bbox 的中心點和寬高
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2

                # 計算四個角點（未旋轉時）相對於中心的位置
                corners = [
                    (-bbox_width / 2, -bbox_height / 2),  # top_left
                    (bbox_width / 2, -bbox_height / 2),  # top_right
                    (bbox_width / 2, bbox_height / 2),  # bottom_right
                    (-bbox_width / 2, bbox_height / 2),  # bottom_left
                ]

                # 旋轉角點
                angle_rad = math.radians(angle)
                rotated_corners = []
                for dx, dy in corners:
                    # 旋轉
                    rotated_x = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
                    rotated_y = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
                    # 加上中心點偏移
                    abs_x = center_x + rotated_x
                    abs_y = center_y + rotated_y
                    # 歸一化
                    norm_x = abs_x / img_width
                    norm_y = abs_y / img_height
                    rotated_corners.append((norm_x, norm_y))

                # 格式：class_id x1 y1 x2 y2 x3 y3 x4 y4
                yolo_line = f"{category_id}"
                for x, y in rotated_corners:
                    yolo_line += f" {x:.6f} {y:.6f}"
                yolo_lines.append(yolo_line)
            else:
                # 標準 YOLO 格式：中心點 + 寬高
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                w = (xmax - xmin) / img_width
                h = (ymax - ymin) / img_height

                yolo_line = (
                    f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                )
                yolo_lines.append(yolo_line)

        self._write_yolo_txt(xml_path, output_folder, yolo_lines, len(objects), stats)
        return stats

    @staticmethod
    def _polygon_aabbs_by_label(root) -> dict[str, list[tuple[float, float, float, float]]]:
        """收集 XML 內每個 label 的 polygon 外接框，供 bndbox 去重比對

        Args:
            root: VOC XML 的根節點

        Returns:
            dict: {label_name: [(xmin, ymin, xmax, ymax), ...]}
        """
        result: dict[str, list[tuple[float, float, float, float]]] = {}
        for obj in root.findall("object"):
            polygon_element = obj.find("polygon")
            if polygon_element is None:
                continue
            name_element = obj.find("name")
            if name_element is None:
                continue
            xs, ys = [], []
            for pt in polygon_element.findall("point"):
                xs.append(float(pt.find("x").text))
                ys.append(float(pt.find("y").text))
            if len(xs) >= 3:
                result.setdefault(name_element.text, []).append(
                    (min(xs), min(ys), max(xs), max(ys))
                )
        return result

    def convert_voc_xml_to_yolo_seg_txt(
        self, xml_path, output_folder, app_state=None, stats: ConvertStats = None
    ) -> ConvertStats:
        """
        轉換單個 VOC XML 檔案到 YOLO Segmentation 格式
        格式: class_id x1 y1 x2 y2 ... xN yN (normalized 0-1)
        Args:
            stats: 轉換統計 (就地更新); 未提供時自行建立
        Returns:
            ConvertStats: 更新後的統計
        """
        if stats is None:
            stats = ConvertStats()

        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_element = root.find("size")
        if size_element is None:
            log.w(f"Warning: No size element found in {xml_path}, skipping")
            return stats
        img_width = int(size_element.find("width").text)
        img_height = int(size_element.find("height").text)
        yolo_lines = []

        # 取得對應的圖檔名
        filename_element = root.find("filename")
        image_filename = filename_element.text if filename_element is not None else Path(xml_path).stem

        # label mode = all 時, 同一個偵測結果會同時存成 bndbox 與 polygon 兩個 object。
        # seg 輸出會把 bndbox 補成矩形 polygon, 於是同一物件出現兩筆 GT (座標不同,
        # ultralytics 的重複列過濾抓不到), 等於每個物件被標兩次。這裡先收集 polygon
        # 的外接框, 後面遇到同 label 且高度重疊的 bndbox 就跳過。
        polygon_aabbs = self._polygon_aabbs_by_label(root)

        objects = root.findall("object")
        for object_element in objects:
            label_name = object_element.find("name").text
            if label_name not in settings.class_names.categories:
                log.w(f"Warning: Label '{label_name}' not in categories")
                stats.not_matched.append((image_filename, label_name))
                continue

            category_id = settings.class_names.categories.get(label_name)
            if category_id is None or not isinstance(category_id, int):
                log.w(
                    f"Warning: Category ID not found for label '{label_name}', skipping"
                )
                stats.not_matched.append((image_filename, label_name))
                continue

            polygon_element = object_element.find("polygon")
            if polygon_element is not None:
                # Use polygon points
                points = []
                for pt in polygon_element.findall("point"):
                    px = float(pt.find("x").text)
                    py = float(pt.find("y").text)
                    points.append((px / img_width, py / img_height))

                # 少於 3 點構不成多邊形, 寫出去會變成畸形 segment
                if len(points) >= 3:
                    yolo_line = f"{category_id}"
                    for nx, ny in points:
                        yolo_line += f" {nx:.6f} {ny:.6f}"
                    yolo_lines.append(yolo_line)
            else:
                # Fallback: use bndbox as a 4-point polygon
                bndbox_element = object_element.find("bndbox")
                if bndbox_element is None:
                    continue
                xmin = int(bndbox_element.find("xmin").text)
                ymin = int(bndbox_element.find("ymin").text)
                xmax = int(bndbox_element.find("xmax").text)
                ymax = int(bndbox_element.find("ymax").text)

                # 同 label 已有覆蓋同一塊區域的 polygon → 這個 bndbox 是重複的
                if any(
                    _aabb_iou((xmin, ymin, xmax, ymax), aabb) >= _SEG_DEDUP_IOU
                    for aabb in polygon_aabbs.get(label_name, [])
                ):
                    stats.seg_dedup += 1
                    continue

                # Normalize
                nx1 = xmin / img_width
                ny1 = ymin / img_height
                nx2 = xmax / img_width
                ny2 = ymax / img_height

                yolo_line = (
                    f"{category_id}"
                    f" {nx1:.6f} {ny1:.6f}"
                    f" {nx2:.6f} {ny1:.6f}"
                    f" {nx2:.6f} {ny2:.6f}"
                    f" {nx1:.6f} {ny2:.6f}"
                )
                yolo_lines.append(yolo_line)

        self._write_yolo_txt(xml_path, output_folder, yolo_lines, len(objects), stats)
        return stats


file_h = FileHandler()
