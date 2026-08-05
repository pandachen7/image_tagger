# 通用小工具：unicode 路徑影像讀寫、同名配對、xml/mask 路徑推導、路徑同一性判斷
# 更新日期: 2026-08-06
import os
from pathlib import Path

import cv2
import numpy as np

from src.utils.logger import getUniqueLogger

log = getUniqueLogger(__file__)


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    # cv2.imread 在 Windows 用 ANSI code page 開檔，遇中文/日文路徑會回 None。
    # 改走 np.fromfile + cv2.imdecode 由 Python 自己讀 bytes，路徑就吃得下任何 unicode。
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, img: np.ndarray, ext: str = ".jpg", params=None) -> bool:
    """以 unicode 安全的方式寫出影像 (cv2.imwrite 在 Windows 遇中文路徑會失敗)。

    走 cv2.imencode + ndarray.tofile 由 Python 自己寫 bytes，路徑就吃得下任何 unicode。

    Args:
        path: 輸出檔案路徑
        img: BGR 影像 (numpy)
        ext: 編碼副檔名 (依此決定格式)，預設 ".jpg"
        params: 傳給 cv2.imencode 的參數 list，例如 [cv2.IMWRITE_JPEG_QUALITY, 95]

    Returns:
        bool: 是否寫出成功
    """
    try:
        ok, buf = cv2.imencode(ext, img, params or [])
        if not ok:
            log.error(f"imencode 失敗: {path}")
            return False
        buf.tofile(str(path))
        return True
    except Exception as e:
        log.error(f"imwrite_unicode 失敗 ({path}): {e}")
        return False


def find_pairs(file_path):
    """
    找同檔名配對, e.g. abc.txt一定要有個abc.jpg
    """
    txt_names = [
        file
        for file in os.listdir(file_path)
        if file.lower().endswith(("jpg", "png", "jpeg"))
    ]
    total_num_of_files = len(txt_names)
    print("Found " + str(total_num_of_files) + ' txt in folder "' + file_path + '"')

    for file_name in txt_names:
        pure_name = os.path.splitext(file_name)[0]
        path_img = os.path.join(file_path, pure_name + ".txt")
        if not Path(path_img).is_file():
            print(f"path_img {path_img} not exists")


def getXmlPath(image_path) -> Path:
    path_tmp = Path(image_path)
    return path_tmp.parent / f"{path_tmp.stem}.xml"


def getMaskPath(image_path) -> Path:
    path_tmp = Path(image_path)
    return path_tmp.parent / f"{path_tmp.stem}_mask.png"


def is_same_path(path_a, path_b) -> bool:
    """判斷兩個路徑是否指向同一個檔案或資料夾。

    兩者都存在時走 os.path.samefile (可正確處理 symlink / junction / 大小寫 /
    Windows 8.3 短檔名); 只要有一邊尚未建立就退回 resolve + normcase 的字串比較。

    Args:
        path_a: 路徑 A (str 或 Path)
        path_b: 路徑 B (str 或 Path)

    Returns:
        bool: 兩者是否指向同一個目標, 無法判斷時回 False
    """
    try:
        pa, pb = Path(path_a), Path(path_b)
        if pa.exists() and pb.exists():
            return os.path.samefile(pa, pb)
        # 尚未建立的路徑無法問檔案系統, 只能正規化後比字串 (Windows 大小寫不敏感)
        return os.path.normcase(str(pa.resolve())) == os.path.normcase(
            str(pb.resolve())
        )
    except Exception as e:
        log.error(f"比對路徑失敗 ({path_a} vs {path_b}): {e}")
        return False
