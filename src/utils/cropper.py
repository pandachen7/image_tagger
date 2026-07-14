# 依畫面標註 (bbox / polygon) 計算 cropped 裁切區域，供動態區/ROI 過濾後的 YOLO dataset 使用
# 更新日期: 2026-07-14
from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.utils.logger import getUniqueLogger
from src.utils.model import Bbox, Polygon

log = getUniqueLogger(__file__)

# cropped 尺寸模式
CROP_MODE_PADDING = "padding"  # 每邊固定外擴 pixel
CROP_MODE_FIXED = "fixed"      # 裁切區至少 fixed_size x fixed_size (置中於框)


@dataclass
class CropTask:
    """單一 cropped 圖片的裁切任務：裁切區域 (原圖座標) + 已平移到裁切區座標系的標註"""

    x0: int
    y0: int
    x1: int
    y1: int
    bboxes: list[Bbox] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)

    @property
    def width(self) -> int:
        """裁切區寬度"""
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        """裁切區高度"""
        return self.y1 - self.y0


def _bbox_aabb(bbox: Bbox) -> tuple[int, int, int, int]:
    """取得 bbox 的軸對齊外接矩形；旋轉框則取旋轉後四角點的外接框

    Args:
        bbox: 目標 bbox

    Returns:
        (x0, y0, x1, y1) 軸對齊外接矩形
    """
    if bbox.angle % 360 == 0:
        return (bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)

    cx = bbox.x + bbox.width / 2
    cy = bbox.y + bbox.height / 2
    rad = math.radians(bbox.angle)
    corners = [
        (-bbox.width / 2, -bbox.height / 2),
        (bbox.width / 2, -bbox.height / 2),
        (bbox.width / 2, bbox.height / 2),
        (-bbox.width / 2, bbox.height / 2),
    ]
    xs = [cx + dx * math.cos(rad) - dy * math.sin(rad) for dx, dy in corners]
    ys = [cy + dx * math.sin(rad) + dy * math.cos(rad) for dx, dy in corners]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _polygon_aabb(polygon: Polygon) -> tuple[int, int, int, int]:
    """取得 polygon 的軸對齊外接矩形

    Args:
        polygon: 目標 polygon

    Returns:
        (x0, y0, x1, y1) 軸對齊外接矩形
    """
    xs = [p[0] for p in polygon.points]
    ys = [p[1] for p in polygon.points]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _fit_region(
    x0: float, y0: float, x1: float, y1: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """將目標區域平移/裁切到影像範圍內；某邊超出時往對邊借像素以維持尺寸

    Args:
        x0, y0, x1, y1: 目標區域 (可能超出影像邊界)
        img_w, img_h: 影像寬高

    Returns:
        (x0, y0, x1, y1) 落在影像內的區域
    """
    w = x1 - x0
    h = y1 - y0

    # 水平方向
    if w >= img_w:
        x0, x1 = 0, img_w
    else:
        if x0 < 0:
            # 左邊沒像素了 → 整個往右移 (往對邊借)
            x1 -= x0
            x0 = 0
        if x1 > img_w:
            # 右邊沒像素了 → 整個往左移 (往對邊借)
            shift = x1 - img_w
            x0 -= shift
            x1 = img_w
            if x0 < 0:
                x0 = 0

    # 垂直方向
    if h >= img_h:
        y0, y1 = 0, img_h
    else:
        if y0 < 0:
            y1 -= y0
            y0 = 0
        if y1 > img_h:
            shift = y1 - img_h
            y0 -= shift
            y1 = img_h
            if y0 < 0:
                y0 = 0

    return (int(x0), int(y0), int(x1), int(y1))


def _expand(
    aabb: tuple[int, int, int, int],
    mode: str,
    padding_px: int,
    fixed_size: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """依尺寸模式將標註外接框擴張成裁切區域

    Args:
        aabb: 標註的軸對齊外接框
        mode: CROP_MODE_PADDING 或 CROP_MODE_FIXED
        padding_px: padding 模式每邊外擴 pixel
        fixed_size: fixed 模式的最小邊長
        img_w, img_h: 影像寬高

    Returns:
        落在影像內的裁切區域 (x0, y0, x1, y1)
    """
    x0, y0, x1, y1 = aabb
    if mode == CROP_MODE_PADDING:
        rx0, ry0 = x0 - padding_px, y0 - padding_px
        rx1, ry1 = x1 + padding_px, y1 + padding_px
    else:
        # fixed：至少 fixed_size，以框中心置中；框比 fixed 大時取框尺寸
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w = max(fixed_size, x1 - x0)
        h = max(fixed_size, y1 - y0)
        rx0, rx1 = cx - w / 2, cx + w / 2
        ry0, ry1 = cy - h / 2, cy + h / 2
    return _fit_region(rx0, ry0, rx1, ry1, img_w, img_h)


def _rects_intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    """判斷兩矩形是否相交"""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _union(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """回傳兩矩形的聯集外接框"""
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _can_merge(c1: dict, c2: dict, mode: str, fixed_size: int) -> bool:
    """判斷兩個 cluster 是否可合併成同一張 cropped

    Args:
        c1, c2: cluster，含 "member" (成員聯集框) 與 "region" (裁切區域)
        mode: 尺寸模式
        fixed_size: fixed 模式最小邊長

    Returns:
        bool: 是否可合併
    """
    if mode == CROP_MODE_FIXED:
        # 成員聯集框需能塞進單一 fixed_size 裁切區才合併
        u = _union(c1["member"], c2["member"])
        return (u[2] - u[0]) <= fixed_size and (u[3] - u[1]) <= fixed_size
    # padding 模式：兩裁切區相交即合併
    return _rects_intersect(c1["region"], c2["region"])


def _translate_bbox(
    bbox: Bbox, rx0: int, ry0: int, rx1: int, ry1: int
) -> Bbox | None:
    """將 bbox 平移到裁切區座標系；軸對齊框裁邊，旋轉框僅平移

    Args:
        bbox: 原圖座標的 bbox
        rx0, ry0, rx1, ry1: 裁切區域 (原圖座標)

    Returns:
        平移後的 Bbox；完全在區域外則回傳 None
    """
    if bbox.angle % 360 != 0:
        # 旋轉框裁邊沒有意義，僅平移中心並保留寬高與角度
        return Bbox(
            bbox.x - rx0, bbox.y - ry0, bbox.width, bbox.height,
            bbox.label, bbox.confidence, bbox.angle,
        )

    # 軸對齊框：與裁切區取交集後平移
    nx0 = max(bbox.x, rx0)
    ny0 = max(bbox.y, ry0)
    nx1 = min(bbox.x + bbox.width, rx1)
    ny1 = min(bbox.y + bbox.height, ry1)
    w = nx1 - nx0
    h = ny1 - ny0
    if w <= 0 or h <= 0:
        return None
    return Bbox(
        int(nx0 - rx0), int(ny0 - ry0), int(w), int(h),
        bbox.label, bbox.confidence, bbox.angle,
    )


def _translate_polygon(
    polygon: Polygon, rx0: int, ry0: int, rx1: int, ry1: int
) -> Polygon | None:
    """將 polygon 平移到裁切區座標系，頂點超出範圍時夾到邊界 (近似裁切)

    Args:
        polygon: 原圖座標的 polygon
        rx0, ry0, rx1, ry1: 裁切區域 (原圖座標)

    Returns:
        平移後的 Polygon；點數不足則回傳 None
    """
    w = rx1 - rx0
    h = ry1 - ry0
    pts = []
    for px, py in polygon.points:
        nx = min(max(px - rx0, 0), w)
        ny = min(max(py - ry0, 0), h)
        pts.append((float(nx), float(ny)))
    if len(pts) < 3:
        return None
    return Polygon(pts, polygon.label, polygon.confidence)


def compute_crops(
    img_w: int,
    img_h: int,
    bboxes: list[Bbox],
    polygons: list[Polygon],
    mode: str = CROP_MODE_FIXED,
    padding_px: int = 50,
    fixed_size: int = 640,
) -> list[CropTask]:
    """依畫面標註計算 cropped 裁切任務清單

    每個標註各自外擴成裁切區域；相鄰、可落在同一裁切區內的標註會貪婪合併成一張。
    每個裁切區會納入所有與其相交的標註 (座標平移到裁切區座標系)。

    Args:
        img_w, img_h: 原圖寬高
        bboxes: 畫面上的 bbox 清單
        polygons: 畫面上的 polygon 清單
        mode: 尺寸模式 (CROP_MODE_PADDING / CROP_MODE_FIXED)
        padding_px: padding 模式每邊外擴 pixel
        fixed_size: fixed 模式最小邊長

    Returns:
        CropTask 清單；無標註時回傳空清單
    """
    # 1. 蒐集所有標註的軸對齊外接框
    items: list[tuple[tuple[int, int, int, int], str, object]] = []
    for b in bboxes:
        items.append((_bbox_aabb(b), "bbox", b))
    for p in polygons:
        if len(p.points) >= 3:
            items.append((_polygon_aabb(p), "poly", p))
    if not items:
        return []

    # 2. 每個標註各自成一個 cluster，再以「聯集框可被單一裁切區涵蓋」為條件貪婪合併
    clusters: list[dict] = []
    for aabb, _kind, _ref in items:
        clusters.append({
            "member": aabb,
            "region": _expand(aabb, mode, padding_px, fixed_size, img_w, img_h),
        })

    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if _can_merge(clusters[i], clusters[j], mode, fixed_size):
                    u = _union(clusters[i]["member"], clusters[j]["member"])
                    clusters[i]["member"] = u
                    clusters[i]["region"] = _expand(
                        u, mode, padding_px, fixed_size, img_w, img_h
                    )
                    clusters.pop(j)
                    merged = True
                    break
            if merged:
                break

    # 3. 每個 cluster 產生 CropTask，納入所有與該區域相交的標註 (平移+裁邊)
    tasks: list[CropTask] = []
    for c in clusters:
        rx0, ry0, rx1, ry1 = c["region"]
        task = CropTask(rx0, ry0, rx1, ry1)
        for aabb, kind, ref in items:
            if not _rects_intersect(aabb, c["region"]):
                continue
            if kind == "bbox":
                nb = _translate_bbox(ref, rx0, ry0, rx1, ry1)
                if nb is not None:
                    task.bboxes.append(nb)
            else:
                npoly = _translate_polygon(ref, rx0, ry0, rx1, ry1)
                if npoly is not None:
                    task.polygons.append(npoly)
        if task.bboxes or task.polygons:
            tasks.append(task)

    return tasks
