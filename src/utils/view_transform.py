# 原始影像 pixel <-> widget pixel 的唯一換算處 (zoom + pan)。
# 更新日期: 2026-08-21
#
# 不變量 (改動本檔以外的地方時請維持):
#
# * Bbox / Polygon 的座標是「原始影像 pixel」, 那是唯一真值; 畫面像素只是投影。
# * 任何 原圖 <-> 螢幕 的換算都必須經由 ViewTransform, 不得在別處自行乘 zoom 或
#   加 offset —— 這是這類標註工具座標漂移的主要來源。
# * 只有兩個狀態: zoom (原圖 px -> widget px 的倍率) 與 off_x/off_y (原圖左上角
#   落在 widget 的位置)。縮放時 offset 由「游標下的原圖點保持不動」重新推導,
#   而非累加修正, 因此連續縮放不累積誤差。
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, QSize

# 允許的縮放範圍; 下限另外還會被 fit_zoom 的比例夾一次 (見 ImageWidget._min_zoom)
MIN_ZOOM = 0.02
MAX_ZOOM = 40.0


@dataclass
class ViewTransform:
    """原始影像座標與 widget 座標之間的縮放 / 平移換算"""

    img_w: int = 1
    img_h: int = 1
    zoom: float = 1.0
    off_x: float = 0.0
    off_y: float = 0.0

    def set_image_size(self, w: int, h: int) -> bool:
        """設定來源影像尺寸

        Args:
            w: 影像寬 (px)
            h: 影像高 (px)

        Returns:
            bool: 尺寸是否與原本不同 (呼叫端用來決定要不要重新 fit)
        """
        if w <= 0 or h <= 0:
            return False
        changed = (w != self.img_w) or (h != self.img_h)
        self.img_w = int(w)
        self.img_h = int(h)
        return changed

    @property
    def span_x(self) -> float:
        """整張影像在 widget 上的寬度 (px)"""
        return self.img_w * self.zoom

    @property
    def span_y(self) -> float:
        """整張影像在 widget 上的高度 (px)"""
        return self.img_h * self.zoom

    def o2v(self, ox: float, oy: float) -> QPointF:
        """原圖座標 -> widget 座標

        Args:
            ox: 原圖 x
            oy: 原圖 y

        Returns:
            QPointF: widget 座標
        """
        return QPointF(ox * self.zoom + self.off_x, oy * self.zoom + self.off_y)

    def v2o(self, vx: float, vy: float) -> QPointF:
        """widget 座標 -> 原圖座標

        回傳值可能落在影像外 (游標移出影像), 由呼叫端決定是否 clamp。

        Args:
            vx: widget x
            vy: widget y

        Returns:
            QPointF: 原圖座標
        """
        if self.zoom <= 0:
            return QPointF(vx, vy)
        return QPointF((vx - self.off_x) / self.zoom, (vy - self.off_y) / self.zoom)

    def o2v_len(self, length: float) -> float:
        """原圖長度 -> widget 長度

        Args:
            length: 原圖上的長度 (px)

        Returns:
            float: 對應的 widget 長度 (px)
        """
        return length * self.zoom

    def v2o_len(self, length: float) -> float:
        """widget 長度 -> 原圖長度

        Args:
            length: widget 上的長度 (px)

        Returns:
            float: 對應的原圖長度 (px)
        """
        if self.zoom <= 0:
            return length
        return length / self.zoom

    def image_rect(self) -> QRectF:
        """整張影像在 widget 上佔的矩形"""
        return QRectF(self.off_x, self.off_y, self.span_x, self.span_y)

    def fit_zoom(self, view: QSize) -> float:
        """讓整張影像剛好塞滿視窗所需的 zoom

        Args:
            view: widget 尺寸

        Returns:
            float: 對應的 zoom; 視窗尺寸無效時回傳目前的 zoom
        """
        if view.width() <= 0 or view.height() <= 0:
            return self.zoom
        return min(view.width() / self.img_w, view.height() / self.img_h)

    def fit(self, view: QSize) -> None:
        """縮放到整張影像可見並置中

        Args:
            view: widget 尺寸
        """
        if view.width() <= 0 or view.height() <= 0:
            return
        self.zoom = max(MIN_ZOOM, min(self.fit_zoom(view), MAX_ZOOM))
        self.center(view)

    def center(self, view: QSize) -> None:
        """把影像置中於視窗

        Args:
            view: widget 尺寸
        """
        self.off_x = (view.width() - self.span_x) / 2.0
        self.off_y = (view.height() - self.span_y) / 2.0

    def follow_view_resize(self, old_view: QSize, new_view: QSize) -> None:
        """視窗尺寸改變時, 讓影像跟著等比例縮放

        不變量是「相對於 fit 的倍率」: 視窗放大多少影像就放大多少, 使用者看到的
        構圖 (影像的哪個範圍落在畫面上) 在 resize 前後保持一致。若只保住 zoom,
        放大視窗只會在影像周圍長出空白, 還得再手動滾一次滾輪。

        比例取 fit_zoom 的變化而不是單純的寬 (或高) 比例: 只拉寬視窗時, 能容納的
        影像大小其實仍受高度限制, fit_zoom 已經把長寬比算進去了。

        錨點取視窗中心, 中心的原圖點在 resize 後仍留在中心。

        Args:
            old_view: 變動前的 widget 尺寸
            new_view: 變動後的 widget 尺寸
        """
        # 首次 resize 的 oldSize 是 (-1, -1), 視窗最小化時則會是 0
        if min(old_view.width(), old_view.height()) <= 0:
            return
        if min(new_view.width(), new_view.height()) <= 0:
            return
        old_fit = self.fit_zoom(old_view)
        new_fit = self.fit_zoom(new_view)
        if old_fit <= 0 or new_fit <= 0 or old_fit == new_fit:
            return
        center_o = self.v2o(old_view.width() / 2.0, old_view.height() / 2.0)
        self.zoom = max(MIN_ZOOM, min(self.zoom * (new_fit / old_fit), MAX_ZOOM))
        # 與 zoom_by 一樣由錨點反推 offset, 連續 resize 才不會累積誤差
        self.off_x = new_view.width() / 2.0 - center_o.x() * self.zoom
        self.off_y = new_view.height() / 2.0 - center_o.y() * self.zoom

    def zoom_by(
        self, factor: float, anchor: QPointF, min_zoom: float = MIN_ZOOM
    ) -> bool:
        """以 anchor 為錨點縮放, 錨點底下的原圖位置維持不動

        Args:
            factor: 縮放倍率 (>1 放大)
            anchor: 錨點的 widget 座標 (通常是游標位置)
            min_zoom: 這次允許的最小 zoom

        Returns:
            bool: zoom 是否真的改變
        """
        lo = max(MIN_ZOOM, min_zoom)
        target = min(max(self.zoom * factor, lo), MAX_ZOOM)
        if target == self.zoom:
            return False
        anchor_o = self.v2o(anchor.x(), anchor.y())
        self.zoom = target
        # 由錨點反推 offset 而非累加修正, 連續滾動才不會累積浮點誤差
        self.off_x = anchor.x() - anchor_o.x() * self.zoom
        self.off_y = anchor.y() - anchor_o.y() * self.zoom
        return True

    def pan_by(self, dx: float, dy: float) -> None:
        """平移檢視

        Args:
            dx: widget x 位移
            dy: widget y 位移
        """
        self.off_x += dx
        self.off_y += dy

    def clamp_offset(self, view: QSize) -> None:
        """影像比視窗小則置中, 比視窗大則不許拖出視窗外

        沒有這道夾制的話, 影像可以被拖到整個離開畫面, 使用者只看到空白而不知道
        該往哪拖回來。

        Args:
            view: widget 尺寸
        """
        if view.width() > 0:
            if self.span_x <= view.width():
                self.off_x = (view.width() - self.span_x) / 2.0
            else:
                self.off_x = min(max(self.off_x, view.width() - self.span_x), 0.0)
        if view.height() > 0:
            if self.span_y <= view.height():
                self.off_y = (view.height() - self.span_y) / 2.0
            else:
                self.off_y = min(max(self.off_y, view.height() - self.span_y), 0.0)
