# 標註的 Undo / Redo 歷史。以「變更前的整份標註快照」為一個單位, 換檔即清空。
# 更新日期: 2026-08-18
from __future__ import annotations

from src.utils.logger import getUniqueLogger
from src.utils.model import Bbox, Polygon

log = getUniqueLogger(__file__)

# 一份快照 = (所有 bbox 的純資料, 所有 polygon 的純資料)
Snapshot = tuple[tuple, tuple]


def make_snapshot(bboxes: list[Bbox], polygons: list[Polygon]) -> Snapshot:
    """把目前的標註轉成一份不可變的快照

    Args:
        bboxes: 目前的 bbox 清單
        polygons: 目前的 polygon 清單

    Returns:
        Snapshot: 可直接用 == 比對是否有變更的純資料
    """
    return (
        tuple(b.snapshot() for b in bboxes),
        tuple(p.snapshot() for p in polygons),
    )


def restore_snapshot(snap: Snapshot) -> tuple[list[Bbox], list[Polygon]]:
    """把快照重建成標註物件

    Args:
        snap: make_snapshot() 產生的快照

    Returns:
        tuple: (bbox 清單, polygon 清單), 皆為新物件
    """
    bbox_snaps, poly_snaps = snap
    return (
        [Bbox.from_snapshot(s) for s in bbox_snaps],
        [Polygon.from_snapshot(s) for s in poly_snaps],
    )


class AnnotationHistory:
    """單一影像的標註 undo / redo 歷史

    快照式而非命令式: 一張圖的標註只有數十個小物件, 複製成本可忽略; 而 resize、
    旋轉、頂點拖曳這類連續操作若用命令物件, 很容易漏記反向狀態。

    歷史屬於「目前這張影像」, 換檔或影片換幀時由 clear() 重置 —— 換檔會重讀 XML,
    保留跨檔歷史會讓 undo 把上一張的框寫進這一張。
    """

    def __init__(self, limit: int = 60) -> None:
        """
        Args:
            limit: undo 堆疊的最大筆數, 超過時丟棄最舊的一筆
        """
        self._limit = max(1, int(limit))
        self._undo: list[Snapshot] = []
        self._redo: list[Snapshot] = []

    @property
    def depth(self) -> int:
        """undo 堆疊目前的步數 (可還原幾步)"""
        return len(self._undo)

    @property
    def can_undo(self) -> bool:
        """是否還有可還原的步驟"""
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        """是否還有可重做的步驟"""
        return bool(self._redo)

    def clear(self) -> None:
        """清空整個歷史 (換檔 / 換幀 / 重新載入時呼叫)"""
        self._undo.clear()
        self._redo.clear()

    def push(self, bboxes: list[Bbox], polygons: list[Polygon]) -> None:
        """在變更「之前」記錄目前狀態

        呼叫端一律在動手改之前呼叫, 因此堆疊裡存的都是「上一步的樣子」。
        任何新動作都會讓 redo 失效, 故一併清空。

        Args:
            bboxes: 變更前的 bbox 清單
            polygons: 變更前的 polygon 清單
        """
        try:
            self._undo.append(make_snapshot(bboxes, polygons))
        except Exception as e:
            log.e(f"建立 undo 快照失敗: {e}")
            return
        if len(self._undo) > self._limit:
            self._undo.pop(0)
        self._redo.clear()

    def drop_last(self) -> None:
        """撤掉剛才 push 的那一筆

        給「按下去了但其實沒改到東西」的情況用 (例如點到 resize 控制點卻沒拖動),
        否則 undo 會出現按了沒反應的空步驟。
        """
        if self._undo:
            self._undo.pop()

    def matches_last(self, bboxes: list[Bbox], polygons: list[Polygon]) -> bool:
        """目前狀態是否與最後一筆快照相同 (即這次動作沒有真的改到東西)

        Args:
            bboxes: 目前的 bbox 清單
            polygons: 目前的 polygon 清單

        Returns:
            bool: 相同則為 True; 堆疊為空時回傳 False
        """
        if not self._undo:
            return False
        try:
            return self._undo[-1] == make_snapshot(bboxes, polygons)
        except Exception as e:
            log.e(f"比對 undo 快照失敗: {e}")
            return False

    def undo(
        self, bboxes: list[Bbox], polygons: list[Polygon]
    ) -> tuple[list[Bbox], list[Polygon]] | None:
        """還原上一步

        Args:
            bboxes: 目前的 bbox 清單 (會被存進 redo 堆疊)
            polygons: 目前的 polygon 清單

        Returns:
            還原後的 (bbox 清單, polygon 清單); 無可還原步驟時回傳 None
        """
        if not self._undo:
            return None
        try:
            self._redo.append(make_snapshot(bboxes, polygons))
            return restore_snapshot(self._undo.pop())
        except Exception as e:
            log.e(f"undo 失敗: {e}")
            return None

    def redo(
        self, bboxes: list[Bbox], polygons: list[Polygon]
    ) -> tuple[list[Bbox], list[Polygon]] | None:
        """重做被還原的那一步

        Args:
            bboxes: 目前的 bbox 清單 (會被存回 undo 堆疊)
            polygons: 目前的 polygon 清單

        Returns:
            重做後的 (bbox 清單, polygon 清單); 無可重做步驟時回傳 None
        """
        if not self._redo:
            return None
        try:
            self._undo.append(make_snapshot(bboxes, polygons))
            return restore_snapshot(self._redo.pop())
        except Exception as e:
            log.e(f"redo 失敗: {e}")
            return None
