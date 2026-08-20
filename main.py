# 程式進入點: 印出 torch/CUDA 資訊並啟動 GUI。
# 另安裝全域例外攔截, 避免 PyQt6 在 Qt slot 內遇到未捕捉例外時直接 abort() 行程 (無 traceback)。
# updated: 2026-08-21
import os

# 必須在任何會連帶 import matplotlib 的套件 (如 ultralytics) 之前設定。
# matplotlib 預設 backend 為 qtagg, import 時會連帶把 PyQt6(Qt) 拉進來; 本 app 用 PyQt6
# 直接繪圖, 不需要 matplotlib 的互動 backend, 改用非 GUI 的 Agg 以避免多餘的 Qt 相依。
os.environ.setdefault("MPLBACKEND", "Agg")

# 關掉 ultralytics 的執行期 AutoUpdate。它在 import 失敗的當下會自己 pip 裝套件
# (SAM3 的 clip、timm 都走這條), 但相依已由 pyproject 釘死並交給 uv sync 管理,
# 偷裝的版本不受 lock 控制, 下次 uv sync 又會被清掉, 於是每次 sync 後第一次跑
# SAM3 都要重下載。設 1 之後缺套件會當場 ModuleNotFoundError, 代表 pyproject
# 漏宣告了東西, 早點爆出來比較好修。
# 只擋套件安裝, 模型權重的自動下載走 attempt_download_asset, 不受影響。
# 用 setdefault 是為了保留從外部環境變數覆寫成 0 的餘地。
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

import sys

import torch  # torch必須比pyqt還早, 以免索引出錯

from src.object_tagger import main
from src.utils.logger import getUniqueLogger

log = getUniqueLogger(__file__)


def _install_excepthook() -> None:
    """安裝全域例外攔截。

    PyQt6 在 Qt slot (如 QAction.triggered) 內拋出未捕捉例外時, 預設會直接
    abort() 整個行程且不印 traceback, 表現為「程式直接跳掉」。改寫 sys.excepthook
    後例外會被完整記錄, 且行程不會被強制結束, 方便定位錯誤。
    """
    def _hook(exc_type: type[BaseException], exc: BaseException, tb) -> None:
        import traceback

        detail = "".join(traceback.format_exception(exc_type, exc, tb))
        log.error(f"未捕捉例外:\n{detail}")

    sys.excepthook = _hook


# 注意：訓練時 DataLoader worker 在 Windows 用 spawn 啟動會重新 import main.py，
# module-level 的 print 會被印很多次，所以放進 __main__ guard。
if __name__ == "__main__":
    _install_excepthook()
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    print("cudnn version:", torch.backends.cudnn.version())
    main()
