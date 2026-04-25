# 系統設定載入：cfg/system.yaml 不存在時自動生成預設範本（含註解）；
# 存在時依 schema migrate（補新欄位、移除過時欄位），保留使用者既有設定值與註解。
# 更新日期: 2026-04-25
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from src.utils.logger import getUniqueLogger

# 使用 ruamel round-trip 模式，dump 時可保留既有註解與排版
yaml = YAML()
log = getUniqueLogger(__file__)


# 不存在 cfg/system.yaml 時，會以此範本寫入；以後 migrate 不會覆蓋使用者調整過的內容
_DEFAULT_TEMPLATE = """\
# 預設標籤 — 按數字碼切換標籤
# 支援單碼 (1)、雙碼 (12)、三碼 (123)
# 若某碼是另一碼的前綴（例如 1 和 12 同時存在），
# 短碼會等待 label_key_timeout 後才套用；長碼則立即套用。
labels:
  1: deer
  2: dog
  6: person
  9: vehicle
  0: unknown

# 繪製 bbox 時預設使用的標籤名稱
default_label: object

# 多碼 label 輸入的等待時間（秒），熟練後可調低以加快短碼反應
label_key_timeout: 2

# bbox 最小高度或寬度, polygon 則以最小邊長度
minimal_bbox_length: 30

# 自動儲存間隔（秒），-1 表示停用
# N 秒自動儲存以免重複性太高，但有些情況可能會跳過幀，請斟酌使用
auto_save_per_second: -1

# 是否在偵測時於 console 顯示 FPS
show_fps: false

# 儲存圖片與標籤的資料夾（可用相對或絕對路徑）
save_folder: ./output

# 是否啟用 Draw / Erase / Fill 遮罩工具
enable_mask_tools: false

# 是否啟用旋轉 (OBB) 功能
enable_obb: false

# 是否啟用 SAM3 模型（需另外申請下載 sam3.pt）
enable_sam3: false
"""


class Config(BaseModel):
    """system.yaml 的 schema；migrate 時以此為基準補/刪欄位"""
    # 載入時忽略已不存在於 schema 的舊欄位
    model_config = ConfigDict(extra="ignore")

    labels: dict[int, str] = {}
    default_label: str = "object"
    label_key_timeout: float = 2.0
    minimal_bbox_length: int = 30
    auto_save_per_second: float = -1
    show_fps: bool = False
    save_folder: str = "./output"
    enable_mask_tools: bool = False
    enable_obb: bool = False
    enable_sam3: bool = False


def load_config(file_path: str = "cfg/system.yaml") -> Config:
    """載入 system.yaml；不存在則寫入預設範本，存在則 migrate schema。

    Args:
        file_path: yaml 檔案路徑

    Returns:
        驗證過、補滿預設值的 Config 物件
    """
    path = Path(file_path)
    # 不存在則生成預設範本
    if not path.exists():
        log.i(f"{file_path} not found, generating default template.")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_DEFAULT_TEMPLATE, encoding="utf-8")
        except Exception as e:
            log.e(f"無法建立 {file_path}: {e}")
            raise

    # 載入既有檔案 (round-trip 保留註解)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.load(f)
    except Exception as e:
        log.e(f"無法解析 {file_path}: {e}")
        raise

    if data is None:
        data = CommentedMap()

    # 用 schema 驗證並補足缺欄位的預設值
    cfg_obj = Config(**dict(data))

    # Migrate: 比對 yaml keys 與 schema keys
    schema_keys = set(Config.model_fields.keys())
    data_keys = set(data.keys())
    obsolete = data_keys - schema_keys   # 已不存在於 schema → 移除
    missing = schema_keys - data_keys    # schema 新增但 yaml 沒有 → 補上

    changed = False
    for key in obsolete:
        try:
            del data[key]
            changed = True
            log.i(f"移除過時設定: {key}")
        except Exception as e:
            log.w(f"移除設定 {key} 失敗: {e}")
    for key in missing:
        data[key] = getattr(cfg_obj, key)
        changed = True
        log.i(f"新增預設設定: {key} = {getattr(cfg_obj, key)}")

    # 有變動才寫回，避免破壞原檔的時間戳與排版
    if changed:
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f)
            log.i(f"Schema migrate 完成: {file_path}")
        except Exception as e:
            log.e(f"寫回 {file_path} 失敗: {e}")

    return cfg_obj


cfg = load_config()


if __name__ == "__main__":
    print(cfg)
