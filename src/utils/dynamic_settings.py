# 動態設定管理：載入/儲存 settings.yaml，自動同步 schema 變更（補新欄位、移除過時欄位）
# 更新日期: 2026-04-25
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from ruamel.yaml import YAML

from src.utils.logger import getUniqueLogger

yaml = YAML()
log = getUniqueLogger(__file__)


class FileSystemSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    folder_path: Optional[str] = None
    file_index: Optional[int] = 0


class ModelsSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    active_model: Optional[str] = None
    # YOLO
    model_path: Optional[str] = None
    yolo_label_mode: Optional[str] = "bbox"  # "seg", "bbox", "all"
    yolo_polygon_tolerance: Optional[float] = 0.002
    # SAM3
    sam3_model_path: Optional[str] = None
    sam3_polygon_tolerance: Optional[float] = 0.002
    sam3_label_mode: Optional[str] = "seg"  # "seg", "bbox", "all"


class ClassNamesSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    categories: Optional[dict] = Field(default_factory=dict)
    text_prompts: Optional[list] = Field(
        default_factory=lambda: ["person", "cat", "dog", "car"]
    )


class TrainingSettings(BaseModel):
    """YOLO 訓練參數暫存：對應 ultralytics model.train() 的常用參數"""
    model_config = ConfigDict(extra="ignore")

    # === 基本 (Train YOLO 主對話框) ===
    last_data_yaml: Optional[str] = ""
    task: Optional[str] = "detect"          # "detect" / "segment"
    model_size: Optional[str] = "s"          # n/s/m/l/x
    version: Optional[str] = "yolo26"
    epochs: Optional[int] = 100
    batch: Optional[int] = 16                # -1 = 自動
    imgsz: Optional[int] = 640
    patience: Optional[int] = 50
    device: Optional[str] = "0"              # "0" / "cpu" / "0,1"
    save_period: Optional[int] = -1

    # === 再訓練 / 續訓 ===
    # 指向之前訓練產出的 .pt（last.pt 或 best.pt）；空字串表示走 version+size 流程
    resume_pt_path: Optional[str] = ""
    # True 時對 ultralytics 傳 resume=True，從原訓練的 epoch、optimizer、scheduler 接續
    # （要求 .pt 來自 ultralytics 的 last.pt，且 dataset / 結構一致）
    resume_mode: Optional[bool] = False

    # === 進階：優化器 ===
    optimizer: Optional[str] = "auto"        # auto / SGD / Adam / AdamW / RMSProp
    lr0: Optional[float] = 0.01
    lrf: Optional[float] = 0.01
    weight_decay: Optional[float] = 0.0005
    warmup_epochs: Optional[float] = 3.0
    warmup_momentum: Optional[float] = 0.8

    # === 進階：幾何增強 ===
    degrees: Optional[float] = 0.0
    translate: Optional[float] = 0.1
    scale: Optional[float] = 0.5
    perspective: Optional[float] = 0.0

    # === 進階：翻轉 ===
    flipud: Optional[float] = 0.0
    fliplr: Optional[float] = 0.5

    # === 進階：色彩 (HSV) ===
    hsv_h: Optional[float] = 0.015
    hsv_s: Optional[float] = 0.7
    hsv_v: Optional[float] = 0.4

    # === 進階：混合增強 ===
    mosaic: Optional[float] = 1.0
    close_mosaic: Optional[int] = 10
    mixup: Optional[float] = 0.0
    copy_paste: Optional[float] = 0.0

    # === 進階：系統 ===
    workers: Optional[int] = 8
    cache: Optional[str] = "false"           # "false" / "ram" / "disk"
    rect: Optional[bool] = False
    amp: Optional[bool] = True
    fraction: Optional[float] = 1.0
    freeze: Optional[int] = 0                # 0 = 不凍結


class Settings(BaseModel):
    # 忽略 yaml 中已不存在於 schema 的舊欄位，避免載入報錯
    model_config = ConfigDict(extra="ignore")

    file_system: FileSystemSettings = Field(default_factory=FileSystemSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    class_names: ClassNamesSettings = Field(default_factory=ClassNamesSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)


def load_settings(file_path="cfg/settings.yaml"):
    """載入 settings，自動補齊新欄位並移除過時欄位，保持 yaml 與最新 schema 同步"""
    path = Path(file_path)
    if not path.exists():
        log.info(f"{file_path} not found, generating default settings.")
        path.parent.mkdir(parents=True, exist_ok=True)
        default = Settings()
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(default.model_dump(), f)
        return default

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f) or {}

    result = Settings(**data)

    # 比對讀入資料與最新 schema，有差異就更新 yaml（補新欄位 / 移除過時欄位）
    current_dump = result.model_dump()
    if data != current_dump:
        log.info(f"Schema changed, updating {file_path}.")
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(current_dump, f)

    return result


def save_settings(file_path="cfg/settings.yaml"):
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(settings.model_dump(), f)


settings = load_settings()
