# 動態設定管理：載入/儲存 settings.yaml，自動同步 schema 變更（補新欄位、移除過時欄位）
# 更新日期: 2026-04-08
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
    model_path: Optional[str] = None
    sam3_model_path: Optional[str] = None
    polygon_tolerance: Optional[float] = 0.002
    sam3_label_mode: Optional[str] = "seg"  # "seg", "bbox", "all"


class ClassNamesSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    categories: Optional[dict] = Field(default_factory=dict)
    text_prompts: Optional[list] = Field(
        default_factory=lambda: ["person", "cat", "dog", "car"]
    )


class Settings(BaseModel):
    # 忽略 yaml 中已不存在於 schema 的舊欄位，避免載入報錯
    model_config = ConfigDict(extra="ignore")

    file_system: FileSystemSettings = Field(default_factory=FileSystemSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    class_names: ClassNamesSettings = Field(default_factory=ClassNamesSettings)


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
