from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from ruamel.yaml import YAML

from src.utils.logger import getUniqueLogger

yaml = YAML()
log = getUniqueLogger(__file__)


class FileSystemSettings(BaseModel):
    folder_path: Optional[str] = None
    file_index: Optional[int] = 0


class ModelsSettings(BaseModel):
    active_model: Optional[str] = None
    model_path: Optional[str] = None
    sam3_model_path: Optional[str] = None
    polygon_tolerance: Optional[float] = 0.002
    sam3_label_mode: Optional[str] = "seg"  # "seg", "bbox", "all"


class ClassNamesSettings(BaseModel):
    categories: Optional[dict] = Field(default_factory=dict)
    text_prompts: Optional[list] = Field(
        default_factory=lambda: ["person", "cat", "dog", "car"]
    )


class Settings(BaseModel):
    file_system: FileSystemSettings = Field(default_factory=FileSystemSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    class_names: ClassNamesSettings = Field(default_factory=ClassNamesSettings)


def load_settings(file_path="cfg/settings.yaml"):
    path = Path(file_path)
    if not path.exists():
        log.info(f"{file_path} not found, generating default settings.")
        path.parent.mkdir(parents=True, exist_ok=True)
        default = Settings()
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(default.model_dump(), f)
        return default

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f)

    return Settings(**data)


def save_settings(file_path="cfg/settings.yaml"):
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(settings.model_dump(), f)


settings = load_settings()
