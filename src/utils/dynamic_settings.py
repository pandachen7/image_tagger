from pydantic import BaseModel
from ruamel.yaml import YAML

from src.utils.loglo import getUniqueLogger

yaml = YAML()
log = getUniqueLogger(__file__)


class Settings(BaseModel):
    model_path: str = None
    folder_path: str = None
    file_index: int = 0
    categories: dict = {}


def load_settings(file_path="cfg/settings.yaml"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.load(f)

    return Settings(**data)


def save_settings(file_path="cfg/settings.yaml"):
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(settings.model_dump(), f)


settings = load_settings()
