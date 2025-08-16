from pathlib import Path

from ruamel.yaml import YAML

from src.loglo import getUniqueLogger

yaml = YAML()
log = getUniqueLogger(__file__)


def update_dynamic_config():
    with open("cfg/settings.yaml", "w", encoding="utf-8") as f:
        yaml.dump(Settings.data, f)


class Settings:
    """
    程式正常關閉後自動儲存
    """

    data = {"model_path": None, "folder_path": None, "file_index": 0, "categories": {}}
    try:
        with open("config/settings.yaml", "r", encoding="utf-8") as f:
            yaml_settings = yaml.load(f)
        data["model_path"] = yaml_settings.get("model_path", None)
        data["folder_path"] = yaml_settings.get("folder_path", None)
        data["file_index"] = int(yaml_settings.get("file_index", 0))
        data["categories"] = yaml_settings.get("categories", {})

        # check validation
        if data["model_path"] is None or Path(data["model_path"]).is_file() is False:
            data["model_path"] = None

        if data["folder_path"] is None or Path(data["folder_path"]).is_dir() is False:
            data["folder_path"] = None

        if not isinstance(data["file_index"], int) or data["file_index"] < 0:
            data["file_index"] = 0

    except Exception as e:
        log.e(f"Load config/settings.yaml failed: {e}")
