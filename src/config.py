from pydantic import BaseModel
from ruamel.yaml import YAML

yaml = YAML()


class Config(BaseModel):
    labels: dict[int, str] = {}
    default_label: str = "object"
    last_used_label: str = "object"
    auto_save_per_second: float = -1
    show_fps: bool = False
    save_folder: str = "./output"


def load_config(yaml_file: str = "cfg/config.yaml") -> Config:
    with open("cfg/system.yaml", "r", encoding="utf-8") as f:
        data = yaml.load(f)
    return Config(**data)


cfg = load_config()


if __name__ == "__main__":
    print(cfg)
