from pydantic import BaseModel


class GlobalParam(BaseModel):
    auto_save_counter: int = 0  # 自動儲存計數器


global_param = GlobalParam()
