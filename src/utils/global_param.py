from pydantic import BaseModel


class GlobalParam(BaseModel):
    # 自動儲存計數器
    auto_save_counter: int = 0

    # 如果user label了, 就必定要儲存
    user_labeling: bool = False


g_param = GlobalParam()
