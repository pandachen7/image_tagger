import os
import time
from datetime import datetime, timedelta

from ultralytics import YOLO

start_time = time.time()
print(f"開始訓練: {datetime.now()}")

data_config_path = os.path.expanduser(
    "~/datasets/img/2024_liyu_lake_voc_split/data.yaml"
)

# 加載模型
# - 如果您想從頭開始訓練，可以使用 .yaml 配置文件，例如 'yolov8n.yaml'
# - 但通常建議使用預訓練權重 .pt 進行遷移學習. 例如設定yolo版本來訓練 e.g. yolo12m.pt
# - 選擇上次中斷的model路徑以進行再訓練
model_name = "yolo12m.pt"
model = YOLO(model_name)

print(f"訓練model資訊: {model_name}")
print(f"使用數據集配置: {data_config_path}")

# 執行訓練
# device=0 代表使用第一張 GPU
results = model.train(
    data=data_config_path,
    mode="detect",  # 這裡是指object detection(detect), 其他還有segmentation(segment), classification(classify)等等
    epochs=500,  # 紅外線大概300~500就差不多
    patience=50,  # 早停和保存
    save=True,
    save_period=100,
    device=0,
    imgsz=640,  # 輸入解析度
    batch=30,  # 如果 VRAM 不足，請降低此數值。-1 代表自動調整, 但低batch會導致梯度不穩, 建議VRAM拿24GB以上來訓練(batch=32)
    name="liyu_lake_voc_ex",
    exist_ok=True,  # 同名資料夾則蓋過
    close_mosaic=10,
    plots=True,
    # 幾何增強（俯視角度）
    degrees=30.0,  # 增加旋轉，俯視很適合
    translate=0.1,  # 增加平移
    scale=0.8,  # 大範圍縮放模擬距離變化, 建議0.5~1.2
    perspective=0.0005,  # 透視變換
    # 翻轉
    flipud=0.2,  # 紅外線俯視增加上下翻轉
    fliplr=0.5,
    # === 色彩增強（對灰階也有效）===
    hsv_h=0.015,  # 色相變化（灰階無色調）, default=0.015
    hsv_s=0.7,  # 飽和度變化, default=0.7
    hsv_v=0.4,  # 亮度變化（模擬紅外線強度變化）, default=0.4
    # === 高級增強 ===
    mosaic=1.0,  # 馬賽克拼接增加多樣性, default=1.0
    # === NMS設定 ===
    iou=0.7,  # 融合IoU閾值, default=0.7
    # === 驗證設定 ===
    val=True,
    # === 其他重要設定 ===
    seed=42,  # 固定隨機種子
    # === 推論相關（訓練時也會影響驗證）===
    conf=0.25,  # 偵測門檻, default=0.25
)

"""
NOTE: 建議這些設定說明直接看官方手冊, AI通常都是照感覺亂給
其他設定
optimizer預設
SGD(lr=0.01, momentum=0.937)

# === 優化器 ===
optimizer="AdamW",  # AdamW對複雜場景更穩定, 但很依賴合適的初始 learning rate, 不然都會出現nan
# === 學習率調整 ===
lr0=0.002,  # 初始學習率
lrf=0.01,  # 最終學習率比例，較低防止過擬合
weight_decay=0.05,  # 稍微增加正則化

# === 高級增強 ===
mixup=0.0,  # 降低MixUp，避免破壞紅外線特徵, 太高會讓loss降不下, 可先關掉
copy_paste=0.0,  # 適度使用, 太高會讓loss降不下, 可先關掉
"""
print("--- 訓練完成 ---")
print(f"訓練結果保存在: {results.save_dir}")

print(f"mAP@0.5: {results.box.map50}")  # mAP@0.5
print(f"mAP@0.5:0.95: {results.box.map}")  # mAP@0.5:0.95
print(f"各類別 AP@0.5:0.95: {results.box.ap}")

print(f"結束訓練: {datetime.now()}")
t_delta = timedelta(seconds=time.time() - start_time)
print(f"訓練共花時: {t_delta}")
