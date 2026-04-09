# YOLO 訓練腳本, 支援 detect/segment 模式
# updated: 2026-04-09
import os
import time
from datetime import datetime, timedelta

from ultralytics import YOLO

if __name__ == "__main__":
    start_time = time.time()
    print(f"開始訓練: {datetime.now()}")

    data_config_path = os.path.expanduser("~/datasets/img/liyu_lake_split/data.yaml")

    # 加載模型
    # - 如果您想從頭開始訓練，可以使用 .yaml 配置文件，例如 'yolov8n.yaml'
    # - 但通常建議使用預訓練權重 .pt 進行遷移學習. 例如設定yolo版本來訓練 e.g. yolo12m.pt
    # - 選擇上次中斷的last.pt的路徑以繼續訓練
    # - 若要訓練 segmentation, 請改用 seg 模型, e.g. yolo12m-seg.pt
    model_info = "yolo12m.pt"
    # model_info = "yolo12m-seg.pt"
    # model_info = "runs/detect/train3/weights/last.pt"
    model = YOLO(model_info)

    print(f"訓練model資訊: {model_info}")
    print(f"使用數據集配置: {data_config_path}")

    # 執行訓練
    # task 由模型架構決定: detect模型(.pt)→偵測, seg模型(-seg.pt)→分割
    # 若需指定可用 task="detect" 或 task="segment"
    results = model.train(
        # === 資料與輸出 ===
        data=data_config_path,  # data.yaml 路徑, 定義 train/val 圖片路徑與類別名稱
        name="liyu_lake_voc_ex",  # 輸出資料夾名稱, 結果存於 runs/detect/<name>/
        exist_ok=True,  # True=同名資料夾直接覆蓋, False=自動加後綴 (train2, train3...)
        plots=True,  # 訓練結束後產生 confusion_matrix, F1_curve, PR_curve 等圖表
        # === 訓練核心參數 ===
        epochs=600,  # 最大訓練輪數, 搭配 patience 可提前停止. 一般 300~600 足夠
        patience=50,  # early stopping: 連續 N 個 epoch mAP 無改善則停止, 0=關閉早停
        batch=30,  # 每批次圖片數. VRAM 不足時降低; -1=自動偵測最大可用 batch. 低 batch 梯度不穩, 建議 ≥16
        imgsz=640,  # 輸入影像解析度(px). 越大精度越好但越慢越吃 VRAM. 常見: 320/640/1280
        device=0,  # 訓練裝置. 0=第一張 GPU, "cpu"=CPU, "0,1"=多 GPU
        seed=42,  # 固定隨機種子, 確保可重現性
        # === 儲存設定 ===
        save=True,  # 是否儲存 best.pt 與 last.pt 權重檔
        save_period=100,  # 每 N 個 epoch 額外存一次 checkpoint (-1=關閉). 長時間訓練建議開啟以防中斷
        # === 驗證設定 ===
        val=True,  # 每個 epoch 結束後跑驗證集, 用於計算 mAP 與觸發 early stopping
        # conf=0.25,  # 驗證時的信心閾值. 會影響 mAP 計算, 盡量不要設定讓它用預設完整曲線
        # === 幾何增強 ===
        degrees=15.0,  # 隨機旋轉角度範圍 ±N°, default=0.0. 俯視場景適合加大
        translate=0.1,  # 隨機平移比例 (相對於圖片尺寸), default=0.1. 範圍 0.0~1.0
        scale=0.8,  # 隨機縮放比例 ±N, default=0.5. 模擬不同距離, 建議 0.5~1.2
        perspective=0.0005,  # 透視變換強度, default=0.0. 極小值即可, 太大會扭曲標註
        # === 翻轉增強 ===
        flipud=0.2,  # 上下翻轉機率, default=0.0. 俯視/紅外線場景可開啟
        fliplr=0.5,  # 左右翻轉機率, default=0.5. 大多數場景都適用
        # === 色彩增強（對灰階也有效）===
        hsv_h=0.015,  # 色相 (Hue) 偏移範圍, default=0.015. 灰階圖無色調可維持預設
        hsv_s=0.7,  # 飽和度 (Saturation) 變化範圍, default=0.7
        hsv_v=0.5,  # 亮度 (Value) 變化範圍, default=0.4. 紅外線場景可略增模擬強度變化
        # === 進階增強 ===
        mosaic=1.0,  # 馬賽克拼接機率, default=1.0. 4 張圖拼成 1 張增加小物件多樣性
        close_mosaic=10,  # 最後 N 個 epoch 關閉 mosaic, default=10. 讓模型最後學完整圖
    )

    """
    NOTE: 設定的詳細說明可直接看官方手冊
    其他可選設定，可直接加入 model.train() 中使用:

    # === 優化器 ===
    # 預設: SGD(lr=0.01, momentum=0.937)
    optimizer="AdamW",  # AdamW 對複雜場景更穩定, 但很依賴合適的初始 lr, 不然容易出現 NaN
    # === 學習率調整 ===
    lr0=0.002,  # 初始學習率, default=0.01. AdamW 建議降到 0.001~0.002
    lrf=0.01,  # 最終學習率 = lr0 × lrf, default=0.01. 越小衰減越多, 防止後期過擬合
    weight_decay=0.05,  # L2 正則化係數, default=0.0005. AdamW 建議 0.01~0.05
    warmup_epochs=3.0,  # 前 N 個 epoch 線性暖機, default=3.0. 穩定初期梯度
    warmup_momentum=0.8,  # 暖機期間起始 momentum, default=0.8

    # === 進階增強 ===
    mixup=0.0,  # MixUp 機率, default=0.0. 兩張圖疊加混合, 太高會模糊紅外線特徵
    copy_paste=0.0,  # Copy-Paste 機率, default=0.0. 複製物件貼到其他圖, 太高讓 loss 降不下

    # === 其他實用參數 ===
    workers=8,  # DataLoader 的 worker 數, default=8. Windows 上若有問題可設為 0
    cache=False,  # True=快取圖片到 RAM, "disk"=快取到硬碟, False=每次從磁碟讀取
    rect=False,  # True=使用矩形訓練(非正方形), 減少 padding 加速, 但可能降低精度
    resume=False,  # True=從 last.pt 繼續訓練 (需搭配 model_info 指向 last.pt)
    amp=True,  # 混合精度訓練, default=True. 減少 VRAM 用量並加速, 建議保持開啟
    fraction=1.0,  # 使用訓練集的比例, default=1.0. 設 0.1 可快速測試 pipeline
    freeze=None,  # 凍結前 N 層不更新, e.g. freeze=10. 遷移學習時可凍結 backbone
    """
    print("--- 訓練完成 ---")
    print(f"訓練結果保存在: {results.save_dir}")

    print(f"mAP@0.5: {results.box.map50}")  # mAP@0.5
    print(f"mAP@0.5:0.95: {results.box.map}")  # mAP@0.5:0.95
    print(f"各類別 AP@0.5:0.95: {results.box.ap}")

    print(f"結束訓練: {datetime.now()}")
    t_delta = timedelta(seconds=time.time() - start_time)
    print(f"訓練共花時: {t_delta}")
