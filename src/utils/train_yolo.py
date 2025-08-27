import os

from ultralytics import YOLO


def main():
    model_name = "yolo12m.pt"

    data_config_path = os.path.expanduser(
        "~/datasets/img/2024_liyu_lake_voc_split/data.yaml"
    )

    # 加載模型
    # 如果您想從頭開始訓練，可以使用 .yaml 配置文件，例如 'yolov8n.yaml'
    # 但通常建議使用預訓練權重 .pt 進行遷移學習
    model = YOLO(model_name)

    print(f"使用模型: {model_name}")
    print(f"使用數據集配置: {data_config_path}")

    # 執行訓練
    # device=0 代表使用第一張 GPU
    results = model.train(
        data=data_config_path,
        mode="detect",
        epochs=1000,
        patience=80,  # 早停和保存
        save=True,
        save_period=100,
        device=0,
        imgsz=640,
        batch=-1,  # 如果 VRAM 不足，請降低此數值。-1 代表自動調整。
        name="liyu_lake_voc_enhanced",
        exist_ok=True,
        close_mosaic=10,
        plots=True,
        # === 學習率調整 ===
        lr0=0.01,  # 初始學習率
        lrf=0.01,  # 最終學習率比例，較低防止過擬合
        momentum=0.9,
        weight_decay=0.001,  # 稍微增加正則化
        warmup_epochs=5.0,  # 更長預熱
        warmup_momentum=0.8,
        # === 灰階+彩色混合的資料增強 ===
        # 幾何增強（俯視角度）
        degrees=60.0,  # 增加旋轉，紅外線下方向識別更困難
        translate=0.2,  # 增加平移
        scale=0.8,  # 大範圍縮放模擬距離變化
        shear=15.0,  # 增加剪切
        perspective=0.0005,  # 透視變換
        # 翻轉
        flipud=0.3,  # 紅外線俯視增加上下翻轉
        fliplr=0.5,
        # === 色彩增強（對灰階也有效）===
        hsv_h=0.01,  # 降低色調變化（灰階無色調）
        hsv_s=0.3,  # 降低飽和度變化
        hsv_v=0.6,  # 增加亮度變化（重要：模擬紅外線強度變化）
        # === 高級增強 ===
        mosaic=0.8,  # 稍微降低，紅外線圖像可能較單調
        mixup=0.1,  # 降低MixUp，避免破壞紅外線特徵
        copy_paste=0.2,  # 適度使用
        # === YOLOv12m 特定參數 ===
        box=10.0,  # 增加box loss權重，紅外線邊界可能模糊
        cls=1.0,  # 增加分類權重，灰階特徵較少
        dfl=2.0,  # 增加DFL權重
        # === 正則化 ===
        label_smoothing=0.15,  # 增加標籤平滑，提高泛化
        dropout=0.1,  # 如果支持，添加dropout
        # === 優化器 ===
        optimizer="AdamW",  # AdamW對複雜場景更穩定
        # === NMS設定 ===
        iou=0.65,  # 稍微降低，紅外線邊界可能不清晰
        # === 多尺度訓練 ===
        rect=True,
        # === 驗證設定 ===
        val=True,
        # === 混合精度 ===
        amp=True,  # 加速訓練
        # === 其他重要設定 ===
        workers=6,  # 減少worker避免記憶體問題
        seed=42,  # 固定隨機種子
        # === 推論相關（訓練時也會影響驗證）===
        conf=0.15,  # 降低置信度閾值，紅外線可能較模糊
        max_det=200,  # 增加最大檢測數
    )

    print("--- 訓練完成 ---")
    print(f"訓練結果保存在: {results.save_dir}")

    print(f"mAP@0.5: {results.box.map50}")  # mAP@0.5
    print(f"mAP@0.5:0.95: {results.box.map}")  # mAP@0.5:0.95
    print(f"各類別 AP@0.5:0.95: {results.box.ap}")


if __name__ == "__main__":
    main()
