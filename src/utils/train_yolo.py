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
        save=True,
        save_period=100,
        device=0,
        imgsz=640,
        batch=-1,  # 如果 VRAM 不足，請降低此數值。-1 代表自動調整。
        name="liyu_lake_voc",
        exist_ok=True,
        close_mosaic=10,
        plots=True,
    )

    print("--- 訓練完成 ---")
    print(f"訓練結果保存在: {results.save_dir}")


if __name__ == "__main__":
    main()
