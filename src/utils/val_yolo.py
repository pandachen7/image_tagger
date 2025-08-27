from ultralytics import YOLO

model_path = r"./runs/detect/liyu_lake_voc/weights/best.pt"

model = YOLO(model_path)

results = model.val(
    conf=0.001,  # 降低置信度閾值，增加recall
    iou=0.7,  # 調整NMS IoU
    max_det=1000,  # 最大檢測數量
)

print(f"mAP@0.5: {results.box.map50}")      # mAP@0.5
print(f"mAP@0.5:0.95: {results.box.map}")   # mAP@0.5:0.95
print(f"各類別 AP@0.5:0.95: {results.box.ap}")
