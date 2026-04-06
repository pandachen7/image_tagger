# 訓練指南

本文說明如何將 Image Tagger 產出的標註，用於訓練 Ultralytics YOLO 模型。
涵蓋 **Object Detection** 和 **Segmentation** 兩種任務。

---

## 整體流程

```
標註圖片 → 儲存 VOC XML → Convert → VOC to YOLO（自動轉換 + 分割 + 產生 yaml）→ 訓練
```

---

## Step 1：標註並轉換

請先完成 [使用教學](./usage.md) 中的標註與儲存步驟。

---

## Step 2：VOC to YOLO（轉換 + 分割 + 產生 yaml）

1. **Convert → Edit Categories**：設定 class name → class_id 的對應
2. **Convert → VOC to YOLO**：選擇含有 VOC XML 的資料夾，在彈出的對話框中設定：
   - **輸出模式**：BBox（Detection）或 Seg（Segmentation）
   - **Train / Val 比例**：預設 100% train，可調整（如 80%/20%）

工具會自動完成以下步驟：
- 將 VOC XML 轉換為 YOLO `.txt`（所有座標為 0~1 正規化值）
- 依比例將圖片和標籤移動到 `images/train`、`images/val` 和 `labels/train`、`labels/val`
- 產生 `dataset_YYYY_MMDD_HHMMSS.yaml`

**YOLO 標籤格式：**

| 模式 | 格式 |
|------|------|
| Detection | `class_id cx cy w h` |
| Segmentation | `class_id x1 y1 x2 y2 ... xN yN` |

**產生的 dataset 結構：**

```
my_dataset/
├── dataset_2026_0406_153042.yaml
├── images/
│   ├── train/
│   └── val/       # 若 train 設為 100% 則不產生
└── labels/
    ├── train/
    └── val/
```

**產生的 yaml 內容範例：**

```yaml
path: /data/my_dataset
train: images/train
val: images/val        # 若無 val set 則不包含此行

nc: 3
names:
    0: person
    1: car
    2: dog
```

> `names` 的編號來自 **Edit Categories** 中設定的 class_id。

---

## Step 3：訓練

### Object Detection

```python
from ultralytics import YOLO

model = YOLO("yolo12m.pt")  # 使用預訓練的 detection 模型

results = model.train(
    data="path/to/data.yaml",
    epochs=300,
    imgsz=640,
    batch=16,       # VRAM 不夠就降低
    device=0,       # 第一張 GPU
)
```

### Segmentation

與 detection 的差別只有兩點：
1. 模型要用 **seg 版本**（檔名含 `-seg`）
2. 標籤檔要是 **polygon 格式**（轉換時選 Seg 模式）

```python
from ultralytics import YOLO

model = YOLO("yolo12m-seg.pt")  # 使用預訓練的 seg 模型

results = model.train(
    data="path/to/data.yaml",
    epochs=300,
    imgsz=640,
    batch=16,
    device=0,
)
```

> Ultralytics 會自動根據模型架構決定 task（detect / segment），不需要手動指定 `task` 或 `mode` 參數。

### 使用專案內的訓練腳本

專案提供了完整範例 `src/for_training/train_yolo.py`，內含常用的訓練參數與增強設定，可依需求調整：

```bash
python src/for_training/train_yolo.py
```

---

## 訓練參數建議

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `epochs` | 訓練輪數 | 300~600，搭配 patience 早停 |
| `patience` | 早停耐心值 | 50（連續 50 epoch 沒進步就停） |
| `batch` | 批次大小 | VRAM 24GB → 32, 12GB → 16, 8GB → 8 |
| `imgsz` | 輸入解析度 | 640（預設），小物件可提高到 1280 |
| `device` | GPU 編號 | `0`（單卡），`[0,1]`（多卡） |

> 完整參數文件請參考 [Ultralytics Train 官方文件](https://docs.ultralytics.com/modes/train/)。

---

## 驗證模型

訓練完成後，最佳權重在 `runs/detect/train/weights/best.pt`（或 `runs/segment/train/weights/best.pt`）。

```bash
python src/for_training/val_yolo.py
```

也可以直接把 `best.pt` 載入 Image Tagger（**Ai → Select Model**）來即時體驗偵測效果。

---

## 常見問題

### 標籤檔和圖片對不上

確認每張圖片都有對應的 `.txt`，且檔名一致（只有副檔名不同）。
例如 `001.jpg` 對應 `001.txt`。沒有標註的背景圖可以放一個空的 `.txt`。

### Detection 和 Segmentation 的標籤格式可以混用嗎？

不行。Detection 模型需要 `cx cy w h` 格式，Segmentation 模型需要 polygon 格式。
轉換前請在 **Convert → Settings** 確認選對模式。

### 訓練到一半可以接續嗎？

可以。將模型路徑指向上次的 `last.pt`：
```python
model = YOLO("runs/detect/train/weights/last.pt")
results = model.train(data="path/to/data.yaml", epochs=600, ...)
```
