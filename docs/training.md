# 訓練指南

本文說明如何將 Image Tagger 產出的標註，用於訓練 Ultralytics YOLO 模型。
涵蓋 **Object Detection** 和 **Segmentation** 兩種任務。

---

## 整體流程

```
標註圖片 → 儲存 VOC XML → 轉換為 YOLO txt → 整理 dataset 結構 → 編寫 data.yaml → 訓練
```

---

## Step 1：標註並轉換

請先完成 [使用教學](./usage.md) 中的標註與轉換步驟。
轉換完成後，你會得到一堆 `.txt` 標籤檔，每個檔案對應一張圖片。

**Detection 格式**（每行一個物件）：
```
class_id cx cy w h
```

**Segmentation 格式**（每行一個物件）：
```
class_id x1 y1 x2 y2 ... xN yN
```

> 所有座標都是 0~1 的正規化值。

---

## Step 2：整理 Dataset 結構

Ultralytics 要求 images 和 labels 放在平行的資料夾中，且檔名必須一一對應。

### 最簡結構（全部當訓練集）

```
my_dataset/
├── data.yaml
└── train/
    ├── images/
    │   ├── 001.jpg
    │   ├── 002.jpg
    │   └── ...
    └── labels/
        ├── 001.txt
        ├── 002.txt
        └── ...
```

### 建議結構（含驗證集和測試集）

```
my_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/          # 選用
    ├── images/
    └── labels/
```

專案內附 `src/for_training/split_dataset.py` 可自動拆分資料集。
修改檔案開頭的設定即可：

```python
SOURCE_DATA_DIR = "~/datasets/img/my_annotated_data"   # 原始標註資料
OUTPUT_DATA_DIR = "~/datasets/img/my_dataset_split"    # 拆分後的輸出
SPLIT_RATIOS = [0.8, 0.1, 0.1]                        # train / valid / test
```

然後執行：
```bash
python src/for_training/split_dataset.py
```

> 注意：執行前需要在 `SOURCE_DATA_DIR/train/` 下準備好 `images/` 和 `labels/` 兩個子資料夾，以及根目錄要有 `data.yaml`。

---

## Step 3：編寫 data.yaml

```yaml
# 路徑為相對於 data.yaml 的位置
train: ./train/images
val: ./valid/images
# test: ./test/images  # 選用

# 類別數量
nc: 3

# 類別名稱（編號必須與 Convert → Edit Categories 一致）
names:
  0: person
  1: cat
  2: dog
```

> `names` 的編號要和你在 Image Tagger 的 **Edit Categories** 中設定的對應一致。
> 例如你設了 `person → 0`，這裡就是 `0: person`。

> 如果只想快速測試，`train` 和 `val` 可以指向同一個資料夾。

---

## Step 4：訓練

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
2. 標籤檔要是 **polygon 格式**（Convert → Settings 選 Seg）

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
