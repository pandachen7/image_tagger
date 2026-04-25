# 訓練指南

本文說明如何將 Image Tagger 產出的標註，用於訓練 Ultralytics YOLO 模型。
涵蓋 **Object Detection** 和 **Segmentation** 兩種任務。

---

## 整體流程

```
標註圖片 → 儲存 VOC XML → Train → VOC to YOLO（轉換 + 分割 + 產生 yaml）→ Train → Train YOLO（GUI 內訓練）
```

---

## Step 1：標註並儲存

請先完成 [使用教學](./usage.md) 中的標註與儲存步驟。

---

## Step 2：VOC to YOLO（轉換 + 分割 + 產生 yaml）

**Train → VOC to YOLO**：選擇含有 VOC XML 的資料夾，在彈出的對話框中設定：
- **Class Mapping**：在對話框內按「編輯 Mapping」設定 class name → class_id 的對應
- **輸出模式**：BBox（Detection）或 Seg（Segmentation）
- **Train / Val 比例**：預設 80%/20%，可調整

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
val: images/val        # 若無 val split 則退回指向 images/train

nc: 3
names:
    0: person
    1: car
    2: dog
```

> `names` 的編號來自 VOC to YOLO 對話框內的 **Class Mapping** 設定。
> ultralytics 規定 `train` 與 `val` 兩個 key 必須存在；本工具產出的 yaml 永遠都會帶上 `val`。

---

## Step 3：訓練

### 方式 A：在 GUI 內訓練（推薦）

**Train → Train YOLO** 直接呼叫 ultralytics 訓練：

1. 選擇 `dataset.yaml`（會自動帶上次用過的或自動搜尋目前資料夾下的 `dataset_*.yaml`）
2. 設定 **Task**（Detect / Segment）、**Model Size**（n/s/m/l/x）、**Version**（預設 `yolo26`）
   - 對話框會顯示組合出的最終模型檔名（例如 `yolo26s.pt` 或 `yolo26m-seg.pt`），ultralytics 會自動下載
3. 調整基本訓練參數：Epochs / Batch / Image Size / Patience / Device / Save Period / Name
4. 需要更細的調整時按「**進階參數...**」：optimizer / lr / 增強 (HSV / Mosaic / MixUp) / cache / freeze / amp ... 等
5. 「**開始訓練**」後會顯示 `Epoch X/Y  mAP50=…`，完成後顯示輸出資料夾與最終 mAP，可按「開啟訓練資料夾」直接開啟 `runs/<task>/<name>/`

> 所有基本與進階參數都暫存在 `cfg/settings.yaml` 的 `training` 區段，下次再開直接帶回上次的值。

### 方式 B：用 Python 腳本

如果想完全用程式控制訓練流程：

```python
from ultralytics import YOLO

# Object Detection
model = YOLO("yolo26s.pt")          # detection 模型
# Segmentation 改用 seg 版本：
# model = YOLO("yolo26s-seg.pt")    # 標籤要是 polygon 格式

results = model.train(
    data="path/to/dataset.yaml",
    epochs=300,
    imgsz=640,
    batch=16,       # VRAM 不夠就降低
    device=0,       # 第一張 GPU
)
```

> Ultralytics 會自動根據模型架構決定 task（detect / segment），不需要手動指定 `task` 或 `mode` 參數。

專案另提供了完整範例 `src/for_training/train_yolo.py`，內含常用的訓練參數與增強設定的詳細註解，可作為進階參考：

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
轉換前請在 **Train → VOC to YOLO** 對話框內選對輸出模式。

### 訓練到一半可以接續嗎？

可以。將模型路徑指向上次的 `last.pt`：
```python
model = YOLO("runs/detect/train/weights/last.pt")
results = model.train(data="path/to/data.yaml", epochs=600, ...)
```
