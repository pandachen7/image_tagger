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
- **Train / Val 比例**：預設 80%/20%，可設 50~95%（**val 一定會保留一部分**，理由見下方）

工具會自動完成以下步驟：
- 將 VOC XML 轉換為 YOLO `.txt`（所有座標為 0~1 正規化值）
- 依比例將圖片和標籤移動到 `images/train`、`images/val` 和 `labels/train`、`labels/val`
- 產生 `dataset_YYYY_MMDD_HHMMSS.yaml`
- 轉檔前會先問要不要清空上一次的 split（保留舊檔會讓同一張圖同時出現在 train 與 val）
- 刪除 ultralytics 的舊標籤快取 `labels/*.cache`（它的 hash 只看檔案大小與路徑，改了 class id 可能不會失效）

**轉檔時的三個重要規則：**

1. **`nc` / `names` 只含實際出現的類別，並重新編號成 0 起連號。**
   Class Mapping 裡沒用到的類別不會寫進 yaml —— 除了讓模型多學幾個空類別，
   `optimizer=auto` 還是用 `nc` 算學習率的（`lr = 0.002*5/(4+nc)`），
   nc 從 2 變成 13 學習率就掉了 2.8 倍。
2. **框的 class name 全部對不上 Class Mapping 的圖片，整張排除在 dataset 外。**
   ultralytics 把「空標籤」和「沒有標籤檔」都當成背景圖，若寫出空 `.txt`，
   圖裡的物件會被當成「背景」教給模型，比整張不收更糟。
   刻意存的背景樣本（本來就沒框）不受影響，仍會保留空 `.txt`。
3. **同一張原圖的裁切（`_cropN`）與同一段影片的幀（`_frameN`）會整組落在同一邊。**
   這些檔案內容高度相似，被拆到 train / val 兩邊會讓驗證分數虛高，
   進而誤導 early stopping 與 best.pt 的挑選。

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
│   └── val/
└── labels/
    ├── train/
    └── val/
```

**產生的 yaml 內容範例：**

```yaml
path: /data/my_dataset
train: images/train
val: images/val

nc: 3
names:
    0: person
    1: car
    2: dog
```

> `names` 的名稱來自 **Class Mapping**，但編號是依「實際出現的類別」重新編過的連號，
> 不一定等於 Class Mapping 裡填的 class_id（對應關係會列在轉換完成的摘要視窗裡）。
>
> **為什麼 val 不能是空的**：ultralytics 用 val 的 mAP50-95 當 fitness，
> `best.pt` 是 fitness 最高的那一輪、early stopping 也看它。
> val 指向 train 的話 fitness 反映的是訓練集表現，早停永遠不會觸發，
> best.pt 會挑到最過擬合的權重，訓練過程顯示的 mAP 也完全不能參考。

---

## Step 3：訓練

### 方式 A：在 GUI 內訓練（推薦）

**Train → Train YOLO** 直接呼叫 ultralytics 訓練：

1. 選擇 `dataset.yaml`（會自動帶上次用過的或自動搜尋目前資料夾下的 `dataset_*.yaml`）
2. **（選填）Resume from .pt**：如果想接續之前訓練的權重，按「瀏覽...」選 `runs/<task>/<name>/weights/last.pt` 或 `best.pt`。留空則走步驟 3 的預訓練模型流程。詳見下方 [再訓練 / 繼續訓練](#再訓練--繼續訓練)
3. 設定 **Task**（Detect / Segment）、**Model Size**（n/s/m/l/x）、**Version**（預設 `yolo26`）
   - 對話框會顯示組合出的最終模型檔名（例如 `yolo26s.pt` 或 `yolo26m-seg.pt`），ultralytics 會自動下載
   - 若上面已指定 Resume `.pt`，這幾項會被鎖住（由 `.pt` 自動決定）
4. 調整基本訓練參數：Epochs / Batch / Image Size / Patience / Device / Save Period / Name
5. 需要更細的調整時按「**進階參數...**」：optimizer / lr / 增強 (HSV / Mosaic / MixUp) / cache / freeze / amp ... 等
6. 「**開始訓練**」後會顯示 `Epoch X/Y  mAP50=…`，完成後顯示輸出資料夾與最終 mAP，可按「開啟訓練資料夾」直接開啟 `runs/<task>/<name>/`

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
| `epochs` | 訓練輪數（上限） | 300~600，GUI 預設 **500**；搭配 patience 早停，設大不會白跑 |
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

### 訓練跑完了，但模型幾乎抓不到東西

先看 `runs/<task>/<name>/labels.jpg`（各類別的實例數）與 `results.csv`。
`train/dfl_loss`、`train/box_loss` 接近 0 表示**正樣本幾乎不存在**，也就是 dataset
裡幾乎沒有有效標註 —— 通常是 class name 對不上 Class Mapping。訓練 log 裡的
`... images, N backgrounds, 0 corrupt` 那一行，`N` 不該接近圖片總數。

另一個常見原因是**總迭代數太少**：ultralytics 的 warmup 是
`nw = max(round(warmup_epochs * nb), 100)`（`nb` = 每輪的 batch 數），
有 100 次迭代的下限。資料量少又只跑十幾輪的話，整段訓練都還在 warmup，
學習率從沒離開起跑點。Train YOLO 在開始前會幫你算這個數字並提醒。

### 顯示的 mAP 很漂亮，實際拿去用卻很差

檢查 val 是不是和 train 重疊或高度相似：同一段影片的相鄰幀、同一張原圖的多個裁切
分到兩邊都會讓分數虛高。本工具轉檔時已經會依來源分組，但若整份資料只來自單一來源
（例如只標了一段影片），就無法分組，摘要視窗會特別提醒。

### Detection 和 Segmentation 的標籤格式可以混用嗎？

不行。Detection 模型需要 `cx cy w h` 格式，Segmentation 模型需要 polygon 格式。
轉換前請在 **Train → VOC to YOLO** 對話框內選對輸出模式。

### 訓練到一半可以接續嗎？

可以，請參考下一節 [再訓練 / 繼續訓練](#再訓練--繼續訓練)。

---

## 再訓練 / 繼續訓練

訓練後（無論是因為中斷、想加 epoch、還是想對其他資料集 fine-tune）都可以拿既有的 `.pt` 接續，分成兩種模式：

| 模式 | 用什麼 .pt | 說明 | 適用情境 |
|------|-----------|------|---------|
| **Resume**（接續同一次訓練） | `last.pt` | 從原訓練中斷的 epoch 繼續，optimizer / scheduler / lr schedule / 增強參數全部沿用原訓練的 `args.yaml` | 訓練被中斷（電腦關機、Ctrl-C），想無痛接續到原本設定的最後一個 epoch |
| **Fine-tune**（用權重做新訓練） | `last.pt` 或 `best.pt`（建議 best） | 把這個 `.pt` 當成「新訓練的初始權重」，依照當前對話框的所有參數開新一輪訓練（會建立新的 `runs/<task>/<name>/` 資料夾） | 想加更多 epoch、想換 dataset、想用更小的 lr 收尾、想換增強策略 |

### 在 GUI 內操作

1. 開啟 **Train → Train YOLO**
2. 在 **Resume from .pt** 欄位按「瀏覽...」選擇之前的 `.pt`：
   - Resume 模式 → 選 `runs/<task>/<name>/weights/last.pt`
   - Fine-tune 模式 → 選 `runs/<task>/<name>/weights/best.pt`（也可以選 `last.pt`）
3. 想要 Resume，再勾選「**Resume mode**」checkbox；想要 Fine-tune 就**不要勾**
4. 設定 `dataset.yaml`：
   - Resume 模式：dataset 結構不能變（class 數、`train`/`val` 路徑要一致）
   - Fine-tune 模式：可以是同一個或全新的 dataset
5. 調整 Epochs / Batch / 等參數後按「**開始訓練**」

> 指定 `.pt` 後，下方的 Task / Model Size / Version 會自動鎖住——這些屬性由 `.pt` 內部決定，不需要也不能在這裡覆寫。
>
> 「最終使用模型」提示列會顯示目前用的是哪一顆 `.pt` 以及處於 Resume 或 Fine-tune 模式。

### Ultralytics 對 Resume 的規則

- Resume 要求 `last.pt` **同層的 `weights/` 上一層** 有 ultralytics 自動產出的 `args.yaml`，沒有的話 ultralytics 會直接報錯。
- Resume 後整份參數都以 `args.yaml` 為準，只有 ultralytics 白名單內的幾項能覆寫：`imgsz` / `batch` / `device` / `patience` / `save_period` / `workers` / `cache` / `close_mosaic` / `freeze` / `val` / `plots`。
- **`epochs` 不在白名單**，Name 與所有 optimizer / lr / 增強參數也不在，填了會被丟掉；因此對話框在勾選 Resume 時會把這些欄位鎖住。想加訓練輪數或改增強請用 Fine-tune 模式。
- 如果 dataset 結構變了（例如多/少 class、改路徑），Resume 會失敗，請改用 Fine-tune。

### 用 Python 腳本接續

```python
from ultralytics import YOLO

# Resume：從 last.pt 接續，args.yaml 帶回原訓練的所有設定
model = YOLO("runs/detect/train/weights/last.pt")
results = model.train(resume=True)

# Fine-tune：用 best.pt 當初始權重，自由換 dataset / epoch / lr
model = YOLO("runs/detect/train/weights/best.pt")
results = model.train(
    data="path/to/new_or_same_dataset.yaml",
    epochs=200,
    lr0=0.001,   # 通常 fine-tune 會降低 lr
    imgsz=640,
)
```
