# 使用教學

## 啟動程式

```bash
python main.py
```

## 基本操作流程

```
開啟資料夾 → 偵測/畫框 → 調整標註 → 儲存 → 轉換格式 → 訓練
```

---

## 開啟檔案

- **File → Open Folder**：開啟一個含有圖片或影片的資料夾
- **File → Open File By Index**：跳到該資料夾中的第 N 個檔案
- **滾輪** 或 **PgUp/PgDn**：瀏覽上/下一個檔案
- **Home/End**：跳到第一個/最後一個檔案

> 狀態欄會顯示目前在第幾個檔案，以及總檔案數。

---

## AI 自動偵測

### YOLO 偵測

1. **快捷鍵 `d`** 或 **Ai → Detect**：對目前的影像執行偵測（首次會自動下載 `yolo26s.pt` 預設模型）
2. **Ai → Select YOLO Model**：選擇自定義的 `.pt` 模型（必須是 Ultralytics 相容的模型）
3. **Ai → Auto Detect**：開啟後，切換檔案時自動偵測

> 如果圖片旁已有同名的 `.xml` 標籤檔且內含 bbox，Auto Detect 不會覆蓋，會優先使用 XML 的標註。

### SAM3 語義分割

由於SAM3是需要申請下載權限, 因此請到  
https://huggingface.co/facebook/sam3  
有獲得授權後就能下載 `sam3.pt`

> 需要先在 `cfg/system.yaml` 中設定 `enable_sam3: true`

1. **Ai → Select SAM3 Model**：選擇 SAM3 的 `.pt` 模型檔
2. **Ai → Use SAM3**：切換為 SAM3 模式
3. **Ai → Edit Text Prompts**：設定要偵測的類別名稱（如 person、cat、dog）
4. **Ai → Detect**：執行 SAM3 推論

SAM3 會根據文字描述自動產生 segmentation mask 並轉為 polygon。
在 `cfg/settings.yaml` 中可設定輸出模式：

| `sam3_label_mode` | 行為 |
|---|---|
| `seg` | 只產生 polygon（建議用於 segment 訓練） |
| `bbox` | 只產生 bounding box |
| `all` | 同時產生 polygon 和 bbox |

> 訓練 segment 時建議用 `seg` 模式，避免 `all` 模式在轉換時產生重複標註。

---

## 手動標註

### BBox 模式（快捷鍵 `b`）

- **左鍵拖曳**：畫出矩形框
- 畫好後，框的角落可以拖曳調整大小
- Focus 中的框會暫時變黃色
- **右鍵** 或 **Esc**：取消正在繪製的框；右鍵也可刪除滑鼠位置下的框（後畫的優先刪除）

### Polygon 模式（快捷鍵 `p`）

- **左鍵點擊**：新增頂點
- 靠近第一個頂點時自動封閉多邊形
- **右鍵** 或 **Esc**：繪製中取消當前多邊形；右鍵也可刪除已有的多邊形

### Select 模式（快捷鍵 `v`）

- 用於選取、移動已有的標註
- 右鍵功能與對應模式相同

### Mask 工具

> 需要先在 `cfg/system.yaml` 中設定 `enable_mask_tools: true`

- **Draw**：筆刷繪製遮罩
- **Erase**：擦除遮罩
- **Fill**：填充封閉區域
- 筆刷大小可透過滑桿調整（1-100 px）

---

## Label 管理

- **`L` 鍵**：彈出視窗，輸入自定義 label 名稱（只影響最後 focus 的框）
- **數字鍵**：快速套用預設 label（在 `cfg/system.yaml` 的 `labels` 區段設定）
  - 支援多碼輸入（如 `1`、`12`、`111`），可對應超過 10 個類別
  - 若輸入的碼唯一且無歧義，會立即套用
  - 若碼是其他更長碼的前綴（如 `1` 和 `12` 同時存在），會等待 `label_key_timeout` 秒後套用
  - 輸入過程中，狀態欄會顯示目前已輸入的碼

### Labels 快捷鍵設定

在 `cfg/system.yaml` 的 `labels` 區段以 `數字碼: 標籤名稱` 的格式定義，例如：

```yaml
labels:
  # --- 1x 哺乳類(大型) ---
  1: deer
  11: bear
  12: boar
  # --- 2x 哺乳類(中型) ---
  2: monkey
  21: dog
  # --- 0 未分類 ---
  0: unknown
```

- **編碼規則**：首碼代表大類，次碼代表子類，建議以樹狀結構分組
- **前綴衝突**：若 `1` 和 `12` 同時存在，按下 `1` 後系統會等待 `label_key_timeout`（預設 2 秒）再套用，讓你有時間繼續輸入 `12`；而 `12` 因為沒有更長的前綴碼，會立即套用
- **`label_key_timeout`**：在 `cfg/system.yaml` 中可調整等待秒數，熟練後建議降低以加快操作速度

---

## 儲存

- **File → Save**（快捷鍵 `s`）：將標註儲存為 VOC XML 格式
- **File → Auto Save**（快捷鍵 `a`）：切換檔案時自動儲存

> 儲存的圖片和 XML 會放在原始資料夾下的 `output` 子資料夾，避免與原始檔案混淆。
> 如果 Mask 工具有啟用，mask 會另存為 `{檔名}_mask.png`。

---

## VOC → YOLO 轉換

1. **Train → VOC to YOLO**：選擇含 VOC XML 的資料夾後，會彈出設定對話框：

   - **Class Mapping**：在對話框內按「編輯 Mapping」設定 class name 到數字編號的對應，例如 `person → 0`、`cat → 1`、`dog → 2`。同一個物件的不同名稱可以對到同一個編號（如 `motor`、`motorbike`、`motorcycle` 都對到 `9`）。

     ![categories](../asset/categories.png)

   - **輸出模式**：

     | 模式 | YOLO 格式 | 用途 |
     |------|-----------|------|
     | BBox | `class_id cx cy w h` | Object Detection 訓練 |
     | Seg | `class_id x1 y1 x2 y2 ... xN yN` | Segmentation 訓練 |
     | OBB | `class_id x1 y1 x2 y2 x3 y3 x4 y4` | 旋轉框偵測（目前停用） |

   - **Train / Val 比例**：預設 80%/20%，可調整（Train 最少 50%）；若設為 100% 不產生 val set，產出的 `dataset.yaml` 仍會把 `val` 退回指向 `train` 以滿足 ultralytics 規範

   轉換完成後，工具會自動：
   - 將圖片和標籤依比例移動到 `images/train`、`images/val` 和 `labels/train`、`labels/val`
   - 在資料夾根目錄產生 `dataset_YYYY_MMDD_HHMMSS.yaml`，可直接用於 Ultralytics 訓練

> 所有座標值都是正規化（0~1）的相對座標。

---

## Train YOLO（GUI 內訓練）

**Train → Train YOLO** 直接在 GUI 內呼叫 ultralytics 訓練，不必再切到終端機跑 Python script。

### 基本流程

1. 選擇 `dataset.yaml`（會自動帶上次用過的；或從上一步 VOC → YOLO 產出的資料夾自動搜尋最新的 `dataset_*.yaml`）
2. 設定 **Task**（Detect / Segment）、**Model Size**（n/s/m/l/x）、**Version**（預設 `yolo26`）
   - 對話框下方會顯示組合出的最終模型檔名，例如 `yolo26s.pt` 或 `yolo26m-seg.pt`，本地不存在時 ultralytics 會自動下載
3. 設定基本訓練參數：

   | 參數 | 說明 |
   |------|------|
   | Epochs | 最大訓練輪數，搭配 Patience 提前停止 |
   | Batch | 每批次圖片數；`-1` = 自動偵測最大 batch |
   | Image Size | 輸入解析度（px），常見 320 / 640 / 1280 |
   | Patience | 連續 N epoch mAP 無改善則停止；0 = 關閉 |
   | Device | `0` = 第一張 GPU；`cpu`；`0,1` = 多卡 |
   | Save Period | 每 N epoch 額外存 checkpoint；-1 = 關閉 |
   | Name | 輸出資料夾名（runs/<task>/<name>/），留空自動帶時間戳 |

4. 按「**開始訓練**」後狀態列會顯示 `Epoch X/Y  mAP50=…`，並在 epoch 結束時更新進度條
5. 訓練中可按「**停止**」優雅地在當前 epoch 結束後中止
6. 訓練完成後下方顯示輸出資料夾、訓練時間、Box / Seg 的 mAP@0.5、mAP@0.5:0.95
7. 「**開啟訓練資料夾**」會用檔案總管打開 `runs/<task>/<name>/`（含 best.pt、last.pt、訓練曲線圖）

### 進階參數

按「**進階參數...**」開啟分頁式對話框，可細調以下類別：

| 分頁 | 內容 |
|------|------|
| 優化器 | optimizer / lr0 / lrf / weight_decay / warmup_epochs / warmup_momentum |
| 幾何 / 翻轉 | degrees / translate / scale / perspective / flipud / fliplr |
| 色彩 | hsv_h / hsv_s / hsv_v |
| 混合增強 | mosaic / close_mosaic / mixup / copy_paste |
| 系統 | workers / cache (false/ram/disk) / rect / amp / fraction / freeze |

> 所有基本與進階參數都會暫存在 `cfg/settings.yaml` 的 `training` 區段，下次再開直接帶回上次的值。
> 各參數意義可懸停滑鼠看 tooltip，或參考 [Ultralytics Train 官方文件](https://docs.ultralytics.com/modes/train/)。

---

## 影片標註

- 開啟含影片的資料夾後，程式會自動辨識影片格式
- **Space**：Play / Pause
- 有播放進度條，可拖曳跳轉
- 按下滑鼠鍵會暫停播放
- 開啟 Auto Save 後，播放期間會自動抽幀儲存，檔名為 `{原檔名}_frame{N}`
- 在 `cfg/system.yaml` 的 `auto_save_per_second` 可設定每幾秒儲存一幀（`-1` 關閉）
