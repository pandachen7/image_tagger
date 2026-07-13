# Image Tagger

對影像、影片畫框 (BBox / Polygon)，產出 object detection 或 segmentation 的訓練 dataset。
支援 Ultralytics YOLO / SAM3 自動偵測，並可將 VOC 標註轉為 YOLO 格式。

![system gui](./asset/system_gui.png)

## 安裝環境與啟動

> 詳細安裝步驟請見 [安裝指南](./docs/installation.md)

**建議使用 [uv](https://docs.astral.sh/uv/)**：安裝套件快，且會依 `pyproject.toml` 自動裝好含 **CUDA 13.0** 的 PyTorch。

```bash
# 建立環境並安裝所有相依（含 CUDA 13.0 的 PyTorch）
uv sync

# 啟動
uv run main.py
```

> **Fallback（不使用 uv 時）**：改用 Python 內建 venv + pip。注意 PyTorch CUDA 版要自行從 cu130 index 安裝，pip 不會讀取 `pyproject.toml` 裡的 uv index 設定。
>
> ```bash
> python -m venv .venv
> .venv\Scripts\Activate.ps1                                                    # Windows PowerShell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
> pip install .
> python main.py
> ```

## 基本使用流程

```
開啟資料夾 → AI 自動偵測 (或手動畫框) → 儲存 VOC XML → 轉成 YOLO 格式 → 訓練模型
```

1. **File → Open Folder** 開啟含有圖片的資料夾
2. **Ai → Set YOLO Model** 設定模型路徑（首次會自動下載 `yolo26s.pt` 預設模型）；如需使用 SAM3，透過 **Ai → Set SAM3 Model** 一併設定模型、輸出模式、Polygon Tolerance 與 Text Prompts
3. **按 `d` 或 Ai → Detect** 偵測物件
4. 手動微調框的位置、名稱後，**File → Save** 或按 `s` 儲存為 VOC XML
5. **Train → VOC to YOLO** 在對話框中設定 Class Mapping（class_name → class_id）、選擇資料夾、輸出模式與 train/val 比例（預設 80/20）→ 自動轉換並產生 `dataset.yaml`
6. **Train → Train YOLO** 直接在 GUI 內訓練：選擇 `dataset.yaml`、Task（Detect / Segment）、Model Size、訓練參數，啟動後顯示 epoch 進度與 mAP，完成後可開啟訓練資料夾。也可指定既有 `.pt` 接續訓練（Resume / Fine-tune）

> 每個步驟的詳細操作說明請見 [使用教學](./docs/usage.md)
> 訓練相關（dataset 結構、data.yaml、segment 訓練）請見 [訓練指南](./docs/training.md)

## 功能總覽

| 功能 | 說明 |
|------|------|
| YOLO 自動偵測 | 載入 `.pt` 模型，支援 detect 與 seg model；seg model 可輸出 bbox / polygon / all，透過 Set YOLO Model 設定 |
| SAM3 語義分割 | 透過文字描述自動產生 polygon / bbox；Set SAM3 Model 整合模型、輸出模式、tolerance、prompts |
| 手動 BBox | 左鍵拖曳畫框，角落可調整大小 |
| 手動 Polygon | 左鍵點擊頂點，靠近起點自動封閉 |
| Mask 工具 | Draw / Erase / Fill 遮罩繪製，但訓練不需要 |
| VOC → YOLO 轉換 | 支援 BBox、Seg、OBB 三種輸出格式，轉換進度條、未對應 class 記錄 |
| Train YOLO (GUI) | 直接在 GUI 內呼叫 ultralytics 訓練，可設定基本參數與進階參數（優化器 / 增強 / cache 等），訓練中顯示進度與 mAP；支援指定既有 `.pt` 做 Resume / Fine-tune 再訓練 |
| Categorize Media | 用 YOLO/SAM3 模型偵測後，依最多次物件名稱自動分類到子資料夾 |
| 影片標註 | 逐幀標註，支援自動抽幀儲存 |

## 快捷鍵

| 按鍵 | 功能 |
|------|------|
| `q` | 離開 |
| `s` | 儲存 |
| `a` | 切換 Auto Save |
| `d` | 執行偵測 (Detect) |
| `l` | 編輯選取框的 label 名稱 |
| `v` | Select 選取模式 |
| `b` | BBox 繪製模式 |
| `p` | Polygon 繪製模式 |
| `數字鍵` | 快速切換預設 label（支援多碼，如 `12`、`111`） |
| `Esc` | 取消正在繪製的 BBox / Polygon |
| `PgUp/PgDn` 或 `←/→` | 上/下一個檔案 |
| `Home/End` | 第一個/最後一個檔案 |
| `Space` | 影片 Play/Pause |
| 滾輪 | 預覽上/下一個檔案 |

## 設定檔

| 檔案 | 用途 |
|------|------|
| `cfg/system.yaml` | 系統設定：預設 label、啟用 SAM3/Mask/OBB 等開關 |
| `cfg/settings.yaml` | 執行期設定：模型路徑、categories 對應、text prompts、訓練參數暫存 (training 區段). 不存在時會自動生成 |

## 常用vs code的快捷組合鍵

- ctrl + shift + `: 開新terminal(e.g. git bash, command prompt)
- ctrl + d: focus在terminal的話, 關閉目前的terminal
- f5: run python程式
- shift + f5: 關閉目前正在跑的python程式
- ctrl + shift + f5: 如果有正在跑的python程式, 則關掉並重跑
- ctrl + `+`: 放大文字
- ctrl + `-`: 縮小文字

## 更新

歷次版本更新與功能異動請見 [更新記錄](./docs/changelog.md)。

## 文件目錄

- [安裝指南](./docs/installation.md) — 環境建置、PyTorch CUDA、常見問題排除
- [使用教學](./docs/usage.md) — 各項功能的詳細操作方式
- [訓練指南](./docs/training.md) — 從標註到訓練 YOLO 模型的完整流程
- [更新記錄](./docs/changelog.md) — 版本更新與功能異動歷程
