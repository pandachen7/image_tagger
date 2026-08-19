# 更新記錄

2026/8
- 新增 **縮放與平移**：滾輪以游標為錨點縮放、中鍵/右鍵拖曳平移、`f` 還原檢視、狀態列常駐顯示倍率
  - 原本的「滾輪翻檔」移到 **Ctrl+滾輪**，熟手的肌肉記憶不會白費；`PgUp/PgDn` 不變
  - 換到**尺寸相同**的影像會保留縮放與位置 —— 逐張比對同一個區域是這工具的主要用法，每張都跳回原狀會讓人重新找一次位置
  - 所有 原圖 <-> 螢幕 的換算集中到 `ViewTransform`（[src/utils/view_transform.py](../src/utils/view_transform.py)），不在別處自行乘 zoom 或加 offset —— 那是這類工具座標漂移的主要來源
  - 進行中的繪製（BBOX 兩點、Polygon 頂點）改存**原圖座標**：先前存的是 widget 座標，畫到一半縮放就會整組錯位。`start_pos` 原本在 resize 路徑存原圖、在 BBOX 路徑存 widget，同一個欄位兩種座標系，已拆成 `draw_start` / `draw_end`
  - 繪製效能：縮小時（一般檢視狀態）用預縮好的 pixmap，平移只是 blit；放大時只畫可見區域。兩者都不重新解碼原圖
- **Select 模式重整**
  - **拖標註內部 = 整體移動**（不變形）。先前點框內只能選取，改大小只能拖四個角
  - **多選後拖任一個已選取的標註 → 整批一起移動**，相對位置不變、選取狀態不被打散；整批移動只算一步 `Ctrl+Z`
  - 移動時碰到影像邊緣就停住；多選是**整組一起夾**，不會出現「碰到邊界的那幾個停住、其他繼續走」而把相對位置弄散
  - **控制點的熱區限縮到選取中的框**：控制點本來就只畫給選取中的框，熱區跟著限縮才不會「看不到卻踩得到」—— 框重疊時最容易誤把旁邊那個拖變形
  - **右鍵在畫布上只負責平移**。先前它負責「刪除滑鼠下的標註」與「取消進行中的繪製」：刪除沒有確認、又與取消繪製共用同一個鍵，太容易誤刪；取消繪製則是畫多邊形畫到一半誤點右鍵就整個重來，代價太大。現在刪除一律走選取後 `Delete`（而且救得回來），取消繪製一律走 `Esc`
  - `Esc` 在沒有進行中的繪製時改為取消選取
- 新增 **Undo / Redo**（`Ctrl+Z` / `Ctrl+Shift+Z`、`Ctrl+Y`，選單 **Edit**）：畫框、畫多邊形、移動、resize、旋轉、拖曳多邊形頂點、刪除（含多選一次刪多個）、改 label，以及**執行偵測**都可還原
  - 偵測 (`d`) 會把現有標註整批換掉，先前誤按就救不回手工標的框，現在納入 undo
  - 快照式而非命令式：一張圖的標註只有數十個小物件，複製成本可忽略；resize / 旋轉 / 頂點拖曳這類連續操作用命令物件很容易漏記反向狀態
  - 歷史屬於**目前這張影像**，換檔或影片換幀時歸零 —— 換檔會重讀該檔的 `.xml`，留著上一張的快照會讓 undo 把別張圖的框寫進這一張
  - 步數上限為 `cfg/system.yaml` 的 `undo_limit`（預設 60）
  - 快照存純資料而不含 `QPen`：`color_pen` 是「選取中 / 拖曳中」的顯示狀態而非標註內容，一併還原會把拖曳中的黃色也留下來
  - 沒有真的改到東西不佔步數：點一下選取沒拖動、點到控制點沒拖動、拖出去又拖回原位、框小於 `minimal_bbox_length`、套用相同 label，都不會留下按了沒反應的空步驟
- 新增 **Undo / Redo**（`Ctrl+Z` / `Ctrl+Shift+Z`、`Ctrl+Y`，選單 **Edit**）：畫框、畫多邊形、resize、旋轉、拖曳多邊形頂點、刪除（含多選一次刪多個）、改 label，以及**執行偵測**都可還原
  - 偵測 (`d`) 會把現有標註整批換掉，先前誤按就救不回手工標的框，現在納入 undo
  - 快照式而非命令式：一張圖的標註只有數十個小物件，複製成本可忽略；resize / 旋轉 / 頂點拖曳這類連續操作用命令物件很容易漏記反向狀態
  - 歷史屬於**目前這張影像**，換檔或影片換幀時歸零 —— 換檔會重讀該檔的 `.xml`，留著上一張的快照會讓 undo 把別張圖的框寫進這一張
  - 步數上限為 `cfg/system.yaml` 的 `undo_limit`（預設 60）
  - 快照存純資料而不含 `QPen`：`color_pen` 是「選取中 / 拖曳中」的顯示狀態而非標註內容，一併還原會把拖曳中的黃色也留下來
  - 沒有真的改到東西不佔步數：點到控制點沒拖動、框小於 `minimal_bbox_length`、套用相同 label 都不會留下按了沒反應的空步驟
- **資安更新**：`torch` 2.12.1 → **2.13.0+cu130**（CVE-2025-3000, Low）、`setuptools` 81.0.0 → **83.0.0**（CVE-2026-59890, Moderate）
  - `torchvision` 一併升到 **0.28.0+cu130**：它硬性要求 `torch==2.13.0`，無法只升 torch
  - torch / torchvision 因相容性反覆出過問題，`pyproject.toml` 改用 `==` **固定版本**，升級時務必兩者一起改
  - `setuptools` 僅為 torch 的間接相依，用 `[tool.uv] constraint-dependencies` 拉下限，不宣告成直接相依
- **本專案改為需要 uv >= 0.12，並建議定期 `uv self update`**
  - PyTorch cu130 index 未發布 `torchvision` 0.28.0 全部 Windows wheel 的 `#sha256`（Linux wheel 正常）；uv <= 0.7.x 會退而比對其他平台的 hash 而必然失敗，報出誤導性的 `Hash mismatch`，uv 0.12.1 才正常
  - 教訓：`uv sync` 出現無法理解的相依 / hash 錯誤時，先更新 uv 再試，通常比改 `pyproject.toml` 有效

2026/7
- 新增 **Cropped 裁切儲存模式**（選單 `Edit` 更名為 `Label`，透過 **Label → Label Mode…** 切換）：只裁切畫面上有框 (bbox / polygon) 的區域，各自存成小圖 + VOC XML，適合對動態區 / ROI 過濾後的目標做放大裁切，提升小物件的 YOLO 偵測訓練效果
  - 尺寸二選一：**固定外擴 padding (px)** 或 **至少固定尺寸 (px，對齊 YOLO 輸入如 640)**；某邊碰到影像邊緣沒有像素時，會往對邊補足像素以維持尺寸
  - 相鄰、能落在同一裁切區內的多個框會**合併**成一張（含多個標籤）；沒有任何框則不儲存
  - 輸出檔名為 `{原檔名}_crop{N}.jpg` / `.xml`（影片再加 `_frame{N}`），產出的 VOC XML 與現有「VOC to YOLO」流程完全相容
- 手動框選 / 繪製的最小標註尺寸 (`minimal_bbox_length`) 改以**原始影像像素**判斷，不再受顯示縮放影響：4K 影像縮小顯示時也能畫出小框，且與偵測結果過濾、存檔座標基準一致
- 繪製 bbox 拖曳過程即時顯示「寬x高=面積」(原圖座標)；bbox 選取的面積標示移到右下角點的右上，避免拉到畫面底部被截斷
- **Select** 調整 bbox 的四角方塊縮小 (邊長 20→10 px)，且點擊命中範圍改以螢幕像素判斷、與方塊大小一致，不再隨影像縮放對不上
- 修正按 **Detect** / 訓練時在子執行緒首次載入模型 (import ultralytics) 導致的 native crash（程式無聲跳出）：改為主執行緒同步載入 / 啟動訓練前先於主執行緒 import
- 修正播放無法同時偵測的問題
- lib (python pkg) 配置升級, 並且以後都以uv為主

2026/5
- Train YOLO 對話框新增 **Resume from .pt** 欄位：可選擇之前訓練的 `last.pt` / `best.pt` 接續訓練；勾選「Resume mode」時走 ultralytics `resume=True` 從原 epoch 接續，不勾選則以該權重做 fine-tune（細節見 [訓練指南 → 再訓練 / 繼續訓練](./training.md#再訓練--繼續訓練)）

2026/4
- 選單 `Convert` 更名為 `Train`，新增 **Train → Train YOLO**：可選 dataset.yaml、Task / Model Size / Version / 訓練參數，內建進度條與訓練結果摘要，並可開啟訓練資料夾
- Train YOLO 對話框新增「進階參數...」按鈕：優化器 (lr0/lrf/weight_decay/warmup) / 幾何 / HSV / Mosaic+MixUp / 系統 (workers/cache/amp/freeze...) 全套參數，全部暫存到 `cfg/settings.yaml` 的 `training` 區段
- VOC → YOLO 預設改為 Train 80% / Val 20%；產生的 `dataset.yaml` 一定包含 `train` 與 `val` 兩個 key（無 val split 時 val 退回指向 train）
- system.yaml可設定各種數字對應的class_name, 並且可在system.yaml設定短編碼的反應時間
- VOC轉yolo格式時, 可在選定資料夾後選擇轉換的方式, 例如train/val的比例
- SAM3 影片 frame bug 修正 — infer_sam3 改為接收 cv_img (numpy array)，不再傳檔案路徑。這樣影片的每一幀都能正確被 SAM3 偵測。
- Ai → SAM3 Output Mode... dialog — 可在 seg / bbox / all 三種模式間切換，設定會存入 settings.yaml。
- VOC → YOLO 轉換增加進度條、未對應 class_name 記錄檔 (not_match_*.txt)、轉換完成摘要對話框
- Ai → Categorize Media — 選擇資料夾與 model，自動偵測每個圖片/影片中出現最多次的物件，依名稱分類到子資料夾。支援 YOLO / YOLO-Seg / SAM3 三種模型，可自動偵測模型類型。
- Convert → VOC to YOLO 對話框整合 Class Mapping（原 Edit Categories）與資料夾選擇，顯示圖片數量
- Ai → Set YOLO Model / Set SAM3 Model 取代原本的 Select Model 選單；SAM3 dialog 整合 Output Mode、Polygon Tolerance、Text Prompts
- YOLO seg model（如 yolo26m-seg.pt）支援 bbox / seg / all 輸出模式與獨立的 Polygon Tolerance 設定
- 模型切換後背景非同步載入，不阻塞 UI；首次使用官方模型名稱時自動下載
