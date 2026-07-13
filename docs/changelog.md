# 更新記錄

2026/7
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
