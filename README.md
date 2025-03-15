本專案可對影像, 影片畫框  
用於object detection的影像訓練dataset  
可使用yolo model來自動偵測, 或是表現VOC的框的資訊

![system gui](./asset/system_gui.png)

功能一覽
- 滾輪可預覽上/下一個檔案
- 預設label種類名稱[設定檔](./config/static.yaml), 按下數字鍵可以直接切換label名稱, 或者按下小寫的`L`來設定指定名稱
  以減少輸入次數
- 關於影片撥放
  - 讀取影片檔案, 並且labeling
  - 使用空白鍵play/pause
  - 有影片播放進度條
  - 可調整速度
  - 按下滑鼠鍵會停止撥放
  - 如果自動儲存開啟, 則按下play後也會自動儲存, 儲存的檔名將會是"[原檔名]_frame[N], 並且圖片跟xml都會抽出
  - 每N秒儲存一筆
- 狀態欄
  - 顯示有多少檔案, 目前在第N個檔案
- 快捷鍵
  - `q`: quit
  - `a`: toggle auto save
  - `l`: 彈出視窗, 輸入label名
  - `數字鍵0~9`: 切換預設的label, 只會針對最後一個label做變更
  - `Page Up/Down` or `方向鍵左/右`: 切換檔案
  - `Home/End`: 切換到最 前/後 檔案
  - `space空白鍵`: play/pause video

## 安裝相關
安裝pytorch的延伸package, 都一定要先裝pytorch的cuda版本; 不然幾乎都是先自動安裝cpu版的, 所以會跑很慢

windows可使用pip來安裝pytorch CUDA版, 詳細可看  
https://pytorch.org/get-started/locally/  
選一個比你電腦的CUDA版本還低的pytorch就行  
**NOTE: 安裝前把所有pytorch的package都清乾淨, 尤其是 torchvision 很容易被遺忘, 沒清乾淨直接裝的話版本配不上一定出錯**
```bash
# 範例CUDA 12.4
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
之後安裝requirements即可
```bash
pip install -r requirements.txt
```

## 雜談
PyQt最棒的地方在於, 畫面上可以直接打中文

### 關於video播放
Google AI Gemini-2.0-pro 跟我都試過了, 沒有辦法把video widget的frame傳到畫布中編輯  
因此用傳統的方式來把opencv frame轉成pixmap  
